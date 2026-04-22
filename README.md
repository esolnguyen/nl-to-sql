# nl_to_sql

A natural-language-to-SQL service. Given a question in English (or any language the underlying LLM speaks), it produces an executable SQL query against a connected relational database, runs it, and optionally generates a natural-language answer summarizing the result.

It is designed as a long-running backend service: schemas are scanned once and cached, example query/answer pairs and per-database instructions are stored to improve agent accuracy, and credentials are encrypted at rest. The agent runtime is built on LangChain and supports both general OpenAI models and OpenAI fine-tuned models trained on your own example SQLs.

---

## Features

- **NL → SQL generation** via a LangChain agent that has tools for: schema introspection, example-SQL retrieval (vector search), instruction lookup, and dry-run execution.
- **NL answer generation** optional follow-up step that summarises the query result in natural language.
- **Streaming endpoint** for incremental SQL generation over Server-Sent Events.
- **Multi-database support** Postgres, MySQL, MSSQL, BigQuery, Snowflake, Databricks, Redshift, ClickHouse, Athena, DuckDB, SQLite.
- **Schema scanning** pulls table/column metadata and sample values once, persists them, and exposes endpoints to re-scan or refresh.
- **Example SQLs (few-shot store)** example NL/SQL pairs are embedded into a vector store (Chroma by default; Pinecone and Astra also wired in) and retrieved at generation time.
- **Per-DB instructions** natural-language guidance ("`status = 'A'` means active") that is fed into every prompt.
- **OpenAI fine-tuning pipeline** turns your stored example SQLs into a JSONL dataset, kicks off a fine-tune job, polls status, and routes future generations to the resulting model when configured.
- **CSV export** for query results, optionally streamed to S3/MinIO.
- **Credential encryption at rest** connection URIs, SSH passwords, and per-DB LLM API keys are Fernet-encrypted before persistence.
- **Pluggable storage and DI** API server, metadata store, vector store, context store, and DB scanner are all selected via env vars and resolved by a tiny custom DI container.

---

## Architecture

```
                ┌──────────────────────────────────────┐
                │              FastAPI                 │
                │   (src/app/server/fastapi/__init__)  │
                └───────────────┬──────────────────────┘
                                │
                ┌───────────────▼──────────────────────┐
                │   API abstract class + NLToSQLAPI    │
                │       (src/app/api/…)                │
                └───────────────┬──────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────────────┐
        │                       │                               │
┌───────▼──────┐   ┌────────────▼────────────┐   ┌──────────────▼────────────┐
│  Services    │   │    LLM Agents           │   │    DB Scanner             │
│ sql_gen,     │   │ sql_generation_agent,   │   │ per-dialect scanners      │
│ nl_gen,      │◄──┤ finetuning_agent,       │   │ (postgres, snowflake,     │
│ db_conn,     │   │ nl_answer_agent,        │   │ bigquery, mssql, …)       │
│ prompt,      │   │ toolkit/                │   └───────────────────────────┘
│ context,     │   └─────────────────────────┘
│ finetuning   │
└──────┬───────┘
       │
┌──────▼───────────┐    ┌────────────────────┐    ┌──────────────────────────┐
│  Repositories    │    │   Vector store     │    │   Target SQL database    │
│  (Mongo-backed)  │    │   (Chroma /        │    │   (any dialect via       │
│                  │    │   Pinecone /       │    │   SQLAlchemy)            │
│                  │    │   Astra)           │    │                          │
└──────────────────┘    └────────────────────┘    └──────────────────────────┘
```

### Request flow "ask a question"

1. Client posts a `Prompt` (text + `db_connection_id`) to `/api/v1/prompts/sql-generations`.
2. `SQLGenerationService` looks up the connection, instantiates the right agent (fine-tuned vs general), pulls relevant example SQLs from the vector store and instructions from Mongo, and runs the LangChain agent loop.
3. The agent calls its tools schema lookup, similar-example search, dry-run query until it converges on a final SQL string.
4. The result is persisted as a `SQLGeneration`, returned to the client, and (optionally) executed via `/execute` or summarised via `/nl-generations`.

### Dependency injection

`Settings` (in `src/app/config.py`) pulls implementation FQNs out of environment variables (`API_SERVER`, `DB_SCANNER`, `NL_TO_SQL_DB`, `VECTOR_STORE`, `CONTEXT_STORE`). `System.instance(SomeAbstractClass)` walks the `_abstract_type_keys` map, imports the configured implementation, instantiates it once, and caches it. This is what allows you to swap Mongo for another metadata store or Chroma for Pinecone without touching call sites.

---

## Quick start (Docker)

```bash
cp .env.example .env

# Generate a Fernet key for ENCRYPT_KEY:
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
# paste it into .env as ENCRYPT_KEY=...

# Set OPENAI_API_KEY and any vector-store credentials you plan to use.

docker network create nl_to_sql_network   # one-time
docker compose up --build
```

The API is exposed on `http://localhost:${CORE_PORT:-80}`. Interactive OpenAPI docs at `/docs`, ReDoc at `/redoc`, heartbeat at `/api/v1/heartbeat`.

### Running locally without Docker

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$(pwd)/src
uvicorn main:app --host 0.0.0.0 --port 8000 --reload \
    --app-dir src --log-config log_config.yml
```

You'll need a reachable Mongo instance pointed at by `MONGODB_URI`.

---

## Configuration

All configuration is read from environment variables (`.env` is auto-loaded by `python-dotenv`). See `.env.example` for the full list.

### Required

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | LLM access for SQL/NL generation and fine-tuning |
| `ENCRYPT_KEY` | Fernet key used to encrypt connection URIs and credentials at rest |
| `MONGODB_URI` | Metadata + history store |
| `MONGODB_DB_NAME` | Mongo database name |

### Dependency injection (importable Python paths)

| Variable | Default |
|---|---|
| `API_SERVER` | `app.api.nl_to_sql_api.NLToSQLAPI` |
| `NL_TO_SQL_DB` | `app.databases.mongodb.mongo.MongoDB` |
| `VECTOR_STORE` | `app.databases.vector.chroma.Chroma` |
| `CONTEXT_STORE` | `app.services.context_store.context_store.ContextStoreService` |
| `DB_SCANNER` | `app.services.db_scanner.SqlAlchemyScannerService` |

### Optional

| Variable | Purpose |
|---|---|
| `AGENT_MAX_ITERATIONS` | Max tool-use iterations per agent run (default 15) |
| `DH_ENGINE_TIMEOUT` | Whole-request timeout in seconds (default 150) |
| `SQL_EXECUTION_TIMEOUT` | Per-query execution timeout (default 3) |
| `UPPER_LIMIT_QUERY_RETURN_ROWS` | Row cap for executed queries (default 50) |
| `EXAMPLE_SQL_COLLECTION` | Vector-store collection name for example SQLs |
| `PINECONE_API_KEY`, `PINECONE_ENVIRONMENT` | Only if `VECTOR_STORE` points at Pinecone |
| `ASTRA_DB_API_ENDPOINT`, `ASTRA_DB_APPLICATION_TOKEN` | Only for Astra vector store |
| `S3_AWS_ACCESS_KEY_ID`, `S3_AWS_SECRET_ACCESS_KEY`, `S3_BUCKET_NAME`, `S3_REGION`, `S3_CUSTOM_ENDPOINT` | CSV export to S3 / MinIO |
| `ONLY_STORE_CSV_FILES_LOCALLY` | If true, skip S3 and keep exports on the container's filesystem |
| `AZURE_API_KEY`, `AZURE_API_VERSION`, `EMBEDDING_MODEL` | Azure-hosted embeddings |

---

## API surface

All routes are under `/api/v1`. A non-exhaustive map:

**Database connections** `POST/GET/PUT /database-connections[/{id}]`
**Table descriptions / schema scan** `POST /table-descriptions/sync-schemas`, `POST /table-descriptions/refresh`, `GET/PUT /table-descriptions[/{id}]`
**Prompts** `POST/GET/PUT /prompts[/{id}]`
**SQL generation** `POST /prompts/{prompt_id}/sql-generations`, `POST /prompts/sql-generations` (one-shot prompt + SQL), `GET/PUT /sql-generations[/{id}]`, `GET /sql-generations/{id}/execute`, `GET /sql-generations/{id}/csv-file`
**NL generation** `POST /sql-generations/{id}/nl-generations`, `POST /prompts/{id}/sql-generations/nl-generations`, `POST /prompts/sql-generations/nl-generations`, `GET/PUT /nl-generations[/{id}]`
**Streaming** `POST /stream-sql-generation` (Server-Sent Events)
**Example SQLs** `POST/GET/PUT/DELETE /example-sqls[/{id}]`
**Instructions** `POST/GET/PUT/DELETE /instructions[/{id}]`
**Fine-tuning** `POST/GET/PUT/DELETE /finetunings[/{id}]`, `POST /finetunings/{id}/cancel`
**Query history** `GET /query-history`
**System** `GET /heartbeat`

The full schema is browseable at `/docs` once the server is running.

---

## Repository layout

```
src/
  main.py                                uvicorn entry point
  app/
    config.py                            Settings + System (DI container)
    api/
      __init__.py                        API abstract class
      nl_to_sql_api.py                   concrete API implementation
      types/                             pydantic request/response models
    server/
      __init__.py                        NlToSQLServer abstract class
      fastapi/__init__.py                FastAPI route wiring
    databases/
      mongodb/                           NlToSQLDatabase abstract + MongoDB impl
      vector/                            VectorDatabase abstract + Chroma impl
      sql_database.py                    SQLAlchemy engine + safety wrappers
    models/                              pydantic domain models
    repositories/                        Mongo-backed persistence per entity
    services/
      sql_generation.py                  generate / execute / export SQL
      nl_generation.py                   summarise SQL results
      database_connection.py             connection lifecycle, encryption
      prompt.py                          prompt persistence
      context_store/                     example-SQL embedding + retrieval
      finetuning/openai_finetuning.py    dataset build + OpenAI fine-tune driver
      db_scanner/                        SQLAlchemy schema scanner
        types/                             per-dialect scanner overrides
    llm_agents/
      sql_generation_agent.py            LangChain agent for SQL gen
      finetuning_agent.py                agent that uses a fine-tuned model
      nl_answer_agent.py                 SQL-result → NL answer
      large_language_model/              LLM wrappers (chat, base)
      toolkit/                           tools exposed to the agents
    constants/                           prompt templates, model contexts, error map
    utils/
      encrypt.py                         Fernet wrapper
      sql_utils.py                       schema filtering + finetuning validation
      custom_error.py                    typed errors + HTTP error mapping
      aws_s3.py                          S3 / MinIO uploads
      timeout.py                         query-timeout helper
      strings.py
Dockerfile
docker-compose.yml                       engine + mongodb + postgres
log_config.yml
requirements.txt
```

---

## Typical workflow

1. **Create a database connection** (`POST /api/v1/database-connections`) with a SQLAlchemy URI; URI and credentials are encrypted before storage.
2. **Sync schemas** (`POST /api/v1/table-descriptions/sync-schemas`) kicks off a background scan that populates table/column metadata.
3. **(Optional) seed examples** (`POST /api/v1/example-sqls`) and **instructions** (`POST /api/v1/instructions`) so the agent has few-shot context.
4. **(Optional) fine-tune** `POST /api/v1/finetunings` builds a dataset from your example SQLs and kicks off an OpenAI fine-tune job. New SQL generations against that DB will automatically use the fine-tuned model when present.
5. **Ask questions** `POST /api/v1/prompts/sql-generations` with `{prompt: {text, db_connection_id}, ...}` to get back generated SQL. Hit `/execute` to run it, or `/nl-generations` to get a natural-language answer back.
6. **Stream** for interactive UIs, `POST /api/v1/stream-sql-generation` returns SSE chunks as the agent generates.

---

## Tests

```bash
pytest
```

(There is no test suite checked in yet `pytest` is wired up so contributions can land alongside fixtures.)

---

## Operational notes

- **Fernet key rotation** is not implemented. If you change `ENCRYPT_KEY`, every previously-stored connection URI / SSH password / per-DB LLM key becomes unreadable. Plan for this.
- **Background tasks** (schema scans, fine-tuning) use FastAPI `BackgroundTasks`. They run in Starlette's threadpool fine for one-off scans, but for high concurrency or jobs > a few minutes, move them to a real worker (Celery / arq / RQ).
- **Query execution** is gated by `SQL_EXECUTION_TIMEOUT` and `UPPER_LIMIT_QUERY_RETURN_ROWS`. Read those before pointing this at production data.
- **Generated SQL is executed.** The agent has a dry-run tool, but ultimately the `/execute` endpoint runs whatever the model produced. Use a least-privileged DB user.
