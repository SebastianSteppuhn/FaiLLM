# FaiLLM — Setup & Postgres Guide

A Framework for edge-case testing LLMs. This guide shows how to install locally and connect PostgreSQL using a `.env` file.

Demo Video: https://sebastian.microflux.de/demo.mp4
Tech Video: https://sebastian.microflux.de/tech.mp4

## 1) Prerequisites
- Python 3.10+ (3.11/3.12 OK)
- pip + venv
- PostgreSQL 13+ (local or remote)
- Optional: Docker (for local Postgres)

## 2) Create venv & install
```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
# If you have requirements.txt:
pip install -r requirements.txt || true
# Minimal set if you don't:
pip install gradio python-dotenv requests
# Choose ONE LLM client:
pip install open          # preferred
# or:
pip install openai
# Postgres client:
pip install psycopg2-binary
```

## 3) Configure `.env`
Create a `.env` in the project root.

### LLM
```
# one of these
OPEN_API_KEY=sk-...            # used by the `open` package
OPENAI_API_KEY=sk-...          # used by the `openai` package

LLM_MODEL=gpt-4o-mini
# Optional: point to an OpenAI-compatible server (vLLM/litellm/LM Studio)
LLM_BASE_URL=
```

### PostgreSQL
Use a single URL (recommended):
```
DATABASE_URL=postgresql+psycopg2://postgres:postgres@localhost:5432/faillm
```
Or discrete vars:
```
PGHOST=localhost
PGPORT=5432
PGUSER=postgres
PGPASSWORD=postgres
PGDATABASE=faillm
```

## 4) Load env & read settings (in `main.py`)
```python
from dotenv import load_dotenv
load_dotenv()

import os
MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
BASE_URL = os.getenv("LLM_BASE_URL") or None
API_KEY = os.getenv("OPEN_API_KEY") or os.getenv("OPENAI_API_KEY")

DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    DB_URL = (
        f"postgresql+psycopg2://{os.getenv('PGUSER','postgres')}:"
        f"{os.getenv('PGPASSWORD','postgres')}@{os.getenv('PGHOST','localhost')}:"
        f"{os.getenv('PGPORT','5432')}/{os.getenv('PGDATABASE','faillm')}"
    )
```

## 5) Run
```bash
python main.py
```
Open the printed local URL.

## 6) Quick Postgres check (optional)
Create `scripts/db_check.py`:
```python
import os
from dotenv import load_dotenv
load_dotenv()

url = os.getenv("DATABASE_URL")
if not url:
    url = (
        f"postgresql://{os.getenv('PGUSER','postgres')}:"
        f"{os.getenv('PGPASSWORD','postgres')}@{os.getenv('PGHOST','localhost')}:"
        f"{os.getenv('PGPORT','5432')}/{os.getenv('PGDATABASE','faillm')}"
    )
print("Using:", url)

import psycopg2
conn = psycopg2.connect(url)
with conn.cursor() as cur:
    cur.execute("SELECT version();")
    print("Postgres:", cur.fetchone()[0])
conn.close()
print("OK")
```

Run:
```bash
python scripts/db_check.py
```

## 7) Docker Postgres (optional)
`docker-compose.yml`:
```yaml
version: "3.8"
services:
  db:
    image: postgres:15
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: faillm
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
volumes:
  pgdata:
```
Then:
```bash
docker compose up -d
```
Use this in `.env`:
```
DATABASE_URL=postgresql+psycopg2://postgres:postgres@localhost:5432/faillm
```

