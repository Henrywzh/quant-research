| Component                 | Recommended                                  |
| ------------------------- | -------------------------------------------- |
| **Database**              | PostgreSQL (for structured, relational data) |
| **Cache / fast I/O**      | Redis or DuckDB (optional)                   |
| **Storage for raw files** | S3 / local `/data/raw/`                      |
| **ORM / Interface**       | SQLAlchemy (Python), or pandas + psycopg2    |
| **ETL / Pipelines**       | Prefect / Airflow / custom cron jobs         |


