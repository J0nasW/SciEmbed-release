"""DuckDB connection wrapper for the science datalake.

Supports two modes:
  1. Local: connects to existing DuckDB file with pre-defined views.
  2. HPC/parquet: creates in-memory DuckDB, registers parquet directories
     as views matching the same schema. Set DATALAKE_PARQUET_ROOT env var
     to the directory containing {fulltext/, s2ag/, openalex/} parquet dirs.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

import duckdb
import pyarrow as pa

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "/path/to/data/science_datalake/datalake.duckdb"

# View definitions for parquet-based mode (HPC)
_PARQUET_VIEWS = {
    "fulltext.papers": "SELECT * FROM read_parquet('{root}/fulltext/unified/*.parquet', hive_partitioning=false)",
    "s2ag.papers": "SELECT * FROM read_parquet('{root}/s2ag/papers/*.parquet', hive_partitioning=false)",
    "s2ag.abstracts": "SELECT * FROM read_parquet('{root}/s2ag/abstracts/*.parquet', hive_partitioning=false)",
    "s2ag.citations": "SELECT * FROM read_parquet('{root}/s2ag/citations/*.parquet', hive_partitioning=false)",
    "s2ag.tldrs": "SELECT * FROM read_parquet('{root}/s2ag/tldrs/*.parquet', hive_partitioning=false)",
    "openalex.works_topics": "SELECT * FROM read_parquet('{root}/openalex/works_topics/*.parquet', hive_partitioning=false)",
    "openalex.works": "SELECT * FROM read_parquet('{root}/openalex/works/*.parquet', hive_partitioning=false)",
    "openalex.topics": "SELECT * FROM read_parquet('{root}/openalex/topics/*.parquet', hive_partitioning=false)",
    "openalex.works_related_works": "SELECT * FROM read_parquet('{root}/openalex/works_related_works/*.parquet', hive_partitioning=false)",
}

# Computed views that depend on base parquet views (registered after base views)
_COMPUTED_VIEWS = {
}


class DatalakeConnection:
    """Managed DuckDB connection to the science datalake.

    Path resolution for db_path:
      - If DATALAKE_PARQUET_ROOT is set: uses in-memory DuckDB with parquet views.
      - Else: explicit db_path arg > DATALAKE_DB env var > default path.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        read_only: bool = True,
        threads: int = 8,
        memory_limit: str = "32GB",
    ) -> None:
        self.parquet_root = os.getenv("DATALAKE_PARQUET_ROOT")
        self.db_path = self._resolve_path(db_path) if not self.parquet_root else None
        self.read_only = read_only
        self.threads = threads
        self.memory_limit = memory_limit
        self._conn: duckdb.DuckDBPyConnection | None = None

    @staticmethod
    def _resolve_path(db_path: str | Path | None) -> Path:
        if db_path is not None:
            return Path(db_path)
        env_path = os.getenv("DATALAKE_DB")
        if env_path:
            return Path(env_path)
        return Path(DEFAULT_DB_PATH)

    def _register_parquet_views(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Register parquet directories as DuckDB views (HPC mode).

        Skips views whose parquet directories are empty or missing, so
        pipelines that only need a subset of tables can run before all
        data has been synced.
        """
        root = self.parquet_root
        for schema in ("fulltext", "s2ag", "openalex"):
            conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        for view_name, query_template in _PARQUET_VIEWS.items():
            query = query_template.format(root=root)
            try:
                conn.execute(f"CREATE OR REPLACE VIEW {view_name} AS {query}")
            except duckdb.IOException:
                logger.warning("Skipping view %s (parquet files not found)", view_name)
        # Register computed views that depend on base views
        for view_name, query in _COMPUTED_VIEWS.items():
            try:
                conn.execute(f"CREATE OR REPLACE VIEW {view_name} AS {query}")
            except Exception:
                logger.warning("Skipping computed view %s (base view not available)", view_name)

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            if self.parquet_root:
                # HPC mode: in-memory DuckDB with parquet views
                self._conn = duckdb.connect(":memory:")
            else:
                # Local mode: connect to existing DuckDB file
                self._conn = duckdb.connect(
                    str(self.db_path),
                    read_only=self.read_only,
                )
            self._conn.execute(f"SET threads = {self.threads}")
            self._conn.execute(f"SET memory_limit = '{self.memory_limit}'")
            self._conn.execute("SET preserve_insertion_order = false")
            self._conn.execute("SET arrow_large_buffer_size = true")
            if self.parquet_root:
                self._register_parquet_views(self._conn)
        return self._conn

    def query(self, sql: str, params: list[Any] | None = None) -> list[tuple[Any, ...]]:
        """Execute a query and return all rows as a list of tuples."""
        if params:
            result = self.conn.execute(sql, params)
        else:
            result = self.conn.execute(sql)
        return result.fetchall()

    def query_df(self, sql: str, params: list[Any] | None = None) -> pa.Table:
        """Execute a query and return results as a PyArrow Table."""
        if params:
            result = self.conn.execute(sql, params)
        else:
            result = self.conn.execute(sql)
        return result.fetch_arrow_table()

    def stream_query(
        self,
        sql: str,
        batch_size: int = 100_000,
        params: list[Any] | None = None,
    ) -> Generator[pa.RecordBatch, None, None]:
        """Execute a query and yield results as PyArrow RecordBatches."""
        if params:
            result = self.conn.execute(sql, params)
        else:
            result = self.conn.execute(sql)
        while True:
            batch = result.fetch_arrow_table(rows_per_batch=batch_size)
            if batch is None or len(batch) == 0:
                break
            for record_batch in batch.to_batches():
                yield record_batch

    def execute(self, sql: str, params: list[Any] | None = None) -> None:
        """Execute a statement without returning results."""
        if params:
            self.conn.execute(sql, params)
        else:
            self.conn.execute(sql)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> DatalakeConnection:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


@contextmanager
def get_datalake(
    db_path: str | Path | None = None,
    read_only: bool = True,
    threads: int = 8,
    memory_limit: str = "32GB",
) -> Generator[DatalakeConnection, None, None]:
    """Context manager for a DatalakeConnection."""
    conn = DatalakeConnection(
        db_path=db_path,
        read_only=read_only,
        threads=threads,
        memory_limit=memory_limit,
    )
    try:
        yield conn
    finally:
        conn.close()
