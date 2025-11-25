#!/usr/bin/env python3
"""
Utility to scan MyScale / ClickHouse databases and create vector indexes
for every column that stores embeddings (e.g. *_embedding Array(Float32)).

Example:
    python build_myscale_vector_indexes.py \\
        --host 112.126.57.89 --port 9000 --user default --password 'xxxxx' \\
        --db-prefix arxiv --dry-run
满意后去掉 --dry-run 执行；若需要强制刷新已有索引加 --rebuild。你也可以用 --databases arxiv bird spider 精确指定库，或用 --include-all-columns 覆盖关键词限制。
"""

from __future__ import annotations

import argparse
import fnmatch
import logging
import re
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    from clickhouse_driver import Client
except ImportError:  # pragma: no cover - surfaced via CLI error message
    Client = None  # type: ignore

# Database names that should never be touched unless explicitly requested.
SYSTEM_DATABASES = {
    "system",
    "information_schema",
    "INFORMATION_SCHEMA",
}


@dataclass(frozen=True)
class VectorColumn:
    database: str
    table: str
    column: str
    column_type: str


@dataclass(frozen=True)
class ExistingIndex:
    database: str
    table: str
    column: str
    name: str
    expr: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create VECTOR INDEX for embedding columns in MyScale databases.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="112.126.57.89", help="MyScale / ClickHouse host.")
    parser.add_argument("--port", type=int, default=9000, help="TCP port.")
    parser.add_argument("--user", default="default", help="User name.")
    parser.add_argument("--password", default="xxxxx", help="User password.")

    parser.add_argument(
        "--databases",
        nargs="+",
        default=None,
        help="Process only these databases. When omitted, all non-system DBs are scanned.",
    )
    parser.add_argument(
        "--db-prefix",
        default=None,
        help="Only process databases whose name starts with this prefix.",
    )
    parser.add_argument(
        "--db-like",
        default=None,
        help="Unix-shell pattern (fnmatch) to filter database names, e.g. 'wikipedia_*'.",
    )
    parser.add_argument(
        "--skip-databases",
        nargs="*",
        default=(),
        help="Databases that must be skipped even if they pass the filters.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional maximum number of databases to process (helps with experimentation).",
    )

    parser.add_argument(
        "--column-keywords",
        nargs="+",
        default=["embedding", "vector"],
        help="Column name keywords that qualify for indexing. Ignored when --include-all-columns is set.",
    )
    parser.add_argument(
        "--include-all-columns",
        action="store_true",
        help="Index every vector-typed column even if it does not match --column-keywords.",
    )
    parser.add_argument(
        "--index-prefix",
        default="v_idx_",
        help="Prefix used for newly created index names.",
    )
    parser.add_argument(
        "--index-type",
        default="MSTG",
        help="Vector index type (e.g. MSTG, HNSW).",
    )
    parser.add_argument(
        "--metric",
        default=None,
        help="Optional metric_type setting passed to the index (e.g. cosine, l2, ip).",
    )
    parser.add_argument(
        "--extra-settings",
        default=None,
        help="Additional raw settings appended after TYPE ... clause, e.g. \"'ef_construction'=200\".",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Drop existing vector indexes for the target columns before recreating them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print ALTER statements without executing them.",
    )

    return parser.parse_args()


def quote_ident(name: str) -> str:
    """Wrap identifiers with backticks and escape existing ones."""
    return f"`{name.replace('`', '``')}`"


def get_client(args: argparse.Namespace) -> Client:
    if Client is None:
        logging.error("clickhouse-driver is not installed. Run `pip install clickhouse-driver` first.")
        sys.exit(1)
    return Client(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
    )


def discover_databases(client: Client, args: argparse.Namespace) -> List[str]:
    rows = client.execute("SELECT name FROM system.databases")
    names = [row[0] for row in rows]
    if args.databases:
        explicit = set(args.databases)
        names = [name for name in names if name in explicit]
    if args.db_prefix:
        names = [name for name in names if name.startswith(args.db_prefix)]
    if args.db_like:
        names = [name for name in names if fnmatch.fnmatch(name, args.db_like)]

    skip = set(args.skip_databases) | SYSTEM_DATABASES
    names = [name for name in names if name not in skip]

    names.sort()
    if args.limit and args.limit > 0:
        names = names[: args.limit]
    return names


def discover_vector_columns(
    client: Client,
    database: str,
    keywords: Sequence[str],
    include_all: bool,
) -> List[VectorColumn]:
    query = """
    SELECT
        table,
        name,
        type
    FROM system.columns
    WHERE database = %(database)s
      AND (
        positionCaseInsensitiveUTF8(type, 'vector(') > 0
        OR positionCaseInsensitiveUTF8(type, 'array(float32') > 0
        OR positionCaseInsensitiveUTF8(type, 'array(float64') > 0
        OR positionCaseInsensitiveUTF8(type, 'array(nullable(float32') > 0
        OR positionCaseInsensitiveUTF8(type, 'array(nullable(float64') > 0
      )
    ORDER BY table, name
    """
    rows = client.execute(query, {"database": database})
    matches: List[VectorColumn] = []
    lowered_keywords = [kw.lower() for kw in keywords]
    for table, column, col_type in rows:
        if not include_all and lowered_keywords:
            column_lower = column.lower()
            if not any(kw in column_lower for kw in lowered_keywords):
                continue
        matches.append(VectorColumn(database, table, column, col_type))
    return matches


def parse_index_expr(expr: str) -> str | None:
    """
    Extract indexed column name from `expr` inside system.vector_indices.
    Example expr: "v_idx_title_embedding title_embedding TYPE MSTG".
    """
    match = re.search(r"\s`?([A-Za-z0-9_]+)`?\s+TYPE\s", expr)
    return match.group(1) if match else None


def discover_existing_indexes(client: Client, databases: Iterable[str]) -> Dict[Tuple[str, str, str], ExistingIndex]:
    if not databases:
        return {}
    query = """
    SELECT database, table, name, expr
    FROM system.vector_indices
    WHERE database IN %(databases)s
    """
    rows = client.execute(query, {"databases": tuple(databases)})
    result: Dict[Tuple[str, str, str], ExistingIndex] = {}
    for database, table, index_name, expr in rows:
        column = parse_index_expr(expr)
        if not column:
            continue
        entry = ExistingIndex(database, table, column, index_name, expr)
        result[(database, table, column)] = entry
    return result


def build_add_sql(
    column: VectorColumn,
    index_name: str,
    args: argparse.Namespace,
) -> str:
    db_ident = f"{quote_ident(column.database)}.{quote_ident(column.table)}"
    idx_ident = quote_ident(index_name)
    col_ident = quote_ident(column.column)

    settings_parts: List[str] = []
    if args.metric:
        settings_parts.append(f"'metric_type' = '{args.metric}'")
    if args.extra_settings:
        settings_parts.append(args.extra_settings)
    settings_clause = ""
    if settings_parts:
        joined = ", ".join(settings_parts)
        settings_clause = f"({joined})"

    return (
        f"ALTER TABLE {db_ident} "
        f"ADD VECTOR INDEX {idx_ident} {col_ident} "
        f"TYPE {args.index_type}{settings_clause}"
    )


def build_drop_sql(column: VectorColumn, index_name: str) -> str:
    db_ident = f"{quote_ident(column.database)}.{quote_ident(column.table)}"
    idx_ident = quote_ident(index_name)
    return f"ALTER TABLE {db_ident} DROP VECTOR INDEX {idx_ident}"


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    client = get_client(args)
    # Required for ALTER TABLE ... VECTOR INDEX
    client.execute("SET allow_experimental_vector_similarity_index = 1")

    databases = discover_databases(client, args)
    if not databases:
        logging.warning("No databases satisfy the filters. Nothing to do.")
        return
    logging.info("Target databases: %s", ", ".join(databases))

    existing = discover_existing_indexes(client, databases)
    logging.info("Found %d existing vector indexes.", len(existing))

    planned_columns: List[VectorColumn] = []
    for db in databases:
        cols = discover_vector_columns(client, db, args.column_keywords, args.include_all_columns)
        planned_columns.extend(cols)
    if not planned_columns:
        logging.info("No vector-compatible columns were discovered. Exiting.")
        return
    logging.info(
        "Discovered %d vector columns (after keyword filtering).",
        len(planned_columns),
    )

    executed = 0
    skipped = 0
    failed = 0
    for column in planned_columns:
        existing_entry = existing.get((column.database, column.table, column.column))
        index_name = existing_entry.name if existing_entry else f"{args.index_prefix}{column.column}"

        statements: List[str] = []
        if existing_entry:
            if args.rebuild:
                statements.append(build_drop_sql(column, existing_entry.name))
                statements.append(build_add_sql(column, index_name, args))
            else:
                skipped += 1
                continue
        else:
            statements.append(build_add_sql(column, index_name, args))

        for statement in statements:
            logging.info("%s", statement)
            if args.dry_run:
                continue
            try:
                client.execute(statement)
                executed += 1
            except Exception as exc:  # pragma: no cover - requires live DB
                failed += 1
                logging.error("Failed to run statement on %s.%s.%s: %s", column.database, column.table, column.column, exc)
                break

    logging.info(
        "Vector index creation finished. executed=%d, skipped=%d (already indexed), failed=%d%s",
        executed,
        skipped,
        failed,
        " [dry-run]" if args.dry_run else "",
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:  # pragma: no cover
        logging.warning("Interrupted by user.")
