# app/utils/db_helpers.py
import os
import pandas as pd
import sqlalchemy
from sqlalchemy import text
from app.config import MYSQL_USER, MYSQL_PASSWORD, MYSQL_HOST, MYSQL_DATABASE
from app.state import get_user_state
from fastapi import Depends
from fastapi import HTTPException


def refresh_tables(connection, table_names, original_table_names) -> None:
    if connection is None:
        print("Cannot refresh tables: connection is None.")
        return
    if hasattr(connection, "cursor"):
        engine = sqlalchemy.create_engine(
            f"mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}/{MYSQL_DATABASE}"
        )
        cursor = connection.cursor(buffered=True)
        try:
            cursor.execute("SHOW TABLES;")
            db_tables = cursor.fetchall()
        finally:
            cursor.close()
        with engine.connect() as conn:
            for tbl in db_tables:
                tbl_name = tbl[0]
                try:
                    df = pd.read_sql_query(f"SELECT * FROM `{tbl_name}`", conn)
                except Exception as e:
                    print(f"Error loading table '{tbl_name}': {e}")
                    continue
                if tbl_name not in [tn for tn, _ in table_names]:
                    table_names.append((tbl_name, df))
                else:
                    idx = next(i for i, (name, _) in enumerate(table_names) if name == tbl_name)
                    table_names[idx] = (tbl_name, df)
    else:
        dialect_name = ""
        if hasattr(connection, "engine"):
            dialect_name = connection.engine.dialect.name
        if dialect_name == "mysql":
            query = text("SHOW TABLES;")
            result = connection.execute(query)
            db_tables = result.fetchall()
            for tbl in db_tables:
                tbl_name = tbl[0]
                try:
                    df = pd.read_sql_query(f"SELECT * FROM `{tbl_name}`", connection)
                except Exception as e:
                    print(f"Error loading table '{tbl_name}': {e}")
                    continue
                if tbl_name not in [tn for tn, _ in table_names]:
                    table_names.append((tbl_name, df))
                else:
                    idx = next(i for i, (name, _) in enumerate(table_names) if name == tbl_name)
                    table_names[idx] = (tbl_name, df)
        else:
            query = text(
                "SELECT table_name FROM v_catalog.tables "
                "WHERE is_system_table = false "
                "AND table_schema NOT IN ('v_catalog','v_monitor','v_internal')"
            )
            result = connection.execute(query)
            db_tables = [(row[0],) for row in result.fetchall()]
            for tbl in db_tables:
                tbl_name = tbl[0]
                try:
                    df = pd.read_sql_query(f"SELECT * FROM `{tbl_name}`", connection)
                except Exception as e:
                    print(f"Error loading table '{tbl_name}': {e}")
                    continue
                if tbl_name not in [tn for tn, _ in table_names]:
                    table_names.append((tbl_name, df))
                else:
                    idx = next(i for i, (name, _) in enumerate(table_names) if name == tbl_name)
                    table_names[idx] = (tbl_name, df)


from typing import List, Union
from sqlalchemy.engine import Engine
from sqlalchemy import text
from app.config import MYSQL_DATABASE
import logging

logger = logging.getLogger("list_tables")
logger.setLevel(logging.INFO)

def list_tables(connection: Union[Engine, any]) -> List[str]:
    """
    Returns a list of user tables from a database connection.
    Supports MySQL (via SQLAlchemy or raw) and Vertica (SQLAlchemy).
    Automatically handles schema detection for Vertica.

    Args:
        connection: SQLAlchemy engine or raw DBAPI connection

    Returns:
        List[str]: Fully qualified table names (schema.table) for Vertica,
                   and simple table names for MySQL
    """
    try:
        # ✅ Case 1: MySQL native (raw DBAPI, like pymysql or mysql.connector)
        if hasattr(connection, "cursor") and "mysql" in str(type(connection)).lower():
            try:
                cursor = connection.cursor()
                db_name = getattr(connection, "database", MYSQL_DATABASE)
                logger.info(f"[MySQL - raw] Listing tables from DB: {db_name}")
                cursor.execute(
                    """
                    SELECT TABLE_NAME 
                    FROM INFORMATION_SCHEMA.TABLES 
                    WHERE TABLE_SCHEMA = %s;
                    """,
                    (db_name,)
                )
                tables = cursor.fetchall()
                return [table[0] for table in tables]
            finally:
                cursor.close()

        # ✅ Case 2: SQLAlchemy engine (MySQL / Vertica)
        elif hasattr(connection, "connect"):
            with connection.connect() as conn:
                # Detect SQL dialect
                if hasattr(conn, "engine"):
                    dialect = conn.engine.dialect.name.lower()
                elif hasattr(connection, "dialect"):
                    dialect = connection.dialect.name.lower()
                else:
                    dialect = "unknown"

                logger.info(f"[SQLAlchemy] Detected dialect: {dialect}")
                db_name = connection.url.database if hasattr(connection, "url") else MYSQL_DATABASE

                if dialect == "mysql":
                    logger.info(f"[MySQL - ORM] Using schema: {db_name}")
                    query = text(
                        "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = :schema"
                    )
                    result = conn.execute(query, {"schema": db_name})
                    tables = [row[0] for row in result.fetchall()]
                    logger.info(f"[MySQL] Tables found: {tables}")
                    return tables

                elif dialect == "vertica":
                    logger.info("[Vertica] Fetching non-system schemas...")
                    try:
                        schema_query = text("""
                            SELECT schema_name 
                            FROM v_catalog.schemata 
                            WHERE is_system_schema = false
                        """)
                        schemas = [row[0] for row in conn.execute(schema_query).fetchall()]
                        logger.info(f"[Vertica] Available schemas: {schemas}")
                    except Exception as e:
                        logger.error(f"[Vertica] Failed to fetch schemas: {e}")
                        return []

                    tables = []
                    for schema in schemas:
                        try:
                            result = conn.execute(text("""
                                SELECT table_name 
                                FROM v_catalog.tables 
                                WHERE table_schema = :schema AND is_system_table = false
                            """), {"schema": schema})
                            rows = result.fetchall()
                            tables.extend([f"{schema}.{row[0]}" for row in rows])
                        except Exception as e:
                            logger.warning(f"[Vertica] Failed to query schema '{schema}': {e}")
                    logger.info(f"[Vertica] Tables found: {tables}")
                    return tables

                else:
                    logger.warning(f"Unsupported DB dialect: {dialect}")
                    return []

        else:
            logger.warning("Unsupported connection type for listing tables.")
            return []

    except Exception as e:
        logger.error(f"[list_tables] Error listing tables: {e}", exc_info=True)
        return []



def get_personal_engine(db_type: str, host: str, user: str, password: str, database: str, port: int):
    try:
        if db_type.lower().startswith("vert"):
            port = port or 5433
            sqlalchemy.dialects.registry.register("vertica.vertica_python", "vertica_sqlalchemy.dialect", "VerticaDialect")
            engine_url = f"vertica+vertica_python://{user}:{password}@{host}:{port}/{database}"
        elif db_type.lower() == "mysql":
            port = port or 3306
            engine_url = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}?buffered=true"
        else:
            port = port or 5432
            engine_url = f"{db_type}://{user}:{password}@{host}:{port}/{database}"
        engine = sqlalchemy.create_engine(engine_url)
        with engine.connect() as connection:
            print(f"Connected to {db_type.upper()} database successfully!")
        return engine
    except Exception as e:
        print(f"Error connecting to {db_type} DB: {e}")
        return None

def connect_personal_db(db_type, host, user, password, database, port=3306):
    try:
        if db_type.lower() == "mysql":
            import pymysql
            from sqlalchemy import create_engine
            url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
            return create_engine(url)

        elif db_type.lower() == "vertica":
            import vertica_python
            conn_info = {
                "host": host,
                "port": port or 5433,
                "user": user,
                "password": password,
                "database": database,
                "connection_timeout": 10,
                "read_timeout": 10,
                "unicode_error": 'strict',
                "ssl": False
            }
            return vertica_python.connect(**conn_info)

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported DB type: {db_type}")

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"❌ Error connecting to user DB: {e}\n{tb}")
        raise HTTPException(
            status_code=500,
            detail=f"❌ Database connection failed: {str(e)}"
        )


def load_tables_from_personal_db(engine, table_list: list) -> tuple:
    loaded_tables = []
    original_tables = []
    dialect = engine.url.get_dialect().name.lower() if engine.url.get_dialect() else ""
    for tbl in table_list:
        try:
            if dialect == "vertica":
                query = f"SELECT * FROM {tbl}"
            else:
                query = f"SELECT * FROM `{tbl}`"
            df = pd.read_sql_query(query, con=engine)
            df.columns = [col.strip().replace(" ", "_").lower() for col in df.columns]
            original_df = df.copy()
            from app.utils.cleaning import clean_data
            df = clean_data(df)
            loaded_tables.append((tbl, df))
            original_tables.append((tbl, original_df))
        except Exception as e:
            print(f"Error loading table '{tbl}': {e}")
    return loaded_tables, original_tables

def disconnect_database(user_state):
    if getattr(user_state, "personal_engine", None):
        try:
            user_state.personal_engine.dispose()
            print("Personal database disconnected.")
        except Exception as e:
            print(f"Error disconnecting personal database: {e}")
        user_state.personal_engine = None

    if getattr(user_state, "mysql_connection", None):
        try:
            user_state.mysql_connection.close()
            print("MySQL connection disconnected.")
        except Exception as e:
            print(f"Error disconnecting MySQL connection: {e}")
        user_state.mysql_connection = None

    user_state.table_names = []
    user_state.original_table_names = []
