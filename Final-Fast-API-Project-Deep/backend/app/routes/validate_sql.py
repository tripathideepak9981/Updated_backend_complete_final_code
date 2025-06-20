# app/routes/validate_sql.py

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from app.utils.auth_helpers import get_current_user
from app.state import get_duckdb_connection
from app.state import get_user_state  # ✅ updated import
from app.models import User


from app.utils.sql_helpers import execute_sql_query
from app.utils.llm_helpers import generate_dynamic_response
from sqlalchemy.orm import Session
from app.database import get_db
import re

import logging
logger = logging.getLogger("validate_sql")
logger.setLevel(logging.INFO)

router = APIRouter()

class SQLValidationRequest(BaseModel):
    sql_query: str

@router.post("/validate_sql")
def validate_sql_query(
    request: SQLValidationRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    user_state = get_user_state(current_user.id)
    sql_query = request.sql_query.strip()
    logger.info(f"User {current_user.username} executed SQL validation: {sql_query}")


    # ✅ Enforce SELECT-only
    if not sql_query.lower().startswith("select"):
        return {
            "status": "error",
            "error_type": "InvalidQuery",
            "message": "Only SELECT queries are allowed. Please remove any INSERT, UPDATE, or DELETE statements."
        }

    # ✅ Ensure data is loaded
    if not user_state.table_names:
        return {
            "status": "error",
            "error_type": "NoDataLoaded",
            "message": "No tables are currently loaded. Please upload or connect your data first."
        }

    try:
        # ✅ Execute the query
        if user_state.personal_engine:

            result_df = execute_sql_query(sql_query, None, user_state.personal_engine)

        else:
            con = get_duckdb_connection(user_state)
            for table_name, df in user_state.table_names:

                con.register(table_name, df)
            result_df = con.execute(sql_query).df()

    except Exception as e:
        raw_message = str(e)
        friendly_msg = "Something went wrong while running your SQL query."

        # 👇 Common error patterns
        if "Unknown column" in raw_message:
            match = re.search(r"Unknown column '(.+?)'", raw_message)
            if match:
                col = match.group(1)
                friendly_msg = f"The column `{col}` does not exist. Please check your column name."

        elif "Table" in raw_message and "does not exist" in raw_message:
            match = re.search(r"Table '(.+?)' doesn't exist", raw_message)
            if match:
                tbl = match.group(1)
                friendly_msg = f"The table `{tbl}` does not exist. Please verify the table name."

        elif "syntax" in raw_message.lower():
            friendly_msg = "There is a syntax error in your SQL query. Please check for commas, quotes, or missing keywords."

        elif "no such table" in raw_message.lower():
            match = re.search(r'no such table: (\w+)', raw_message)
            if match:
                tbl = match.group(1)
                friendly_msg = f"The table `{tbl}` was not found in the current session."

        return {
            "status": "error",
            "error_type": "ExecutionError",
            "message": friendly_msg,
            "raw_error": raw_message  # Optional for debugging/logs
        }

    # ✅ Handle empty results
    if result_df.empty:
        return {
            "status": "success",
            "result": [],
            "message": "Query executed but returned no rows."
        }

    # ✅ 1 row × 1 column → LLM response
    if result_df.shape == (1, 1):
        column_name = result_df.columns[0]
        value = result_df.iloc[0, 0]

        try:
            result_text = generate_dynamic_response(
                sql_query,
                column_name,
                value
            )
        except Exception:
            # Fallback manual label if LLM fails
            label = column_name.replace("_", " ").replace("(", "").replace(")", "").title()
            result_text = f"{label}: {value}"

        return {
            "status": "success",
            "result": result_text
        }

    # ✅ Normal result (table with multiple rows or columns)
    return {
        "status": "success",
        "result": result_df.to_dict(orient="records")
    }
