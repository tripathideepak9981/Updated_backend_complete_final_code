# app/routes/query.py

import re
import sqlalchemy
import pandas as pd
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from app.utils.auth_helpers import get_current_user
from app.models import User
from app.state import get_user_state, get_duckdb_connection  # ✅ Updated import
from app.database import get_db
from app.config import MODEL_NAME, GOOGLE_API_KEY

from app.utils.llm_helpers import (
    build_analysis_prompt,
    build_followup_prompt,
    parse_analysis_response,
    extract_sql_from_llm_response,
    GoogleGenerativeAI,
    generate_statistical_response,
    generate_non_sql_response
)

from app.utils.sql_helpers import (
    clean_sql_query,
    execute_sql_query
)

from app.utils.data_processing import (
    summarize_schema_for_llm,
    generate_structured_business_overview
)

router = APIRouter()
llm = GoogleGenerativeAI(model=MODEL_NAME, api_key=GOOGLE_API_KEY)

class UserQuery(BaseModel):
    query: str

@router.post("/execute_query")
def execute_user_query(
    user_query: UserQuery,
    current_user: User = Depends(get_current_user),
    db: sqlalchemy.orm.Session = Depends(get_db),
):
    user_state = get_user_state(current_user.id)

    query_text = user_query.query.strip()

    if not user_state.table_names:
        raise HTTPException(status_code=400, detail="No tables loaded.")

    schema_info = summarize_schema_for_llm(user_state.table_names)
    overview_stats = generate_structured_business_overview(user_state.table_names)
    preview_df = user_state.table_names[0][1] if user_state.table_names else pd.DataFrame()

    # ✅ 1. Check if statistical query
    if any(word in query_text.lower() for word in ["mean", "median", "mode", "standard deviation", "std", "min", "max"]):
        result = generate_statistical_response(query_text, preview_df, llm)
        if result and result.get("status") == "success":
            return result

    last_entry = user_state.get_last_chat_entry()
    is_followup = last_entry and len(query_text.split()) < 6

    sql_query = ""
    explanation = ""

    try:
        if is_followup:
            followup_prompt = build_followup_prompt(
                query_text,
                last_entry["sql_query"],
                pd.DataFrame(last_entry["result_preview"])
            )
            llm_response = llm(followup_prompt)
            sql_query = extract_sql_from_llm_response(llm_response)
            explanation = "Follow-up query executed."
        else:
            full_prompt = build_analysis_prompt(query_text, schema_info, overview_stats)
            llm_response = llm(full_prompt)
            parsed = parse_analysis_response(llm_response)
            sql_query = extract_sql_from_llm_response(parsed.get("sql", ""))
            explanation = parsed.get("explanation", "").strip()

        if not sql_query:
            # No SQL returned → fallback (NOT execution failure)
            return generate_non_sql_response(query_text, overview_stats, preview_df, llm)

        sql_query = clean_sql_query(sql_query)

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": "Failed to generate SQL from your query."
        }

    # ✅ 2. Try to execute SQL (but DO NOT fallback if it fails)
    try:
        if user_state.personal_engine:
            result_df = execute_sql_query(sql_query, query_text, user_state.personal_engine)
        else:
            con = user_state.duckdb_conn  # ✅ Per-user DuckDB
            for tname, df in user_state.table_names:
                con.register(tname, df)
            result_df = con.execute(sql_query).df()

        # ✅ 3. Save interaction
        user_state.add_chat_entry(query_text, sql_query, result_df)

        return {
            "status": "success",
            "query": query_text,
            "sql_query": sql_query,
            "response": explanation or "Query executed successfully.",
            "result": result_df.to_dict(orient="records") if not result_df.empty else []
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": "SQL was generated but failed to execute. Check the query or your data."
        }

# ✅ Fix reset_context
@router.post("/reset_context")
def reset_chat_context(current_user: User = Depends(get_current_user)):
    user_state = get_user_state(current_user.id)
    user_state.reset()
    return {"status": "chat context reset"}

@router.get("/initial_suggestions")
def get_initial_suggestions(current_user: User = Depends(get_current_user)):
    user_state = get_user_state(current_user.id)
    return {"suggested_questions": user_state.initial_suggestions}
