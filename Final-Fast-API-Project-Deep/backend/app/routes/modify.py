# app/routes/modify.py

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from app.state import get_user_state  # ✅ Per-user state
from app.utils.llm_helpers import translate_natural_language_to_sql, GoogleGenerativeAI
from app.utils.sql_helpers import execute_sql_query
from app.utils.db_helpers import refresh_tables
import sqlalchemy
from sqlalchemy import create_engine
from app.config import GOOGLE_API_KEY, MYSQL_USER, MYSQL_PASSWORD, MYSQL_HOST, MYSQL_DATABASE, MODEL_NAME

router = APIRouter()

class ModificationRequest(BaseModel):
    command: str

# Initialize LLM instance
try:
    llm = GoogleGenerativeAI(model=MODEL_NAME, api_key=GOOGLE_API_KEY)
except Exception as e:
    raise RuntimeError(f"Failed to initialize LLM: {e}")

@router.post("/modify_data")
def modify_data(
    request: ModificationRequest,
    user_state = Depends(get_user_state)  # ✅ Inject user-specific state
):
    if not user_state.table_names:
        raise HTTPException(status_code=400, detail="No tables available.")

    # Fetch or create the database connection
    connection = user_state.personal_engine
    if connection is None:
        try:
            db_url = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}/{MYSQL_DATABASE}"
            connection = create_engine(db_url)
            user_state.personal_engine = connection
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to establish DB connection: {e}")

    # Construct schema info
    schema_info = "\n".join([
        f"Table: {name}, Columns: {', '.join(df.columns)}"
        for name, df in user_state.table_names
    ])

    # Translate command to SQL
    try:
        sql_query = translate_natural_language_to_sql(request.command, schema_info, llm)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM translation failed: {e}")

    # Execute the query
    try:
        if hasattr(connection, "cursor"):  # Raw DBAPI connection
            cursor = connection.cursor(buffered=True)
            try:
                cursor.execute(sql_query)
                connection.commit()
            finally:
                cursor.close()
        else:  # SQLAlchemy engine
            with connection.begin() as conn:
                conn.execute(sqlalchemy.text(sql_query))

        # Refresh tables
        refresh_tables(connection, user_state.table_names, user_state.original_table_names)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing modification: {e}")

    return {"status": "modification executed", "sql_query": sql_query}
