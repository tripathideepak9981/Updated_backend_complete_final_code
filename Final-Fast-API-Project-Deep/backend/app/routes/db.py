# app/routes/db.py
 
from fastapi import APIRouter, HTTPException, Body, Depends
from pydantic import BaseModel
from typing import List
from app.utils.db_helpers import connect_personal_db, list_tables, disconnect_database
from app.state import get_user_state
from fastapi.encoders import jsonable_encoder
import logging
import pandas as pd
import math
from app.models import User
from app.utils.auth_helpers import get_current_user
from fastapi import HTTPException, APIRouter, Depends
from sqlalchemy import text
from app.utils.db_helpers import connect_personal_db
from app.models import User
from app.state import get_user_state
from app.config import MYSQL_USER, MYSQL_PASSWORD, MYSQL_HOST
import logging
 
router = APIRouter()
logger = logging.getLogger("db")
logger.setLevel(logging.DEBUG)
 
class DBConnectionParams(BaseModel):
    db_type: str
    host: str
    port: int
    user: str
    password: str
    database: str
 
def clean_nan(obj):
    """
    Recursively traverse lists and dictionaries, replacing any float('nan') with None.
    """
    if isinstance(obj, list):
        return [clean_nan(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: clean_nan(value) for key, value in obj.items()}
    elif isinstance(obj, float):
        if math.isnan(obj):
            return None
        else:
            return obj
    else:
        return obj
 
@router.post("/connect_db")
def connect_db(
    params: DBConnectionParams,
    current_user: User = Depends(get_current_user)
):
    logger.info(f"Attempting DB connection for user: {current_user.username}")
    try:
        engine = connect_personal_db(
            params.db_type,
            params.host,
            params.user,
            params.password,
            params.database,
            params.port
        )
        if engine is None:
            logger.error("Engine is None after connect_personal_db()")
            raise HTTPException(status_code=500, detail="Database connection failed.")

        user_state = get_user_state(current_user.id)
        user_state.personal_engine = engine  # ✅ CORRECT

        tables = list_tables(engine)
        logger.info(f"Connected successfully. Tables: {tables}")
        return jsonable_encoder({"status": "connected", "tables": tables})
    except Exception as e:
        logger.exception("Error during connect_db")
        raise HTTPException(status_code=500, detail=f"Error connecting to DB: {e}")
 
@router.post("/load_tables")
def load_tables(
    table_names: List[str] = Body(...),
    current_user: User = Depends(get_current_user)
):
    user_state = get_user_state(current_user.id)

    if not getattr(user_state, "personal_engine", None):
        raise HTTPException(status_code=400, detail="No personal database connected.")
   
    engine = user_state.personal_engine
    previews = {}
    loaded_tables = []

    for table in table_names:
        try:
            query = f"SELECT * FROM `{table}`;"
            df = pd.read_sql_query(query, engine)
            loaded_tables.append((table, df))
            logger.info(f"Fetched table '{table}' with shape: {df.shape}")
           
            if df.empty:
                logger.warning(f"Table {table} is empty.")
                previews[table] = "No data available (table is empty)."
            else:
                preview_data = df.head(10).to_dict(orient="records")
                preview_data = clean_nan(preview_data)
                previews[table] = preview_data if preview_data else "No preview data available."
        except Exception as e:
            logger.error(f"Error fetching data for table '{table}': {e}")
            previews[table] = f"Error fetching data: {e}"
   
    user_state.table_names = loaded_tables

    response = {
        "status": "tables loaded",
        "tables": table_names,
        "previews": previews,
        "debug": "direct fetch preview"
    }
    logger.info(f"Final Response: {response}")
    return jsonable_encoder(response)

 
@router.post("/disconnect")
def disconnect(
    current_user: User = Depends(get_current_user)
):
    user_state = get_user_state(current_user.id)  # ✅ Fix here
    disconnect_database(user_state)
    return jsonable_encoder({"status": "disconnected"})
 
 

logger = logging.getLogger(__name__)
 
@router.delete("/delete_table/{table_name}")
def delete_table(table_name: str, current_user: User = Depends(get_current_user)):
    """
    Deletes a specific table from the user's personal database.
    """
    # Use the custom connect_personal_db method to get the engine connection
    engine = connect_personal_db(
        db_type="mysql",  # Using the same parameters as in other parts of the app
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=current_user.dynamic_db
    )
 
    if not engine:
        raise HTTPException(status_code=500, detail="Failed to connect to database.")
 
    # Check if the table exists in the database
    try:
        with engine.connect() as conn:
            result = conn.execute(text(f"SHOW TABLES LIKE '{table_name}'"))
            if result.fetchone() is None:
                raise HTTPException(status_code=404, detail="Table not found in database.")
    except Exception as e:
        logger.error(f"Error checking table existence: {e}")
        raise HTTPException(status_code=500, detail="Error checking table existence.")
 
    # Attempt to delete the table
    try:
        with engine.connect() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS `{table_name}`"))
            logger.info(f"Table '{table_name}' deleted successfully for user {current_user.username}.")
       
        # Remove the table from user's state (in-memory)
        user_state = get_user_state(current_user.id)
        user_state.table_names = [t for t in user_state.table_names if t[0] != table_name]  # Remove table from state
       
        return {"status": "success", "message": f"Table '{table_name}' deleted from database."}
   
    except Exception as e:
        logger.error(f"Failed to delete table '{table_name}': {e}")
        raise HTTPException(status_code=500, detail="Failed to delete table.")
 
 
@router.get("/load_user_tables_with_preview")
def load_user_tables_with_preview(
    current_user: User = Depends(get_current_user)
):
    from app.utils.db_helpers import connect_personal_db, list_tables
    import pandas as pd
    from app.state import get_user_state
    from fastapi.encoders import jsonable_encoder
    import math
    import time
    from app.utils.data_processing import get_data_preview  # Use your existing function

    def clean_nan_fast(df: pd.DataFrame) -> list:
        """
        Fast conversion of NaNs to None for JSON serializability.
        """
        return df.replace({math.nan: None}).to_dict(orient="records")

    start_time = time.time()

    user_state = get_user_state(current_user.id)
    engine = connect_personal_db(
        db_type="mysql",
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=current_user.dynamic_db
    )

    if not engine:
        raise HTTPException(status_code=500, detail="Database connection failed.")

    table_names = list_tables(engine)
    loaded_tables = []
    original_tables = []
    previews = []

    for table_name in table_names:
        try:
            # ✅ LIMIT query to fetch only 20 rows — fast and efficient
            df = pd.read_sql_query(f"SELECT * FROM `{table_name}` LIMIT 20", con=engine)

            loaded_tables.append((table_name, df))
            original_tables.append((table_name, df.copy()))

            preview = clean_nan_fast(df)
            previews.append({
                "table_name": table_name,
                "preview": preview
            })
        except Exception as e:
            logger.warning(f"Failed to load table '{table_name}': {e}")

    user_state.table_names = loaded_tables
    user_state.original_table_names = original_tables
    user_state.personal_engine = engine

    duration = round(time.time() - start_time, 2)
    logger.info(f"[Preview Load] Completed in {duration}s for user {current_user.username}")

    return {
        "status": "success",
        "tables": previews,
        "load_time_sec": duration
    }
