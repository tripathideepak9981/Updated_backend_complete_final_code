# app/utils/data_processing.py
import pandas as pd
import re
import math

def load_data(file) -> pd.DataFrame:
    try:
        if file.filename.endswith(".csv"):
            return pd.read_csv(file.file)
        elif file.filename.endswith(".xlsx"):
            return pd.read_excel(file.file)
        else:
            return pd.DataFrame()
    except Exception as e:
        print(f"Error loading file {file.filename}: {e}")
        return pd.DataFrame()

def generate_table_name(file_name: str, existing_names: set = None) -> str:
    """
    Generate a base table name from the file name.
    For example, "Sales Data.xlsx" becomes "sales_data".

    If an existing_names set is provided and the generated base name already exists,
    this function raises a ValueError indicating that a duplicate file is not allowed.
    """
    base_name = file_name.split('.')[0].replace(" ", "_").lower()
    if existing_names is not None and base_name in existing_names:
        raise ValueError(f"Duplicate file '{file_name}' not allowed.")
    return base_name


def clean_nan(obj):
    """
    Recursively replaces all float NaN values with None.
    This ensures JSON-serializable output for FastAPI responses.
    """
    import math

    if isinstance(obj, list):
        return [clean_nan(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    elif isinstance(obj, float) and math.isnan(obj):
        return None
    return obj

# app/utils/data_processing.py

import math
import numpy as np

def clean_nan_and_numpy(obj):
    """
    Recursively convert:
      - numpy data types (int64, float64, etc.) to Python int/float
      - NaN/None to None
      - lists/dicts recursively
    """
    if isinstance(obj, list):
        return [clean_nan_and_numpy(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: clean_nan_and_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, float) and math.isnan(obj):
        return None
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif obj is None:
        return None
    return obj



def get_data_preview(df: pd.DataFrame, max_rows: int = 20) -> list[dict]:
    """
    Returns top `max_rows` rows of the DataFrame with all columns,
    converting NaN values to None for safe JSON serialization.
    """
    preview_df = df.head(max_rows)
    preview = preview_df.to_dict(orient="records")

    from .data_processing import clean_nan  # Make sure this import matches your structure
    return clean_nan(preview)



def generate_detailed_overview_in_memory(table_names: list) -> str:
    overview_text_parts = []
    for tname, df in table_names:
        row_count = len(df)
        numeric_cols = df.select_dtypes(include=["number"])
        if not numeric_cols.empty:
            desc = numeric_cols.describe().T
            stats_info = []
            for col, row_data in desc.iterrows():
                stats_info.append(
                    f"- {col}: min={row_data['min']:.2f}, max={row_data['max']:.2f}, mean={row_data['mean']:.2f}, std={row_data['std']:.2f}"
                )
            numeric_stats_text = "Numeric columns summary:\n" + "\n".join(stats_info)
        else:
            numeric_stats_text = "(No numeric columns found.)"
        categorical_cols = df.select_dtypes(include=["object"])
        if not categorical_cols.empty:
            cat_info = []
            for col in categorical_cols.columns:
                value_counts = df[col].value_counts(dropna=False).head(3)
                top_vals = ", ".join([f"{idx} ({count})" for idx, count in value_counts.items()])
                cat_info.append(f"- {col}: top values → {top_vals}")
            categorical_stats_text = "Categorical columns summary:\n" + "\n".join(cat_info)
        else:
            categorical_stats_text = "(No categorical columns found.)"
        block = (
            f"Table: {tname}\n"
            f"Row Count: {row_count}\n"
            f"{numeric_stats_text}\n"
            f"{categorical_stats_text}\n"
            "----\n"
        )
        overview_text_parts.append(block)
    return "\n".join(overview_text_parts)

def summarize_schema_for_llm(table_names: list[tuple[str, pd.DataFrame]]) -> str:
    """
    Creates a concise schema summary string from all uploaded tables for LLM prompting.
    """
    parts = []
    for tname, df in table_names:
        row_count = len(df)
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

        numeric_stats = []
        for col in num_cols:
            col_data = df[col].dropna()
            if not col_data.empty:
                stats = f"{col} (min: {col_data.min():.2f}, max: {col_data.max():.2f}, mean: {col_data.mean():.2f})"
                numeric_stats.append(stats)

        cat_stats = []
        for col in cat_cols:
            top_vals = df[col].dropna().value_counts().head(3).index.tolist()
            cat_stats.append(f"{col} (e.g. {', '.join(map(str, top_vals))})")

        part = f"""Table: {tname}
- Rows: {row_count}
- Numeric: {', '.join(numeric_stats) or 'None'}
- Categorical: {', '.join(cat_stats) or 'None'}"""
        parts.append(part)
    return "\n\n".join(parts)


def build_data_stats_for_prompt(table_data) -> str:
    """
    Build human-readable statistical overview for prompt guidance.
    Input: list of (table_name, DataFrame)
    Output: string for prompt
    """
    import pandas as pd

    lines = []
    for table_name, df in table_data:
        lines.append(f"📊 Table: {table_name}")
        lines.append(f"  - Rows: {len(df)}, Columns: {len(df.columns)}")

        num_cols = df.select_dtypes(include='number').columns
        cat_cols = df.select_dtypes(include='object').columns

        # Top categorical
        for col in cat_cols[:2]:  # Limit to top 2
            top_vals = df[col].value_counts(normalize=True).head(3)
            formatted = ", ".join([f"{idx} ({val:.1%})" for idx, val in top_vals.items()])
            lines.append(f"  - Top values for '{col}': {formatted}")

        # Summary for numeric
        for col in num_cols[:2]:  # Limit to top 2
            mean = df[col].mean()
            min_val = df[col].min()
            max_val = df[col].max()
            lines.append(f"  - '{col}': mean = {mean:.2f}, min = {min_val}, max = {max_val}")

        lines.append("")  # spacing

    return "\n".join(lines)


def generate_structured_business_overview(table_names: list) -> str:
    parts = []

    for tname, df in table_names:
        row_count = len(df)
        block = [f"Table: {tname} — Total rows: {row_count}"]

        numeric_cols = df.select_dtypes(include=["number"])
        if not numeric_cols.empty:
            for col in numeric_cols.columns:
                col_data = df[col].dropna()
                if col_data.empty:
                    continue
                block.append(
                    f"- {col}: min={col_data.min():.2f}, max={col_data.max():.2f}, avg={col_data.mean():.2f}, std={col_data.std():.2f}"
                )

        cat_cols = df.select_dtypes(include=["object", "category"])
        for col in cat_cols.columns:
            top_vals = df[col].dropna().value_counts().head(3)
            if not top_vals.empty:
                top_str = ", ".join([f"{k} ({v})" for k, v in top_vals.items()])
                block.append(f"- {col}: top → {top_str}")

        parts.append("\n".join(block))

    return "\n\n".join(parts)

# app/utils/data_processing.py

def generate_structured_overview_for_df(df: pd.DataFrame, name: str) -> str:
    """
    Generate simple metrics overview for a given DataFrame
    """
    num_rows, num_cols = df.shape
    col_stats = []
    for col in df.columns:
        non_nulls = df[col].notnull().sum()
        unique_vals = df[col].nunique()
        dtype = str(df[col].dtype)
        col_stats.append(f"- {col} ({dtype}): {non_nulls} non-null, {unique_vals} unique")

    return f"""
📁 Table: {name}
✅ Rows: {num_rows}, Columns: {num_cols}

📌 Column Summary:
{chr(10).join(col_stats[:10])}
""".strip()



