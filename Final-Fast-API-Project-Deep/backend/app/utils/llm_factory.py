# app/utils/llm_facotry.py


from app.utils.llm_helpers import GoogleGenerativeAI
from app.config import GOOGLE_API_KEY, MODEL_NAME

_llm_instance = None  # ✅ Private global singleton

def get_llm():
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = GoogleGenerativeAI(model=MODEL_NAME, api_key=GOOGLE_API_KEY)
    return _llm_instance
