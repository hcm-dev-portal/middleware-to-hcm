# app/services/factory.py
from .db_service import SQLServerDatabaseService
from .nlp_service_2 import LanguageNativeNLPService

def create_enhanced_nlp_service(db_service: SQLServerDatabaseService) -> LanguageNativeNLPService:
    # Centralized place to tweak defaults
    return LanguageNativeNLPService(db_service=db_service, model_name="gpt-4.1", temperature=0.1)
