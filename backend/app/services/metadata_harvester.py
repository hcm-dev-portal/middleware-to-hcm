# backend/app/services/metadata_harvester.py
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from app.services.db_service import SQLServerDatabaseService
from app.core.paths import METADATA_PATH

logger = logging.getLogger(__name__)

def harvest_metadata(
    db_service: SQLServerDatabaseService,
    schemas: List[str] = None,
    max_columns_per_table: int = 200
) -> Dict[str, Any]:
    """
    Harvest metadata from the database for specified schemas.
    
    Returns a dictionary with database schema information including
    tables and their columns for each schema.
    """
    if schemas is None:
        schemas = ["dbo"]
    
    logger.info(f"Harvesting metadata for schemas: {schemas}")
    
    metadata = {
        "database": db_service.config.get("database", "Unknown"),
        "server": db_service.config.get("server", "Unknown"),
        "schemas": {}
    }
    
    total_tables = 0
    
    for schema_name in schemas:
        logger.info(f"Processing schema: {schema_name}")
        schema_metadata = {}
        
        try:
            # Get all tables in this schema
            tables = db_service.get_schema_tables(schema_name)
            logger.info(f"Found {len(tables)} tables in schema {schema_name}")
            
            for table_name in tables:
                try:
                    # Get column information for this table
                    columns = db_service.get_table_columns(schema_name, table_name)
                    
                    # Limit columns if needed
                    if len(columns) > max_columns_per_table:
                        logger.warning(f"Table {schema_name}.{table_name} has {len(columns)} columns, limiting to {max_columns_per_table}")
                        columns = columns[:max_columns_per_table]
                    
                    if columns:  # Only include tables with columns
                        schema_metadata[table_name] = {
                            "columns": columns,
                            "column_count": len(columns)
                        }
                        total_tables += 1
                        
                        if total_tables % 50 == 0:  # Progress logging
                            logger.info(f"Processed {total_tables} tables so far...")
                    
                except Exception as e:
                    logger.warning(f"Failed to get columns for {schema_name}.{table_name}: {e}")
                    continue
            
            metadata["schemas"][schema_name] = schema_metadata
            logger.info(f"Successfully processed {len(schema_metadata)} tables in schema {schema_name}")
            
        except Exception as e:
            logger.error(f"Failed to process schema {schema_name}: {e}")
            metadata["schemas"][schema_name] = {}
    
    logger.info(f"Metadata harvesting complete. Total tables processed: {total_tables}")
    return metadata

def save_metadata(metadata: Dict[str, Any], path: Optional[Path] = None) -> Path:
    """Save metadata to JSON file"""
    file_path = path or METADATA_PATH
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving metadata to: {file_path}")
    
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Metadata saved successfully to: {file_path}")
    return file_path

def load_metadata(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Load metadata from JSON file"""
    file_path = path or METADATA_PATH
    
    if not file_path.exists():
        logger.warning(f"Metadata file not found: {file_path}")
        return None
    
    try:
        with file_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        logger.info(f"Metadata loaded successfully from: {file_path}")
        return metadata
        
    except Exception as e:
        logger.error(f"Failed to load metadata from {file_path}: {e}")
        return None

def get_metadata_summary(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Get a summary of the metadata"""
    if not metadata:
        return {"error": "No metadata available"}
    
    summary = {
        "database": metadata.get("database"),
        "server": metadata.get("server"),
        "schemas": {}
    }
    
    total_tables = 0
    total_columns = 0
    
    for schema_name, schema_data in metadata.get("schemas", {}).items():
        table_count = len(schema_data)
        column_count = sum(table_info.get("column_count", 0) for table_info in schema_data.values())
        
        summary["schemas"][schema_name] = {
            "table_count": table_count,
            "column_count": column_count
        }
        
        total_tables += table_count
        total_columns += column_count
    
    summary["totals"] = {
        "tables": total_tables,
        "columns": total_columns
    }
    
    return summary

if __name__ == "__main__":
    """Test metadata harvesting"""
    import sys
    
    print("Testing metadata harvesting...")
    
    try:
        from app.services.db_service import SQLServerDatabaseService
        
        db = SQLServerDatabaseService()
        if not db.test_connection():
            print("Database connection failed!")
            sys.exit(1)
        
        print("Harvesting metadata for dbo schema...")
        metadata = harvest_metadata(db, schemas=["dbo"], max_columns_per_table=20)
        
        summary = get_metadata_summary(metadata)
        print(f"Summary: {summary}")
        
        # Save to file
        save_path = save_metadata(metadata)
        print(f"Metadata saved to: {save_path}")
        
        # Test loading
        loaded = load_metadata()
        if loaded:
            print("Metadata loaded successfully!")
            print(f"Database: {loaded.get('database')}")
            print(f"Schemas: {list(loaded.get('schemas', {}).keys())}")
        
    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)