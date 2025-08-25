# backend/test_rebuild.py
import sys
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Add the backend directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

try:
    from app.core.paths import DATA_DIR, METADATA_PATH, INDEX_PATH
    from app.services.db_service import SQLServerDatabaseService
    
    print("=== Rebuild Test ===")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"DATA_DIR exists: {DATA_DIR.exists()}")
    print(f"METADATA_PATH: {METADATA_PATH}")
    print(f"INDEX_PATH: {INDEX_PATH}")
    print(f"Metadata exists: {METADATA_PATH.exists()}")
    print(f"Index exists: {INDEX_PATH.exists()}")
    
    # Test database connection
    print("\nTesting database connection...")
    db = SQLServerDatabaseService()
    connected = db.test_connection()
    print(f"Database connected: {connected}")
    
    if connected:
        # Try to get some tables
        tables = db.get_schema_tables("dbo")
        print(f"Found {len(tables)} tables")
        if tables:
            print(f"First 5 tables: {tables[:5]}")
            
            # Try to get columns for first table
            first_table = tables[0]
            columns = db.get_table_columns("dbo", first_table)
            print(f"Table {first_table} has {len(columns)} columns")
            
    # Try importing the other services
    print("\nTesting imports...")
    try:
        from app.services.metadata_harvester import harvest_metadata, save_metadata
        print("✓ metadata_harvester imported")
    except Exception as e:
        print(f"✗ metadata_harvester failed: {e}")
        
    try:
        from app.services.indexer import build_index, save_index, load_index
        print("✓ indexer imported")
    except Exception as e:
        print(f"✗ indexer failed: {e}")
        
    # Try a small rebuild
    if connected:
        print("\nTrying small rebuild...")
        try:
            # Get just 5 tables for testing
            small_metadata = {
                "database": db.config.get("database"),
                "schemas": {
                    "dbo": {}
                }
            }
            
            test_tables = tables[:5]  # Just first 5 tables
            for table in test_tables:
                try:
                    columns = db.get_table_columns("dbo", table)
                    if columns:
                        small_metadata["schemas"]["dbo"][table] = {
                            "columns": columns,
                            "column_count": len(columns)
                        }
                except Exception as e:
                    print(f"Failed to get columns for {table}: {e}")
            
            print(f"Created metadata for {len(small_metadata['schemas']['dbo'])} tables")
            
            # Save metadata
            save_metadata(small_metadata)
            print(f"Metadata saved to: {METADATA_PATH}")
            print(f"Metadata file exists now: {METADATA_PATH.exists()}")
            
            # Build and save index
            index = build_index(small_metadata)
            save_index(index)
            print(f"Index saved to: {INDEX_PATH}")
            print(f"Index file exists now: {INDEX_PATH.exists()}")
            
            # Test loading
            loaded_index = load_index()
            if loaded_index:
                print(f"✓ Index loaded successfully with {len(loaded_index.get('tables', {}))} tables")
            else:
                print("✗ Failed to load index")
                
            print("\n=== Test completed successfully! ===")
            
        except Exception as e:
            print(f"Rebuild test failed: {e}")
            import traceback
            traceback.print_exc()
    
except Exception as e:
    print(f"Failed to import modules: {e}")
    import traceback
    traceback.print_exc()