# backend/rebuild_hr_index.py
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from app.services.db_service import SQLServerDatabaseService
from app.services.metadata_harvester import save_metadata
from app.services.indexer import build_index, save_index

def rebuild_hr_index():
    print("=== Rebuilding Index with HR Tables ===\n")
    
    db = SQLServerDatabaseService()
    if not db.test_connection():
        print("ERROR: Database connection failed!")
        return False
    
    # Get all tables
    all_tables = db.get_schema_tables("dbo")
    print(f"Found {len(all_tables)} total tables in dbo schema")
    
    # Filter for HR-relevant tables using keywords
    hr_keywords = [
        'PSN',        # Person/Personnel
        'EMP',        # Employee  
        'ORG',        # Organization
        'STAFF',      # Staff
        'DEPT',       # Department
        'ATD',        # Attendance
        'PAY',        # Payroll
        'LEAVE',      # Leave management
        'OVERTIME',   # Overtime
        'ACCOUNT',    # Account info
        'BASEINFO',   # Base information
        'CONTACT',    # Contact info
        'EDUCATION',  # Education
        'WORK',       # Work experience
        'FAMILY',     # Family info
        'FLOWER',     # Workflow forms
        'LMS',        # Learning management
        'CPC',        # Competency
        'BONUS',      # Compensation
        'EXAM'        # Examinations
    ]
    
    hr_tables = []
    for table in all_tables:
        table_upper = table.upper()
        if any(keyword in table_upper for keyword in hr_keywords):
            hr_tables.append(table)
    
    print(f"Found {len(hr_tables)} HR-related tables")
    print(f"Sample HR tables: {hr_tables[:10]}")
    
    # Limit to reasonable number for testing
    selected_tables = hr_tables[:100]  # First 100 HR tables
    print(f"Selected {len(selected_tables)} tables for indexing")
    
    # Build metadata
    metadata = {
        "database": db.config.get("database"),
        "server": db.config.get("server"),
        "schemas": {
            "dbo": {}
        }
    }
    
    processed_count = 0
    for table_name in selected_tables:
        try:
            columns = db.get_table_columns("dbo", table_name)
            if columns:
                metadata["schemas"]["dbo"][table_name] = {
                    "columns": columns,
                    "column_count": len(columns)
                }
                processed_count += 1
                
                if processed_count % 20 == 0:
                    print(f"Processed {processed_count} tables...")
                    
        except Exception as e:
            print(f"Failed to process {table_name}: {e}")
            continue
    
    print(f"\nSuccessfully processed {processed_count} tables")
    
    # Save metadata
    print("Saving metadata...")
    save_metadata(metadata)
    
    # Build and save index
    print("Building search index...")
    index = build_index(metadata)
    save_index(index)
    
    # Verify the new index
    print(f"\nNew index contains:")
    print(f"- {len(index.get('tables', {}))} tables")
    print(f"- {len(index.get('by_token', {}))} searchable tokens")
    
    # Show some sample HR tables that were indexed
    hr_table_names = [name for name in index.get('tables', {}).keys() if any(kw in name.upper() for kw in ['PSN', 'EMP', 'ORG', 'ATD', 'PAY'])]
    print(f"Sample HR tables in index: {hr_table_names[:10]}")
    
    print("\n=== HR Index Rebuild Complete! ===")
    return True

if __name__ == "__main__":
    success = rebuild_hr_index()
    if success:
        print("\nYou can now test HR queries like:")
        print("- 'Show me employees by department'")
        print("- 'Find employee information'") 
        print("- 'Show attendance records'")
        print("- 'List payroll data'")
    else:
        print("\nRebuild failed!")
        sys.exit(1)