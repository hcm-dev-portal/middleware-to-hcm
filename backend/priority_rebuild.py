# backend/priority_rebuild.py
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from app.services.db_service import SQLServerDatabaseService
from app.services.metadata_harvester import save_metadata
from app.services.indexer import build_index, save_index

def priority_rebuild():
    print("=== Priority-Based HR Index Rebuild ===\n")
    
    db = SQLServerDatabaseService()
    if not db.test_connection():
        print("ERROR: Database connection failed!")
        return False
    
    all_tables = db.get_schema_tables("dbo")
    print(f"Found {len(all_tables)} total tables")
    
    # Priority-based table selection
    priority_groups = {
        "CRITICAL": {
            "patterns": ["ORGSTDSTRUCT", "PSNACCOUNT", "PSNBASEINFO", "EMPLOYEE", "DEPARTMENT"],
            "tables": []
        },
        "CORE_HR": {
            "patterns": ["PSN", "ORG", "EMP", "STAFF"],
            "tables": []
        },
        "PAYROLL": {
            "patterns": ["PAY_TW_", "PAY"],
            "tables": []
        },
        "WORKFLOW": {
            "patterns": ["FLOWER", "FLOW"],
            "tables": []
        },
        "LEARNING": {
            "patterns": ["LMS_", "EXAM"],
            "tables": []
        },
        "ATTENDANCE": {
            "patterns": ["ATD"],
            "tables": []
        }
    }
    
    # Categorize tables by priority
    for table in all_tables:
        table_upper = table.upper()
        categorized = False
        
        for priority, group in priority_groups.items():
            if not categorized:
                for pattern in group["patterns"]:
                    if pattern in table_upper:
                        group["tables"].append(table)
                        categorized = True
                        break
    
    # Show categorization
    for priority, group in priority_groups.items():
        print(f"{priority}: {len(group['tables'])} tables")
        if group["tables"]:
            print(f"  Sample: {group['tables'][:5]}")
    
    # Select tables with priority order
    selected_tables = []
    target_per_category = {
        "CRITICAL": 50,   # All critical tables
        "CORE_HR": 30,    # Most important HR tables
        "PAYROLL": 15,    # Key payroll tables
        "WORKFLOW": 15,   # Workflow forms
        "LEARNING": 10,   # Learning management
        "ATTENDANCE": 20  # Some attendance tables
    }
    
    for priority, group in priority_groups.items():
        limit = target_per_category.get(priority, 10)
        selected_from_group = group["tables"][:limit]
        selected_tables.extend(selected_from_group)
        print(f"Selected {len(selected_from_group)} {priority} tables")
    
    print(f"\nTotal selected: {len(selected_tables)} tables")
    print(f"Key tables included:")
    
    # Check for specific critical tables
    critical_checks = ["ORGSTDSTRUCT", "PSNACCOUNT", "PSNBASEINFO", "EMPLOYEE"]
    for check in critical_checks:
        found = [t for t in selected_tables if check in t.upper()]
        if found:
            print(f"  ✓ {check}: {found[:3]}")
        else:
            print(f"  ✗ {check}: NOT FOUND")
    
    # Build metadata with selected tables
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
                
                if processed_count % 25 == 0:
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
    
    # Verify critical tables are now indexed
    indexed_tables = list(index.get('tables', {}).keys())
    print(f"\nVerification - Critical tables now in index:")
    for check in critical_checks:
        found = [t for t in indexed_tables if check in t.upper()]
        if found:
            print(f"  ✓ {check}: {len(found)} tables")
        else:
            print(f"  ✗ {check}: STILL MISSING")
    
    print(f"\nNew index stats:")
    print(f"- {len(index.get('tables', {}))} tables indexed")
    print(f"- {len(index.get('by_token', {}))} searchable tokens")
    
    print("\n=== Priority Rebuild Complete! ===")
    return True

if __name__ == "__main__":
    success = priority_rebuild()
    if success:
        print("\nNow test queries like:")
        print("- 'Show me employees by department'")
        print("- 'Find person account information'")
        print("- 'Show organization structure'")
    else:
        print("\nRebuild failed!")
        sys.exit(1)