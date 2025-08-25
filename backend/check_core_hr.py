# backend/check_core_hr.py
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from app.services.db_service import SQLServerDatabaseService

def check_core_hr_tables():
    print("=== Checking Core HR Tables ===\n")
    
    db = SQLServerDatabaseService()
    all_tables = db.get_schema_tables("dbo")
    
    # Key HR tables we'd expect in an HR system
    core_hr_patterns = [
        'PSN',          # Person/Personnel
        'EMP',          # Employee
        'ORG',          # Organization
        'STAFF',        # Staff
        'ACCOUNT',      # Account
        'BASEINFO',     # Base Info
        'CONTACT',      # Contact
        'EDUCATION',    # Education
        'WORK',         # Work
        'FAMILY',       # Family
        'POSITION',     # Position
        'ROLE',         # Role
        'USER'          # User
    ]
    
    print("Core HR tables found:")
    for pattern in core_hr_patterns:
        matching_tables = [t for t in all_tables if pattern in t.upper()]
        if matching_tables:
            print(f"\n{pattern} tables ({len(matching_tables)}):")
            for table in matching_tables[:10]:  # Show first 10
                print(f"  - {table}")
            if len(matching_tables) > 10:
                print(f"  ... and {len(matching_tables) - 10} more")
        else:
            print(f"\n{pattern} tables: NONE FOUND")
    
    # Check what's in current index vs what should be
    print(f"\n=== Table Priority Analysis ===")
    
    # Most important for "employees by department" queries
    critical_tables = []
    important_tables = []
    
    for table in all_tables:
        table_upper = table.upper()
        
        # Critical for employee/department queries
        if any(pattern in table_upper for pattern in ['PSNACCOUNT', 'PSNBASEINFO', 'ORGSTDSTRUCT', 'EMPLOYEE', 'DEPARTMENT']):
            critical_tables.append(table)
        # Important for HR operations
        elif any(pattern in table_upper for pattern in ['PSN', 'ORG', 'EMP', 'STAFF']) and not table_upper.startswith('ATD'):
            important_tables.append(table)
    
    print(f"\nCRITICAL tables for employee/dept queries ({len(critical_tables)}):")
    for table in critical_tables:
        print(f"  - {table}")
    
    print(f"\nIMPORTANT non-attendance HR tables ({len(important_tables)}):")
    for table in important_tables[:15]:
        print(f"  - {table}")
    if len(important_tables) > 15:
        print(f"  ... and {len(important_tables) - 15} more")
    
    # Check if these are in our current index
    print(f"\n=== Current Index Analysis ===")
    from app.services.indexer import load_index
    index = load_index()
    indexed_tables = list(index.get('tables', {}).keys())
    
    critical_in_index = [t for t in critical_tables if f"dbo.{t}" in indexed_tables]
    important_in_index = [t for t in important_tables if f"dbo.{t}" in indexed_tables]
    
    print(f"Critical tables in current index: {len(critical_in_index)}/{len(critical_tables)}")
    for table in critical_in_index:
        print(f"  ✓ {table}")
    
    missing_critical = [t for t in critical_tables if f"dbo.{t}" not in indexed_tables]
    if missing_critical:
        print(f"\nMISSING critical tables:")
        for table in missing_critical:
            print(f"  ✗ {table}")
    
    print(f"\nImportant tables in current index: {len(important_in_index)}/{len(important_tables)}")

if __name__ == "__main__":
    check_core_hr_tables()