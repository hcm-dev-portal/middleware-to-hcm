# backend/debug_search.py
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from app.services.indexer import load_index
from app.services.retriever import search_tables, _tok

def debug_search():
    print("=== Debugging Search Function ===\n")
    
    # Load the current index
    index = load_index()
    if not index:
        print("ERROR: No index found!")
        return
    
    print(f"Index database: {index.get('database')}")
    print(f"Index has {len(index.get('tables', {}))} tables")
    print(f"Index has {len(index.get('by_token', {}))} tokens")
    
    # Show some sample tables
    tables = list(index.get('tables', {}).keys())
    print(f"\nSample tables in index: {tables[:10]}")
    
    # Show some sample tokens
    by_token = index.get('by_token', {})
    tokens = list(by_token.keys())
    print(f"\nSample tokens in index: {tokens[:20]}")
    
    # Test the specific query that's failing
    query = "Show me active employees by department"
    print(f"\nTesting query: '{query}'")
    
    # Break down the tokenization
    query_tokens = _tok(query)
    print(f"Query tokens: {query_tokens}")
    
    # Check which tokens exist in index
    print("\nToken matches:")
    for token in query_tokens:
        if token in by_token:
            tables_for_token = by_token[token]['tables']
            print(f"  '{token}' -> {len(tables_for_token)} tables: {list(tables_for_token.keys())[:5]}")
        else:
            print(f"  '{token}' -> NOT FOUND")
    
    # Run the actual search with fixed function
    from app.services.retriever import search_tables
    results = search_tables(index, query, schema_filter="dbo", top_k=10)
    print(f"\nSearch results: {results}")
    
    # Test with simpler queries
    simple_queries = [
        "action",
        "process", 
        "api",
        "employee",
        "department",
        "person",
        "org"
    ]
    
    print("\nTesting simple token queries:")
    for sq in simple_queries:
        results = search_tables(index, sq, schema_filter="dbo", top_k=3)
        print(f"  '{sq}' -> {len(results)} results: {[r[0] for r in results]}")

if __name__ == "__main__":
    debug_search()