from sqlalchemy import create_engine, text

def hybrid_search(user_query_vector, target_user_id):
    # Connect to the pgvector container
    engine = create_engine("postgresql+psycopg2://admin:password@localhost:5432/ai_app")
    
    # Hybrid Query: Vector distance (<=>) combined with a relational WHERE clause
    sql = text("""
        SELECT document_text, embedding <=> :vector AS distance 
        FROM documents 
        WHERE user_id = :uid 
        ORDER BY distance ASC 
        LIMIT 5;
    """)
    
    with engine.connect() as conn:
        # Execute the search natively in Postgres
        results = conn.execute(sql, {
            "vector": str(user_query_vector), 
            "uid": target_user_id
        })
        return [row for row in results]
