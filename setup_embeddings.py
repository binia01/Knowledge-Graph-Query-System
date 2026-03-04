"""
Setup script — generates vector embeddings for Question nodes
and creates the vector index in Neo4j.

Run this ONCE before using vector/hybrid search features:
    python setup_embeddings.py
"""

import sys
import time
from src.config import Config
from src.database import Neo4jDatabase
from src.vector_search import VectorSearch


def main():
    errors = Config.validate()
    if errors:
        print("Configuration errors:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)

    print("Connecting to Neo4j...")
    db = Neo4jDatabase()

    if not db.verify_connectivity():
        print("ERROR: Cannot connect to Neo4j. Check your connection settings.")
        db.close()
        sys.exit(1)

    print("Connected successfully!")

    # Check if Stack Overflow data exists
    questions = db.run_query("MATCH (q:Question) RETURN count(q) AS count")
    question_count = questions[0]["count"] if questions else 0
    print(f"Found {question_count} questions in the database.")

    if question_count == 0:
        print("\nNo questions found! Please load the Stack Overflow dataset first.")
        print("In Neo4j Browser, run:  :play stackoverflow")
        print("Or import your own Stack Overflow dump.")
        db.close()
        sys.exit(1)

    vs = VectorSearch(db)

    # Create vector index
    print("\nCreating vector index...")
    try:
        vs.create_vector_index()
        print("Vector index created (or already exists).")
    except Exception as e:
        print(f"Warning: Could not create vector index: {e}")

    # Generate embeddings
    print("\nGenerating embeddings for questions... (using title + body)")
    print("This calls Google Gemini's embedding API for each question.")
    print("It may take a minute...\n")

    try:
        start = time.time()
        count = vs.generate_question_embeddings()
        elapsed = time.time() - start
        print(f"Generated {count} embeddings in {elapsed:.1f} seconds.")
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        db.close()
        sys.exit(1)

    # Verify
    with_embeddings = db.run_query(
        "MATCH (q:Question) WHERE q.embedding IS NOT NULL RETURN count(q) AS count"
    )
    emb_count = with_embeddings[0]["count"] if with_embeddings else 0
    print(f"\nQuestions with embeddings: {emb_count}/{question_count}")

    if vs.db.check_vector_index_exists("stackoverflow_embeddings"):
        print("Vector index: READY")
    else:
        print("Vector index: NOT FOUND (vector search may not work)")

    print("\nSetup complete! You can now use vector and hybrid search.")
    db.close()


if __name__ == "__main__":
    main()
