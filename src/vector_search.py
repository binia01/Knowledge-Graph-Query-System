"""
Vector search module — handles embedding generation, storage,
and similarity-based retrieval from Neo4j vector indexes.
"""

from src.config import Config
from src.database import Neo4jDatabase
from src.llm import get_embeddings


class VectorSearch:
    """Manages vector embeddings and similarity search in Neo4j."""

    def __init__(self, db: Neo4jDatabase):
        self.db = db
        self.embeddings = get_embeddings()
        self.index_name = "stackoverflow_embeddings"

    def create_vector_index(self):
        """Create the vector index on Question.embedding if it doesn't exist."""
        if self.db.check_vector_index_exists(self.index_name):
            return

        self.db.run_query(f"""
            CREATE VECTOR INDEX {self.index_name} IF NOT EXISTS
            FOR (q:Question) ON q.embedding
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {Config.VECTOR_DIMENSIONS},
                    `vector.similarity_function`: 'cosine'
                }}
            }}
        """)

    def generate_question_embeddings(self):
        """
        Generate and store embeddings for all Question nodes that
        don't already have one. Uses title + body as the text.
        """
        questions = self.db.run_query("""
            MATCH (q:Question)
            WHERE q.embedding IS NULL
            RETURN q.title AS title, q.body_markdown AS body,
                   q.link AS link, elementId(q) AS node_id
        """)

        if not questions:
            return 0

        count = 0
        for question in questions:
            text = self._question_to_text(question)
            embedding = self.embeddings.embed_query(text)
            self.db.run_query(
                """
                MATCH (q:Question) WHERE elementId(q) = $node_id
                SET q.embedding = $embedding
                """,
                {"node_id": question["node_id"], "embedding": embedding},
            )
            count += 1

        return count

    def similarity_search(self, query: str, top_k: int | None = None) -> list[dict]:
        """
        Perform a vector similarity search against the stackoverflow_embeddings index.
        Returns questions ranked by cosine similarity.
        """
        top_k = top_k or Config.VECTOR_TOP_K
        query_embedding = self.embeddings.embed_query(query)

        results = self.db.run_query(
            """
            CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector)
            YIELD node AS question, score
            WHERE score > $threshold
            RETURN question.title AS title, question.body_markdown AS body,
                   question.link AS link, score
            ORDER BY score DESC
            """,
            {
                "index_name": self.index_name,
                "top_k": top_k,
                "query_vector": query_embedding,
                "threshold": Config.SIMILARITY_THRESHOLD,
            },
        )
        return results

    def _question_to_text(self, question: dict) -> str:
        """Convert a question node to a text string for embedding."""
        title = question.get('title', '')
        body = question.get('body', '') or question.get('body_markdown', '')
        return f"{title}\n{body}" if body else title

    def hybrid_search(
        self, query: str, cypher_filter: str, top_k: int | None = None
    ) -> list[dict]:
        """
        Perform a hybrid search: vector similarity + graph relationship filter.

        Args:
            query: The semantic query text for vector similarity.
            cypher_filter: Additional Cypher WHERE/MATCH clause to filter results.
            top_k: Number of vector candidates to retrieve before filtering.
        """
        top_k = top_k or Config.VECTOR_TOP_K * 2  # Get more candidates for filtering
        query_embedding = self.embeddings.embed_query(query)

        # Build hybrid query: vector search first, then graph filter
        hybrid_cypher = f"""
            CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector)
            YIELD node AS question, score
            WHERE score > $threshold
            {cypher_filter}
            RETURN question.title AS title, question.body_markdown AS body,
                   question.link AS link, score
            ORDER BY score DESC
        """

        results = self.db.run_query(
            hybrid_cypher,
            {
                "index_name": self.index_name,
                "top_k": top_k,
                "query_vector": query_embedding,
                "threshold": Config.SIMILARITY_THRESHOLD,
            },
        )
        return results
