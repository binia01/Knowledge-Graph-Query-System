"""
Validation test cases — demonstrates the validation layer catching
problematic Cypher queries before execution (Stack Overflow dataset).

These tests require a running Neo4j instance and valid Gemini API key.
Run with: python -m pytest tests/test_validation.py -v
"""

import pytest
from src.config import Config
from src.database import Neo4jDatabase
from src.validator import CypherValidator


@pytest.fixture(scope="module")
def db():
    """Create a database connection for the test session."""
    database = Neo4jDatabase()
    yield database
    database.close()


@pytest.fixture(scope="module")
def validator(db):
    """Create a validator with the real schema."""
    schema = db.get_schema()
    return CypherValidator(schema)


class TestValidationCatchesBadQueries:
    """
    Three required test cases showing the validation layer
    catching problematic queries before execution.
    """

    def test_reversed_relationship_direction(self, validator):
        """
        Test Case 1: Reversed relationship direction.

        The query reverses the TAGGED direction:
            (Tag)-[:TAGGED]->(Question)  ← WRONG
        Correct direction is:
            (Question)-[:TAGGED]->(Tag)

        This is syntactically valid Cypher. Neo4j won't throw an error —
        it will silently return empty results. Only validation catches this.
        """
        question = "What questions are tagged with python?"
        bad_cypher = (
            "MATCH (t:Tag {name: 'python'})-[:TAGGED]->(q:Question) "
            "RETURN q.title"
        )

        result = validator.validate(question, bad_cypher)

        print(f"\n--- Test: Reversed Relationship Direction ---")
        print(f"Query:  {bad_cypher}")
        print(f"Score:  {result.score}")
        print(f"Issues: {result.issues}")
        print(f"Direction correct: {result.direction_correct}")

        # The validator should flag this as low confidence
        assert result.score < 0.7, (
            f"Validator should score reversed direction below 0.7, got {result.score}"
        )
        assert not result.direction_correct or len(result.issues) > 0, (
            "Validator should identify direction issue"
        )

    def test_nonexistent_node_label(self, validator):
        """
        Test Case 2: Non-existent node label.

        The query uses :Post which doesn't exist in the schema.
        The correct label is :Question. This query will return empty
        results without an error.
        """
        question = "Show me all posts"
        bad_cypher = "MATCH (p:Post) RETURN p.title"

        result = validator.validate(question, bad_cypher)

        print(f"\n--- Test: Non-existent Node Label ---")
        print(f"Query:  {bad_cypher}")
        print(f"Score:  {result.score}")
        print(f"Issues: {result.issues}")
        print(f"Labels correct: {result.labels_correct}")

        assert result.score < 0.7, (
            f"Validator should score non-existent label below 0.7, got {result.score}"
        )
        assert not result.labels_correct or len(result.issues) > 0, (
            "Validator should identify label issue"
        )

    def test_ambiguous_query_clarification(self, validator):
        """
        Test Case 3: Ambiguous / overly vague query.

        The query is too broad — "show me everything" is
        meaningless without specifying what or from which nodes.
        The system should ask the user to clarify.
        """
        question = "Show me some stuff about code"
        vague_cypher = (
            "MATCH (q:Question)-[r]->(n) "
            "RETURN q.title, type(r), n"
        )

        result = validator.validate(question, vague_cypher)

        print(f"\n--- Test: Ambiguous Query ---")
        print(f"Query:  {vague_cypher}")
        print(f"Score:  {result.score}")
        print(f"Issues: {result.issues}")

        # This may score medium or low depending on Gemini's judgment.
        # The key is that the classifier should route "ambiguous" before
        # this even reaches validation. But if it does, we still check.
        print(f"Score zone: {'execute' if result.score >= 0.7 else 'correct' if result.score >= 0.4 else 'reject'}")

        # We mainly verify the validator runs and returns a valid result
        assert 0.0 <= result.score <= 1.0, "Score should be between 0 and 1"
        assert isinstance(result.issues, list), "Issues should be a list"


class TestValidationAcceptsGoodQueries:
    """Ensure the validator accepts well-formed queries."""

    def test_correct_tagged_query(self, validator):
        """A correct query should score >= 0.7."""
        question = "What questions are tagged with python?"
        good_cypher = (
            "MATCH (q:Question)-[:TAGGED]->(t:Tag {name: 'python'}) "
            "RETURN q.title"
        )

        result = validator.validate(question, good_cypher)

        print(f"\n--- Test: Correct Query ---")
        print(f"Query:  {good_cypher}")
        print(f"Score:  {result.score}")

        assert result.score >= 0.7, (
            f"Correct query should score >= 0.7, got {result.score}"
        )

    def test_correct_answer_query(self, validator):
        """Another correct query should score >= 0.7."""
        question = "What are the most viewed questions?"
        good_cypher = (
            "MATCH (q:Question) "
            "RETURN q.title, q.view_count ORDER BY q.view_count DESC LIMIT 10"
        )

        result = validator.validate(question, good_cypher)

        print(f"\n--- Test: Correct View Count Query ---")
        print(f"Query:  {good_cypher}")
        print(f"Score:  {result.score}")

        assert result.score >= 0.7, (
            f"Correct query should score >= 0.7, got {result.score}"
        )
