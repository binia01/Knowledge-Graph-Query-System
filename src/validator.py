"""
Cypher query validator — uses Gemini as a judge to score generated
Cypher queries before execution.

Score zones:
  >= 0.7  → execute
  0.4–0.69 → auto-correct and retry
  < 0.4  → reject, ask user to clarify
"""

import json
import re
from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate
from src.llm import get_llm


VALIDATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a Neo4j Cypher query validator. Evaluate the following Cypher
query against the provided schema and original question.

SCHEMA:
{schema}

ORIGINAL QUESTION:
{question}

CYPHER QUERY:
{cypher}

Evaluate the query and respond in STRICT JSON format with these keys:
- "score": a float from 0.0 to 1.0 indicating confidence in correctness
- "issues": a list of strings describing any problems found
- "is_valid_syntax": boolean indicating if the Cypher syntax is valid
- "direction_correct": boolean indicating if relationship directions match the schema
- "labels_correct": boolean indicating if node labels match the schema exactly
- "has_return": boolean indicating if the query has a RETURN clause

Score guidelines:
- 1.0: Perfect query, correct syntax, correct direction, correct labels, answers the question
- 0.7-0.9: Minor issues but will produce correct results
- 0.4-0.69: Has issues that may cause wrong results (wrong direction, missing filter, etc.)
- 0.0-0.39: Fundamentally wrong (wrong labels, no RETURN, answers different question)

Return ONLY the JSON object, no other text.""",
    ),
    ("human", "Validate this query."),
])


@dataclass
class ValidationResult:
    """Result of validating a Cypher query."""
    score: float
    issues: list[str]
    is_valid_syntax: bool
    direction_correct: bool
    labels_correct: bool
    has_return: bool


class CypherValidator:
    """Validates Cypher queries using Gemini as a judge."""

    def __init__(self, schema: str):
        self.schema = schema
        self.llm = get_llm()
        # Pre-parse schema labels for cheap heuristic checks
        self._schema_labels = set(
            re.findall(r':(\w+)', schema)
        )

    # ------------------------------------------------------------------
    # Fast heuristic checks (no LLM call needed)
    # ------------------------------------------------------------------

    def _heuristic_check(self, cypher: str) -> ValidationResult | None:
        """
        Run cheap, local validations and return early when the query
        is clearly broken.  Returns None when the heuristics are
        inconclusive and the LLM judge should be consulted.
        """
        issues: list[str] = []

        has_return = bool(re.search(r'\bRETURN\b', cypher, re.IGNORECASE))
        if not has_return:
            issues.append("Query has no RETURN clause")

        # Minimal syntax sanity
        is_valid_syntax = has_return and any(
            kw in cypher.upper()
            for kw in ("MATCH", "CALL", "UNWIND", "CREATE", "MERGE", "WITH")
        )
        if not is_valid_syntax:
            issues.append("Query has no recognized read clause (MATCH, CALL, etc.)")

        # Check that labels used in the query appear in the schema
        used_labels = set(re.findall(r':(\w+)', cypher))
        unknown = used_labels - self._schema_labels
        labels_correct = len(unknown) == 0
        if not labels_correct:
            issues.append(f"Unknown labels not in schema: {unknown}")

        # If obvious failures found, return immediately — no LLM needed
        if issues:
            score = 0.25 if has_return else 0.1
            return ValidationResult(
                score=score,
                issues=issues,
                is_valid_syntax=is_valid_syntax,
                direction_correct=True,   # can't check without LLM
                labels_correct=labels_correct,
                has_return=has_return,
            )

        # Heuristics passed — defer to LLM for deeper validation
        return None

    # ------------------------------------------------------------------

    def validate(self, question: str, cypher: str) -> ValidationResult:
        """
        Validate a Cypher query.  Runs fast heuristic checks first;
        only calls the LLM judge when the heuristics are inconclusive.
        """
        quick = self._heuristic_check(cypher)
        if quick is not None:
            return quick

        chain = VALIDATION_PROMPT | self.llm
        response = chain.invoke({
            "schema": self.schema,
            "question": question,
            "cypher": cypher,
        })

        raw = response.content.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = re.sub(r"```(?:json)?\s*", "", raw)
            raw = raw.replace("```", "").strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # If Gemini returned non-JSON, treat as low confidence
            return ValidationResult(
                score=0.3,
                issues=["Validator returned non-JSON response: " + raw[:200]],
                is_valid_syntax=False,
                direction_correct=False,
                labels_correct=False,
                has_return=False,
            )

        return ValidationResult(
            score=float(data.get("score", 0.0)),
            issues=data.get("issues", []),
            is_valid_syntax=data.get("is_valid_syntax", False),
            direction_correct=data.get("direction_correct", False),
            labels_correct=data.get("labels_correct", False),
            has_return=data.get("has_return", False),
        )
