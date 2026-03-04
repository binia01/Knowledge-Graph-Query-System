"""
Cypher query generator — uses Gemini + schema to convert
natural language questions into Cypher queries.
"""

import re
from langchain_core.prompts import ChatPromptTemplate
from src.llm import get_llm

# Cypher keywords that mark the start of a valid query
_CYPHER_START = re.compile(
    r'^\s*(MATCH|CALL|WITH|UNWIND|CREATE|MERGE|OPTIONAL\s+MATCH|RETURN)',
    re.IGNORECASE | re.MULTILINE,
)


CYPHER_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a Neo4j Cypher expert. Given the user's question and the
database schema below, generate a syntactically correct Cypher query.

SCHEMA:
{schema}

RULES:
1. Use ONLY the node labels, relationship types, and properties shown in the schema.
2. Pay careful attention to relationship DIRECTIONS shown in the schema.
3. Always include a RETURN clause.
4. Use case-sensitive labels exactly as shown in the schema.
5. For string matching, prefer case-insensitive: toLower(n.prop) CONTAINS toLower('value').
6. Do NOT use any labels, types, or properties that are not in the schema.
7. Return ONLY the Cypher query, nothing else. No explanation, no markdown fences.""",
    ),
    ("human", "{question}"),
])


CYPHER_CORRECTION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a Neo4j Cypher expert. The following Cypher query was generated
but received a low confidence score. Fix the identified issues and return
ONLY the corrected Cypher query. No explanation, no markdown.

SCHEMA:
{schema}

ORIGINAL QUERY:
{original_query}

ISSUES FOUND:
{issues}

ORIGINAL QUESTION:
{question}""",
    ),
    ("human", "Please fix the query."),
])


def _extract_cypher(raw: str) -> str:
    """
    Robustly extract a Cypher query from LLM output that may contain
    markdown fences, explanatory text before/after the query, or
    other artifacts.
    """
    text = raw.strip()

    # 1. Try to pull content from markdown code fences first
    match = re.search(r"```(?:\w+)?\n?(.*?)```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    else:
        # Remove stray triple-backtick artifacts
        text = text.replace("```", "").strip()

    # 2. If there's preamble text before the first Cypher keyword,
    #    strip it (e.g. "Here is the query:\nMATCH ...")
    start = _CYPHER_START.search(text)
    if start and start.start() > 0:
        text = text[start.start():]

    # 3. Remove any trailing explanation after the query.
    #    After collecting Cypher lines, stop when we hit obvious prose.
    lines = text.split('\n')
    cypher_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        # After we've collected some Cypher, if a line starts with a
        # lowercase letter and has no Cypher keywords, it's likely prose.
        if stripped and not stripped.startswith('//') and cypher_lines:
            if (
                stripped[0].islower()
                and not re.search(
                    r'\b(MATCH|WHERE|RETURN|ORDER|LIMIT|WITH|UNWIND|AND|OR|SET|'
                    r'DELETE|REMOVE|CALL|YIELD|AS|DESC|ASC|CASE|WHEN|THEN|ELSE|'
                    r'END|NOT|IN|IS|NULL|EXISTS|COUNT|DISTINCT|OPTIONAL|MERGE|'
                    r'CREATE|FOREACH|ON|DETACH|toLower|toUpper|size|count|sum|'
                    r'avg|min|max|collect|keys|labels|type|id|elementId|node|'
                    r'score)\b',
                    stripped,
                    re.IGNORECASE,
                )
            ):
                break
        cypher_lines.append(line)

    text = '\n'.join(cypher_lines).strip()

    # 4. Strip trailing semicolons (Neo4j driver doesn't want them)
    text = text.rstrip(';').strip()

    return text


class CypherGenerator:
    """Generates Cypher queries from natural language using Gemini."""

    def __init__(self, schema: str):
        self.schema = schema
        self.llm = get_llm()

    def generate(self, question: str) -> str:
        """Generate a Cypher query for the given natural language question."""
        chain = CYPHER_GENERATION_PROMPT | self.llm
        response = chain.invoke({
            "schema": self.schema,
            "question": question,
        })
        return _extract_cypher(response.content)

    def correct(self, question: str, original_query: str, issues: str) -> str:
        """Attempt to auto-correct a Cypher query based on identified issues."""
        chain = CYPHER_CORRECTION_PROMPT | self.llm
        response = chain.invoke({
            "schema": self.schema,
            "original_query": original_query,
            "issues": issues,
            "question": question,
        })
        return _extract_cypher(response.content)
