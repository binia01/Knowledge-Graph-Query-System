"""
Query classifier — determines whether a user question requires
graph traversal, vector search, hybrid search, or agent-based decomposition.
"""

from langchain_core.prompts import ChatPromptTemplate
from src.llm import get_llm

CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a query classifier for a Stack Overflow knowledge graph system.
Given a user question, classify it into ONE of the following categories:

1. "graph" — The answer can be found through direct graph traversal
   (explicit relationships like who answered what, tags, user stats, etc.)
   Examples: "Who answered the most questions about Python?", "What tags are most frequently used with 'neo4j'?",
   "Show me all questions posted by user 'JohnDoe'"

2. "vector" — The answer requires semantic similarity / meaning-based search
   (recommendations, thematic similarity, finding questions by content)
   Examples: "Find questions about optimizing recursive functions", "How do I fix a NullPointerException?",
   "Recommend questions similar to 'How to parse JSON in Python?'"

3. "hybrid" — The answer requires BOTH semantic similarity AND graph relationships
   Examples: "Find questions about memory leaks in C++ but only from users with high reputation",
   "Show me python questions similar to 'list comprehension' that have the 'performance' tag"

4. "agent" — The answer requires multiple dependent steps where the next step
   depends on results from previous steps
   Examples: "Compare the top answerers for Python vs Java",
   "Which user has the most accepted answers on questions about 'multithreading'?"

5. "ambiguous" — The question is too vague or unclear to determine what data is needed
   Examples: "Show me connections", "Tell me about code",
   "What's interesting?"

Respond with ONLY a JSON object:
{{"type": "<graph|vector|hybrid|agent|ambiguous>", "reason": "<brief explanation>"}}""",
    ),
    ("human", "{question}"),
])


class QueryClassifier:
    """Classifies user questions into query types."""

    VALID_TYPES = {"graph", "vector", "hybrid", "agent", "ambiguous"}

    def __init__(self):
        self.llm = get_llm()

    def classify(self, question: str) -> dict:
        """
        Classify a question and return {"type": str, "reason": str}.
        """
        import json
        import re

        chain = CLASSIFICATION_PROMPT | self.llm
        response = chain.invoke({"question": question})
        raw = response.content.strip()

        # Strip markdown fences
        if raw.startswith("```"):
            raw = re.sub(r"```(?:json)?\s*", "", raw)
            raw = raw.replace("```", "").strip()

        try:
            data = json.loads(raw)
            qtype = data.get("type", "graph")
            if qtype not in self.VALID_TYPES:
                qtype = "graph"
            return {"type": qtype, "reason": data.get("reason", "")}
        except json.JSONDecodeError:
            # Default to graph if classification fails
            return {"type": "graph", "reason": "Classification failed, defaulting to graph"}
