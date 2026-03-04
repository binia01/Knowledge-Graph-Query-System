"""
Main pipeline orchestrator — ties together all components:
classification → generation → validation → execution → response.
"""

import logging

from langchain_core.prompts import ChatPromptTemplate

from src.config import Config
from src.cache import QueryCache
from src.cypher_generator import CypherGenerator
from src.database import Neo4jDatabase
from src.llm import get_llm
from src.memory import ConversationMemory
from src.query_classifier import QueryClassifier
from src.validator import CypherValidator
from src.vector_search import VectorSearch
from src.agent import QueryAgent

logger = logging.getLogger(__name__)


ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful Stack Overflow knowledge assistant. Convert the raw database
results into a natural, human-friendly answer. Be concise but informative.

Formatting rules:
- Use **bold** for emphasis and titles/names.
- Use numbered lists (1. 2. 3.) for ordered results.
- Use bullet points (- item) for unordered lists.
- Keep paragraphs short.

If the results are empty, say so clearly and suggest the user rephrase.

Conversation history (for context on follow-up questions):
{chat_history}""",
    ),
    (
        "human",
        """Original question: {question}
Query executed: {cypher}
Raw results: {results}

Please provide a natural language answer.""",
    ),
])

VECTOR_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful Stack Overflow recommendation assistant. Based on the
similarity search results below, provide natural recommendations or answers.
Include similarity scores as a percentage when relevant.

Formatting rules:
- Use **bold** for emphasis and titles/names.
- Use numbered lists (1. 2. 3.) for ordered results.
- Use bullet points (- item) for unordered lists.
- Keep paragraphs short.

Conversation history:
{chat_history}""",
    ),
    (
        "human",
        """Original question: {question}
Similar questions found (with similarity scores):
{results}

Please provide natural language recommendations or answers.""",
    ),
])

AMBIGUOUS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful Stack Overflow knowledge assistant. The user's question
is ambiguous or too vague to answer directly. Politely ask for clarification.
Suggest 2-3 specific ways they could rephrase their question.

Conversation history:
{chat_history}""",
    ),
    ("human", "{question}"),
])

FOLLOW_UP_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a Neo4j Cypher expert. The user is asking a follow-up question.
Use the conversation history to resolve any pronouns or references
(e.g., "they", "those", "it") to their actual values.

Rewrite the follow-up question as a complete, standalone question.

Conversation history:
{chat_history}

Return ONLY the rewritten question, nothing else.""",
    ),
    ("human", "{question}"),
])

HYBRID_FILTER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a Neo4j Cypher expert. The user's question requires a hybrid search
(vector similarity + graph filtering). Extract the graph filter part and write it
as a Cypher WHERE/MATCH clause fragment that filters the 'question' variable.

SCHEMA:
{schema}

The variable 'question' already refers to Question nodes from the vector search.
Write ONLY the additional Cypher filter clause. For example:
  AND EXISTS {{ MATCH (question)-[:TAGGED]->(:Tag {{name: 'python'}}) }}

Return ONLY the Cypher filter fragment, nothing else.""",
    ),
    ("human", "{question}"),
])


class Pipeline:
    """
    Main orchestration pipeline that processes user questions end-to-end.
    """

    def __init__(self):
        self.db = Neo4jDatabase()
        self.schema = self.db.get_schema()
        self.classifier = QueryClassifier()
        self.generator = CypherGenerator(self.schema)
        self.validator = CypherValidator(self.schema)
        self.vector_search = VectorSearch(self.db)
        self.memory = ConversationMemory()
        self.cache = QueryCache()
        self.agent = QueryAgent(self.db, self.schema, self.memory)
        self.llm = get_llm()

    def close(self):
        """Clean up resources."""
        self.db.close()

    def process_question(self, question: str) -> dict:
        """
        Process a natural language question through the full pipeline.

        Returns a dict with:
          - answer: str (natural language answer)
          - query_type: str (graph/vector/hybrid/agent/ambiguous)
          - cypher: str | None (generated Cypher, if applicable)
          - validation: dict | None (validation result, if applicable)
          - cached: bool
          - steps: list (agent steps, if applicable)
        """
        # Check cache first
        cached = self.cache.get(question)
        if cached is not None:
            return {**cached, "cached": True}

        # Resolve follow-ups using conversation history
        resolved_question = self._resolve_follow_up(question)

        # Classify the question
        classification = self.classifier.classify(resolved_question)
        query_type = classification["type"]

        # Route to the appropriate handler
        handlers = {
            "graph": self._handle_graph,
            "vector": self._handle_vector,
            "hybrid": self._handle_hybrid,
            "agent": self._handle_agent,
            "ambiguous": self._handle_ambiguous,
        }

        handler = handlers.get(query_type, self._handle_graph)
        try:
            result = handler(question, resolved_question)
        except Exception as e:
            if _is_rate_limit_error(e):
                logger.warning("Rate limit hit during %s handler: %s", query_type, e)
                return {
                    "answer": (
                        "The AI service is temporarily rate-limited. "
                        "Please wait a moment and try again."
                    ),
                    "query_type": query_type,
                    "cypher": None,
                    "validation": None,
                    "cached": False,
                    "steps": [],
                }
            raise

        result["query_type"] = query_type
        result["classification_reason"] = classification["reason"]
        result["cached"] = False

        # Store in memory and cache
        self.memory.add_exchange(question, result["answer"])
        if query_type != "ambiguous":
            self.cache.put(question, result)

        return result

    def _resolve_follow_up(self, question: str) -> str:
        """If there's conversation history, resolve pronouns/references."""
        history = self.memory.get_history_string()
        if not history:
            return question

        chain = FOLLOW_UP_PROMPT | self.llm
        response = chain.invoke({
            "chat_history": history,
            "question": question,
        })
        resolved = response.content.strip()
        return resolved if resolved else question

    def _handle_graph(self, original_question: str, resolved_question: str) -> dict:
        """Handle a graph traversal question."""
        # Generate Cypher
        cypher = self.generator.generate(resolved_question)

        # Quick sanity check — if generation returned empty, bail early
        if not cypher or not cypher.strip():
            return {
                "answer": "I could not generate a valid query for that question. Please try rephrasing.",
                "cypher": cypher,
                "validation": None,
                "steps": [],
            }

        # Validate
        validation = self.validator.validate(resolved_question, cypher)

        # Score-based routing
        if validation.score < Config.CONFIDENCE_MED:
            # Score < 0.4 → reject
            return {
                "answer": (
                    f"I'm not confident enough to run this query (score: {validation.score:.2f}). "
                    f"Issues found: {'; '.join(validation.issues)}. "
                    "Could you please rephrase your question more specifically?"
                ),
                "cypher": cypher,
                "validation": self._validation_to_dict(validation),
                "steps": [],
            }

        if validation.score < Config.CONFIDENCE_HIGH:
            # Score 0.4–0.69 → auto-correct
            issues_text = "; ".join(validation.issues)
            corrected_cypher = self.generator.correct(
                resolved_question, cypher, issues_text
            )
            # Re-validate the corrected query
            validation2 = self.validator.validate(resolved_question, corrected_cypher)
            if validation2.score >= Config.CONFIDENCE_MED:
                cypher = corrected_cypher
                validation = validation2
            else:
                return {
                    "answer": (
                        f"I tried to correct the query but still not confident "
                        f"(score: {validation2.score:.2f}). "
                        f"Issues: {'; '.join(validation2.issues)}. "
                        "Please rephrase your question."
                    ),
                    "cypher": corrected_cypher,
                    "validation": self._validation_to_dict(validation2),
                    "steps": [],
                }

        # Execute the query
        try:
            results = self.db.run_query(cypher)
        except Exception as e:
            return {
                "answer": f"Database error: {str(e)}. Please try rephrasing your question.",
                "cypher": cypher,
                "validation": self._validation_to_dict(validation),
                "steps": [],
            }

        # Generate natural language answer
        answer = self._humanize_results(original_question, cypher, results)

        return {
            "answer": answer,
            "cypher": cypher,
            "validation": self._validation_to_dict(validation),
            "steps": [],
        }

    def _handle_vector(self, original_question: str, resolved_question: str) -> dict:
        """Handle a vector similarity search question."""
        try:
            results = self.vector_search.similarity_search(resolved_question)
        except Exception as e:
            return {
                "answer": (
                    f"Vector search error: {str(e)}. "
                    "The vector index may not be set up yet. "
                    "Please run the setup script first."
                ),
                "cypher": None,
                "validation": None,
                "steps": [],
            }

        if not results:
            return {
                "answer": "No similar questions found. Try a different description or lower the similarity threshold.",
                "cypher": None,
                "validation": None,
                "steps": [],
            }

        # Format results for the LLM
        history = self.memory.get_history_string()
        chain = VECTOR_ANSWER_PROMPT | self.llm
        response = chain.invoke({
            "chat_history": history,
            "question": original_question,
            "results": str(results),
        })

        return {
            "answer": response.content.strip(),
            "cypher": f"Vector search: top {len(results)} similar questions",
            "validation": {"score": 1.0, "issues": [], "type": "vector"},
            "steps": [],
        }

    def _handle_hybrid(self, original_question: str, resolved_question: str) -> dict:
        """Handle a hybrid (vector + graph) search question."""
        # Extract the graph filter from the question
        try:
            chain = HYBRID_FILTER_PROMPT | self.llm
            response = chain.invoke({
                "schema": self.schema,
                "question": resolved_question,
            })
            cypher_filter = response.content.strip()

            # Clean up the filter
            if cypher_filter.startswith("```"):
                lines = cypher_filter.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                cypher_filter = "\n".join(lines).strip()

            results = self.vector_search.hybrid_search(resolved_question, cypher_filter)
        except Exception as e:
            # Fallback to pure vector search
            return self._handle_vector(original_question, resolved_question)

        if not results:
            return {
                "answer": "No questions matched both the semantic similarity and graph filter criteria.",
                "cypher": f"Hybrid search with filter: {cypher_filter}",
                "validation": {"score": 1.0, "issues": [], "type": "hybrid"},
                "steps": [],
            }

        history = self.memory.get_history_string()
        chain = VECTOR_ANSWER_PROMPT | self.llm
        response = chain.invoke({
            "chat_history": history,
            "question": original_question,
            "results": str(results),
        })

        return {
            "answer": response.content.strip(),
            "cypher": f"Hybrid search with filter: {cypher_filter}",
            "validation": {"score": 1.0, "issues": [], "type": "hybrid"},
            "steps": [],
        }

    def _handle_agent(self, original_question: str, resolved_question: str) -> dict:
        """Handle a complex multi-step question via the ReAct agent."""
        result = self.agent.run(resolved_question)
        return {
            "answer": result["answer"],
            "cypher": "Agent-based multi-step execution",
            "validation": {"score": 1.0, "issues": [], "type": "agent"},
            "steps": result.get("steps", []),
        }

    def _handle_ambiguous(self, original_question: str, resolved_question: str) -> dict:
        """Handle an ambiguous question by asking for clarification."""
        history = self.memory.get_history_string()
        chain = AMBIGUOUS_PROMPT | self.llm
        response = chain.invoke({
            "chat_history": history,
            "question": original_question,
        })

        return {
            "answer": response.content.strip(),
            "cypher": None,
            "validation": None,
            "steps": [],
        }

    def _humanize_results(
        self, question: str, cypher: str, results: list[dict]
    ) -> str:
        """Convert raw Neo4j results into a natural language answer."""
        history = self.memory.get_history_string()
        chain = ANSWER_PROMPT | self.llm
        response = chain.invoke({
            "chat_history": history,
            "question": question,
            "cypher": cypher,
            "results": str(results[:30]),  # Limit to avoid token overflow
        })
        return response.content.strip()

    @staticmethod
    def _validation_to_dict(validation) -> dict:
        """Convert a ValidationResult to a serializable dict."""
        return {
            "score": validation.score,
            "issues": validation.issues,
            "is_valid_syntax": validation.is_valid_syntax,
            "direction_correct": validation.direction_correct,
            "labels_correct": validation.labels_correct,
            "has_return": validation.has_return,
        }

    def get_cache_stats(self) -> dict:
        """Return cache statistics."""
        return self.cache.stats


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Return True if *exc* looks like an API rate-limit / quota error."""
    text = str(exc).lower()
    return any(kw in text for kw in ("429", "rate limit", "resource exhausted", "quota"))

    def clear_conversation(self):
        """Clear conversation history."""
        self.memory.clear()

    def clear_cache(self):
        """Clear the query cache."""
        self.cache.clear()
