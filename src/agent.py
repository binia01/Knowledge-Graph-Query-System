"""
Agent module — handles complex multi-step questions using the
ReAct pattern. The agent decides at runtime what queries to run
and how many steps are needed.
"""

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool

from src.config import Config
from src.cypher_generator import CypherGenerator
from src.database import Neo4jDatabase
from src.llm import get_llm
from src.memory import ConversationMemory


AGENT_PROMPT = PromptTemplate.from_template("""You are a Stack Overflow knowledge graph assistant with access to a Neo4j database.
You answer complex technical questions that require multiple query steps.

You have access to the following tools:

{tools}

DATABASE SCHEMA:
{schema}

Use the following format:

Question: the input question you must answer
Thought: think about what you need to do step by step
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question (write a natural, human-friendly response)

IMPORTANT RULES:
1. Break complex questions into smaller steps.
2. Use the Cypher tool to query the database.
3. Each query should be a valid Cypher query matching the schema.
4. Pay attention to relationship directions in the schema.
5. Use the results from previous steps to inform the next step.
6. Always provide a comprehensive Final Answer.

{chat_history}

Question: {input}
{agent_scratchpad}""")


class QueryAgent:
    """ReAct agent for complex multi-step questions."""

    def __init__(self, db: Neo4jDatabase, schema: str, memory: "ConversationMemory"):
        self.db = db
        self.schema = schema
        self.memory = memory
        self.llm = get_llm()
        self.cypher_gen = CypherGenerator(schema)
        self.current_question = None

    def _cypher_tool_func(self, query: str) -> str:
        """Execute a Cypher query and return results as a string."""
        from src.cypher_generator import _extract_cypher
        query = _extract_cypher(query)

        try:
            results = self.db.run_query(query)
            if not results:
                return "No results found."
            # Limit output to avoid token overflow
            result_str = str(results[:20])
            if len(results) > 20:
                result_str += f"\n... and {len(results) - 20} more results"
            return result_str
        except Exception as e:
            error_msg = str(e)
            # Try to auto-correct syntax errors once
            if "SyntaxError" in error_msg and self.current_question:
                try:
                    corrected_query = self.cypher_gen.correct(
                        question=self.current_question,
                        original_query=query,
                        issues=error_msg,
                    )
                    # Recursively try the corrected query (only once since correct() is deterministic per call)
                    # But to avoid infinite loop, we won't recurse into _cypher_tool_func, just run_query directly
                    results = self.db.run_query(corrected_query)
                    if not results:
                        return "No results found (after auto-correction)."
                    result_str = str(results[:20])
                    if len(results) > 20:
                        result_str += f"\n... and {len(results) - 20} more results"
                    return f"Auto-corrected query executed successfully.\nOriginal error: {error_msg}\nResults: {result_str}"
                except Exception as e2:
                    return f"Query error: {error_msg}\nAuto-correction failed: {str(e2)}"
            
            return f"Query error: {error_msg}"

    def run(self, question: str) -> dict:
        """
        Run the agent on a complex question.
        Returns {"answer": str, "steps": list[str]}.
        """
        self.current_question = question
        tools = [
            Tool(
                name="CypherExecutor",
                func=self._cypher_tool_func,
                description=(
                    "Execute a Cypher query on the Neo4j Stack Overflow database. "
                    "Input should be a valid Cypher query string. "
                    "Returns the query results as text."
                ),
            ),
        ]

        agent = create_react_agent(
            llm=self.llm,
            tools=tools,
            prompt=AGENT_PROMPT,
        )

        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10,
            return_intermediate_steps=True,
        )

        chat_history = self.memory.get_history_string()

        try:
            result = executor.invoke({
                "input": question,
                "schema": self.schema,
                "chat_history": chat_history,
            })

            # Extract intermediate steps for display
            steps = []
            for action, observation in result.get("intermediate_steps", []):
                steps.append({
                    "thought": getattr(action, "log", ""),
                    "tool": action.tool,
                    "input": action.tool_input,
                    "result": str(observation)[:500],
                })

            return {
                "answer": result.get("output", "I could not determine the answer."),
                "steps": steps,
            }
        except Exception as e:
            return {
                "answer": f"Agent encountered an error: {str(e)}",
                "steps": [],
            }
