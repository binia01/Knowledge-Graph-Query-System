"""
Conversation memory module — maintains chat history so the system
can handle follow-up questions with context.

Uses langchain_core.chat_history.InMemoryChatMessageHistory instead of
the deprecated langchain.memory.ConversationBufferMemory.
"""

from langchain_core.chat_history import InMemoryChatMessageHistory


class ConversationMemory:
    """Manages conversation history for follow-up question support."""

    def __init__(self):
        self.history = InMemoryChatMessageHistory()

    def add_exchange(self, question: str, answer: str):
        """Store a question-answer exchange in memory."""
        self.history.add_user_message(question)
        self.history.add_ai_message(answer)

    def get_history_string(self) -> str:
        """Return the full conversation history as a formatted string."""
        messages = self.history.messages
        if not messages:
            return ""

        history_parts = []
        for msg in messages:
            role = "User" if msg.type == "human" else "Assistant"
            history_parts.append(f"{role}: {msg.content}")

        return "\n".join(history_parts)

    def get_messages(self) -> list[dict]:
        """Return conversation history as a list of dicts for the dashboard."""
        return [
            {"role": "user" if msg.type == "human" else "assistant", "content": msg.content}
            for msg in self.history.messages
        ]

    def clear(self):
        """Clear all conversation history."""
        self.history.clear()
