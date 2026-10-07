"""
Simple in-memory conversation history, keyed by WhatsApp sender number.

This is intentionally lightweight for an MVP: history lives in a Python
dict and resets whenever the server restarts. If you need history to
survive restarts (or to run across multiple server processes), swap the
dict below for a small SQLite table or a Redis store — the get/add
functions are the only things you'd need to change.
"""

# How many past exchanges (user + assistant pairs) to keep per sender.
# Higher = more context for follow-up questions, but more tokens sent to
# the LLM on every request (slower + eats into free-tier rate limits faster).
MAX_TURNS = 5

# {sender: [{"role": "user"/"assistant", "content": "..."}, ...]}
_conversations: dict[str, list[dict]] = {}


def get_history(sender: str) -> list[dict]:
    """Return the stored conversation history for this sender (oldest first)."""
    return _conversations.get(sender, [])


def add_turn(sender: str, user_message: str, assistant_reply: str) -> None:
    """Append a user/assistant exchange to this sender's history, trimming old turns."""
    history = _conversations.setdefault(sender, [])
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": assistant_reply})

    # Keep only the most recent MAX_TURNS exchanges (each exchange = 2 messages)
    max_messages = MAX_TURNS * 2
    if len(history) > max_messages:
        del history[: len(history) - max_messages]


def clear_history(sender: str) -> None:
    """Wipe history for a sender — useful if they want to start a fresh topic."""
    _conversations.pop(sender, None)