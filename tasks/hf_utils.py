"""
Helpers for converting HuggingFace dataset conversation formats to nanochat messages.
"""

SYSTEM_ROLES = {"system"}
USER_ROLES = {"user", "human", "instruction", "prompt"}
ASSISTANT_ROLES = {"assistant", "gpt", "model", "bot", "response"}
TOOL_ROLES = {"function_response", "function", "tool", "tool_response"}


def _map_role(role_raw, is_first):
    role = str(role_raw).strip().lower()
    if role in SYSTEM_ROLES:
        return "system" if is_first else "user"
    if role in USER_ROLES:
        return "user"
    if role in ASSISTANT_ROLES:
        return "assistant"
    if role in TOOL_ROLES:
        return "user"
    raise ValueError(f"Unknown role '{role_raw}' in conversation")


def _merge_same_role(messages):
    merged = []
    for msg in messages:
        if merged and msg["role"] == merged[-1]["role"]:
            merged[-1]["content"] += "\n\n" + msg["content"]
        else:
            merged.append(msg)
    return merged


def _validate_messages(messages):
    assert messages, "Conversation must have at least one message"
    if messages[0]["role"] == "system":
        start = 1
    else:
        start = 0
    for i, msg in enumerate(messages[start:], start=0):
        expected = "user" if i % 2 == 0 else "assistant"
        assert msg["role"] == expected, f"Message {i} has role {msg['role']} but should be {expected}"
        assert isinstance(msg["content"], str), "Message content must be a string"


def convert_conversations(conversations, role_key="from", content_key="value"):
    messages = []
    for item in conversations:
        role = _map_role(item[role_key], is_first=len(messages) == 0)
        content = item[content_key]
        messages.append({"role": role, "content": content})
    messages = _merge_same_role(messages)
    _validate_messages(messages)
    return messages
