"""
conftest.py: pytest configuration for surveychat tests.

app.py is a self-contained Streamlit application: importing it runs
top-level Streamlit calls (st.set_page_config, st.error, st.stop, etc.)
and also calls load_dotenv() and constructs an OpenAI client.

These patches are applied at module level (not inside a fixture) so that
they are in place before pytest imports test_helpers.py at collection time.
"""

import os
import sys
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Ensure a dummy API key is present so app.py's key-validation passes.
# ---------------------------------------------------------------------------
os.environ.setdefault("OPENAI_API_KEY", "sk-test-key")

# ---------------------------------------------------------------------------
# Replace streamlit, dotenv, and openai with mocks BEFORE any test module
# imports app.py.  Module-level conftest code runs at collection time, which
# is earlier than any fixture lifecycle.
# ---------------------------------------------------------------------------
_st_mock = MagicMock()
# Pre-populate session_state with the keys that app.py expects at the top
# level so that UI rendering code after the passcode gate doesn't raise
# KeyError when the module is imported during test collection.
_st_mock.session_state = {
    "passcode_accepted": True,
    "condition_index":   0,
    "chat_ended":        False,
    "confirm_end":       False,
    "has_sent_message":  False,
    "limit_reached":     False,
    "messages":          [],
}
_st_mock.cache_resource = lambda fn: fn   # no-op decorator
# st.columns(...) is unpacked into two variables by the chat layout (e.g.
# `_end_col, _ = st.columns([2, 4])`).  A bare MagicMock iterates as empty and
# would raise on unpack, so return a fixed pair of context-manager mocks.
_st_mock.columns.return_value = [MagicMock(), MagicMock()]

sys.modules.setdefault("streamlit",             _st_mock)
sys.modules.setdefault("streamlit.components",      MagicMock())
sys.modules.setdefault("streamlit.components.v1",   MagicMock())
sys.modules.setdefault("dotenv",                 MagicMock())
sys.modules.setdefault("openai",                 MagicMock())
