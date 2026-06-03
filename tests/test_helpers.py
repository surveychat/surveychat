"""tests/test_helpers.py: unit tests for the three pure helper functions
defined in app.py: validate_passcode_routing, build_api_messages, and
build_transcript.

External dependencies (streamlit, dotenv, openai) are replaced with
MagicMock objects by conftest.py before this module is imported.
"""

import sys
from unittest.mock import call

import pytest

# ---------------------------------------------------------------------------
# Import the module under test.
# conftest.py has already patched streamlit/dotenv/openai in sys.modules,
# so this import executes app.py's top-level code against the mocks.
# ---------------------------------------------------------------------------
import app


# ===========================================================================
#  validate_passcode_routing
# ===========================================================================

class TestValidatePasscodeRouting:
    """Tests for validate_passcode_routing(conditions, n_conditions)."""

    def _st(self):
        """Return the mocked streamlit module."""
        return sys.modules["streamlit"]

    def _reset_st(self):
        """Clear any previous st.error / st.stop call records."""
        st = self._st()
        st.error.reset_mock()
        st.stop.reset_mock()

    # --- happy-path cases ---------------------------------------------------

    def test_survey_mode_no_passcodes_ok(self):
        """Single condition without a passcode is valid (survey mode)."""
        self._reset_st()
        conditions = [{"name": "A", "system_prompt": "…", "model": "m"}]
        app.validate_passcode_routing(conditions, 1)
        self._st().stop.assert_not_called()

    def test_all_conditions_have_unique_passcodes_ok(self):
        """Two conditions each with a unique passcode is valid."""
        self._reset_st()
        conditions = [
            {"name": "A", "passcode": "ALPHA", "system_prompt": "…", "model": "m"},
            {"name": "B", "passcode": "BETA",  "system_prompt": "…", "model": "m"},
        ]
        app.validate_passcode_routing(conditions, 2)
        self._st().stop.assert_not_called()

    def test_passcodes_case_insensitive_uniqueness_ok(self):
        """Passcodes that differ only in case are treated as duplicates."""
        self._reset_st()
        conditions = [
            {"name": "A", "passcode": "alpha", "system_prompt": "…", "model": "m"},
            {"name": "B", "passcode": "ALPHA", "system_prompt": "…", "model": "m"},
        ]
        app.validate_passcode_routing(conditions, 2)
        self._st().error.assert_called_once()
        self._st().stop.assert_called_once()

    def test_extra_conditions_beyond_n_are_ignored(self):
        """Only the first n_conditions entries are validated."""
        self._reset_st()
        conditions = [
            {"name": "A", "passcode": "ALPHA", "system_prompt": "…", "model": "m"},
            {"name": "B", "passcode": "BETA",  "system_prompt": "…", "model": "m"},
            # Extra entry deliberately missing a passcode; should be ignored.
            {"name": "C", "system_prompt": "…", "model": "m"},
        ]
        app.validate_passcode_routing(conditions, 2)
        self._st().stop.assert_not_called()

    # --- error cases --------------------------------------------------------

    def test_partial_passcode_configuration_triggers_error(self):
        """Only some conditions having a passcode is an error."""
        self._reset_st()
        conditions = [
            {"name": "A", "passcode": "ALPHA", "system_prompt": "…", "model": "m"},
            {"name": "B",                      "system_prompt": "…", "model": "m"},
        ]
        app.validate_passcode_routing(conditions, 2)
        self._st().error.assert_called_once()
        self._st().stop.assert_called_once()

    def test_blank_passcode_triggers_error(self):
        """A passcode that is an empty string (or only whitespace) is an error."""
        self._reset_st()
        conditions = [
            {"name": "A", "passcode": "ALPHA", "system_prompt": "…", "model": "m"},
            {"name": "B", "passcode": "   ",   "system_prompt": "…", "model": "m"},
        ]
        app.validate_passcode_routing(conditions, 2)
        self._st().error.assert_called_once()
        self._st().stop.assert_called_once()

    def test_duplicate_passcodes_trigger_error(self):
        """Two conditions sharing the same passcode is an error."""
        self._reset_st()
        conditions = [
            {"name": "A", "passcode": "SAME", "system_prompt": "…", "model": "m"},
            {"name": "B", "passcode": "SAME", "system_prompt": "…", "model": "m"},
        ]
        app.validate_passcode_routing(conditions, 2)
        self._st().error.assert_called_once()
        self._st().stop.assert_called_once()

    def test_three_arm_all_unique_ok(self):
        """Three conditions each with a unique passcode is valid."""
        self._reset_st()
        conditions = [
            {"name": "A", "passcode": "ALPHA", "system_prompt": "…", "model": "m"},
            {"name": "B", "passcode": "BETA",  "system_prompt": "…", "model": "m"},
            {"name": "C", "passcode": "GAMMA", "system_prompt": "…", "model": "m"},
        ]
        app.validate_passcode_routing(conditions, 3)
        self._st().stop.assert_not_called()


# ===========================================================================
#  build_api_messages
# ===========================================================================

class TestBuildApiMessages:
    """Tests for build_api_messages(conversation, system_prompt)."""

    def test_empty_conversation_returns_system_message_only(self):
        """With no prior conversation, only the system message is returned."""
        result = app.build_api_messages([], "Be helpful.")
        assert result == [{"role": "system", "content": "Be helpful."}]

    def test_conversation_is_prepended_with_system_message(self):
        """The system message is always the first element."""
        conversation = [
            {"role": "user",      "content": "Hello",     "timestamp": "…"},
            {"role": "assistant", "content": "Hi there!", "timestamp": "…"},
        ]
        result = app.build_api_messages(conversation, "System prompt.")
        assert result[0] == {"role": "system", "content": "System prompt."}

    def test_timestamps_are_stripped_from_output(self):
        """The 'timestamp' key must not appear in the messages sent to the API."""
        conversation = [
            {"role": "user", "content": "Test", "timestamp": "2026-01-01T00:00:00+00:00"},
        ]
        result = app.build_api_messages(conversation, "Prompt.")
        for msg in result:
            assert "timestamp" not in msg

    def test_roles_and_content_are_preserved(self):
        """Role and content are forwarded exactly as-is."""
        conversation = [
            {"role": "user",      "content": "Question?", "timestamp": "t1"},
            {"role": "assistant", "content": "Answer.",   "timestamp": "t2"},
        ]
        result = app.build_api_messages(conversation, "System.")
        assert result[1] == {"role": "user",      "content": "Question?"}
        assert result[2] == {"role": "assistant", "content": "Answer."}

    def test_output_length_equals_conversation_plus_one(self):
        """The returned list has exactly len(conversation) + 1 items."""
        conversation = [{"role": "user", "content": f"msg {i}", "timestamp": ""} for i in range(5)]
        result = app.build_api_messages(conversation, "System.")
        assert len(result) == 6


# ===========================================================================
#  build_transcript
# ===========================================================================

class TestBuildTranscript:
    """Tests for build_transcript(messages)."""

    def test_returns_dict_with_messages_key(self):
        """The transcript is a dict with a 'messages' key."""
        result = app.build_transcript([])
        assert isinstance(result, dict)
        assert "messages" in result

    def test_empty_conversation_gives_empty_messages_list(self):
        result = app.build_transcript([])
        assert result["messages"] == []

    def test_user_role_relabelled_to_participant(self):
        """'user' must be renamed 'participant' in the transcript."""
        messages = [{"role": "user", "content": "Hello", "timestamp": "t"}]
        result = app.build_transcript(messages)
        assert result["messages"][0]["role"] == "participant"

    def test_assistant_role_unchanged(self):
        """'assistant' role must remain 'assistant'."""
        messages = [{"role": "assistant", "content": "Hi", "timestamp": "t"}]
        result = app.build_transcript(messages)
        assert result["messages"][0]["role"] == "assistant"

    def test_content_is_preserved(self):
        messages = [{"role": "user", "content": "My answer.", "timestamp": "t"}]
        result = app.build_transcript(messages)
        assert result["messages"][0]["content"] == "My answer."

    def test_timestamp_is_included(self):
        messages = [{"role": "user", "content": "x", "timestamp": "2026-03-06T14:22:01+00:00"}]
        result = app.build_transcript(messages)
        assert result["messages"][0]["timestamp"] == "2026-03-06T14:22:01+00:00"

    def test_multi_turn_conversation_length(self):
        """Every message in the input appears in the transcript."""
        messages = [
            {"role": "user",      "content": "Q1", "timestamp": "t1"},
            {"role": "assistant", "content": "A1", "timestamp": "t2"},
            {"role": "user",      "content": "Q2", "timestamp": "t3"},
            {"role": "assistant", "content": "A2", "timestamp": "t4"},
        ]
        result = app.build_transcript(messages)
        assert len(result["messages"]) == 4

    def test_condition_name_and_model_are_not_exposed(self):
        """Condition name and model identifiers must not appear in the transcript."""
        messages = [{"role": "user", "content": "Hello", "timestamp": "t"}]
        result = app.build_transcript(messages)
        for msg in result["messages"]:
            assert "condition" not in msg
            assert "model" not in msg


# ===========================================================================
#  mask_unshared_messages
# ===========================================================================

class TestMaskUnsharedMessages:
    """Tests for mask_unshared_messages(messages, unshared_indices)."""

    def test_unshared_participant_message_is_redacted(self):
        messages = [
            {"role": "user", "content": "Private detail", "timestamp": "t1"},
        ]
        result = app.mask_unshared_messages(messages, {0})
        assert result[0]["content"] == "Message unshared by participant"

    def test_following_assistant_reply_is_redacted(self):
        messages = [
            {"role": "user", "content": "Private detail", "timestamp": "t1"},
            {"role": "assistant", "content": "You said private detail.", "timestamp": "t2"},
        ]
        result = app.mask_unshared_messages(messages, {0})
        assert (
            result[1]["content"]
            == "Assistant response hidden because the previous participant message was unshared"
        )

    def test_later_messages_are_preserved(self):
        messages = [
            {"role": "user", "content": "Private detail", "timestamp": "t1"},
            {"role": "assistant", "content": "You said private detail.", "timestamp": "t2"},
            {"role": "user", "content": "Share this", "timestamp": "t3"},
        ]
        result = app.mask_unshared_messages(messages, {0})
        assert result[2]["content"] == "Share this"

    def test_original_messages_are_not_mutated(self):
        messages = [
            {"role": "user", "content": "Private detail", "timestamp": "t1"},
            {"role": "assistant", "content": "You said private detail.", "timestamp": "t2"},
        ]
        app.mask_unshared_messages(messages, {0})
        assert messages[0]["content"] == "Private detail"
        assert messages[1]["content"] == "You said private detail."
