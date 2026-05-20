# =============================================================================
#  surveychat - Chatbot Surveys and Randomized Experiments
# =============================================================================
#
#  PURPOSE
#  -------
#  surveychat supports two use modes:
#
#  Survey mode  (N_CONDITIONS = 1)
#    Every participant talks to the same chatbot.  No passcodes, no
#    randomization.  Use this for open-ended interviews, cognitive
#    interviewing, pilot testing, or any qualitative data collection that
#    benefits from a conversational format rather than a plain text box.
#    Examples: exploratory interviews, pilot testing, cognitive debriefs,
#    and any study where adaptive follow-up questions driven by participant
#    responses would produce richer data than a fixed question list.
#
#  Experiment mode  (N_CONDITIONS >= 2)
#    Participants are routed to one of N chatbot "conditions", each defined
#    by a unique system prompt and model choice.  Use this for A/B tests or
#    multi-arm studies that compare how different chatbot styles affect
#    participant responses, attitudes, or behaviour.
#    Examples: comparing empathetic vs. neutral interviewers, testing
#    different question orderings, or manipulating response thoroughness.
#
#  In both modes the participant chats, clicks End, and copies a JSON
#  transcript back into the parent survey tool (e.g. Qualtrics).
#
#  TRANSCRIPT FORMAT
#  -----------------
#  After clicking End, the participant receives a JSON block:
#
#      {
#        "messages": [
#          {
#            "role":      "participant",
#            "content":   "Hello!",
#            "timestamp": "2026-03-06T14:22:01.123456+00:00"
#          },
#          {
#            "role":      "assistant",
#            "content":   "Hi there, how can I help you today?",
#            "timestamp": "2026-03-06T14:22:03.456789+00:00"
#          }
#        ]
#      }
#
#  Timestamps are UTC ISO-8601 with an explicit +00:00 offset so they are
#  unambiguous across time zones.  Condition name and model are deliberately
#  excluded so participants cannot infer their assigned arm.
#
#  Parse in Python:
#      import json, pandas as pd
#      data = json.loads(transcript_string)
#      df   = pd.DataFrame(data["messages"])   # one row per turn
#
#  Parse in R:
#      library(jsonlite)
#      data <- fromJSON(transcript_string)
#      df   <- as.data.frame(data$messages)    # one row per turn
#
#  INTEGRATION WITH SURVEY TOOLS
#  ------------------------------
#  Survey mode:
#    (1) Add a Text / Graphic block in Qualtrics with a link to the app.
#    (2) After the chat, add a Text Entry question where participants
#        paste their transcript.
#
#  Experiment mode:
#    (1) Use Qualtrics Survey Flow > Randomizer to split participants.
#    (2) In each arm display the matching passcode and the app URL.
#    (3) After the chat, add a Text Entry question for the transcript.
#    (4) Export responses - treatment assignment is recovered from the
#        passcode stored in the relevant Qualtrics branch variable.
#
#  DEPLOYMENT OPTIONS
#  ------------------
#  Local development:
#      streamlit run app.py
#
#  Streamlit Community Cloud (free, no server needed):
#      Push the repo to GitHub, connect at share.streamlit.io, and add
#      OPENAI_API_KEY under Advanced settings → Secrets.
#
#  Cloud VM (e.g. AWS EC2, DigitalOcean, Azure):
#      pip install -r requirements.txt
#      streamlit run app.py --server.port 80 --server.headless true
#      Serve HTTPS via Caddy or nginx (required for Qualtrics iFrame embeds).
#
#  LLM PROVIDERS
#  -------------
#  Set API_BASE_URL to any chat-completions-compatible endpoint:
#      OpenAI:            https://api.openai.com/v1
#      Azure via LiteLLM: https://your-proxy.azurewebsites.net
#      OpenRouter:        https://openrouter.ai/api/v1
#      Local (LiteLLM):   http://localhost:4000
#
#  QUICK START
#  -----------
#  1. Edit the RESEARCHER CONFIGURATION section below.
#  2. Add your OPENAI_API_KEY to the .env file (see .env.example).
#  3. Run:   streamlit run app.py
#
#  FORKING & REUSE
#  ---------------
#  This file is intentionally self-contained.  The only section you need
#  to edit for most studies is the RESEARCHER CONFIGURATION block below.
#  Everything else - session management, participant routing, transcript
#  export, and the chat UI - is handled for you automatically.
#
# =============================================================================


# ── Standard library ──────────────────────────────────────────────────────────
import json
import os
import random
from datetime import datetime, timezone

# ── Third-party ───────────────────────────────────────────────────────────────
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv          # reads .env into os.environ automatically

from config import * # Reading the config files

# Load the .env file so that OPENAI_API_KEY is available via os.environ
# even when the app is run without pre-exporting it in the shell.
load_dotenv()


# =============================================================================
#  HELPER FUNCTIONS
# =============================================================================
#
#  Three helper functions used by the main interface below:
#
#  validate_passcode_routing(conditions, n_conditions)
#    Checks that the passcode configuration is internally consistent and
#    halts the app with an actionable error message if not.  Called during
#    the configuration-validation phase, before any participant-facing UI
#    is rendered.
#
#  build_api_messages(conversation, system_prompt)
#    Constructs the full message list sent to the LLM API for each turn,
#    prepending the hidden system prompt at position 0.
#
#  build_transcript(messages)
#    Formats the conversation history as the JSON transcript object shown
#    to the participant at the end of the session.

def validate_passcode_routing(conditions: list, n_conditions: int) -> None:
    """
    Check passcode-routing configuration and halt the app on any inconsistency.

    Enforces three invariants:
      1. If any active condition defines a "passcode" field, every active
         condition must define one (no partial configuration).
      2. Every passcode value must be a non-empty string after stripping
         leading and trailing whitespace.
      3. All passcodes must be unique when compared case-insensitively.

    Any violation triggers a descriptive on-screen error via st.error() and
    stops execution with st.stop(), so researchers see the problem
    immediately rather than discovering it mid-study.

    Parameters
    ----------
    conditions : list[dict]
        The full CONDITIONS list from the researcher configuration section.
    n_conditions : int
        The N_CONDITIONS value.  Only the first n_conditions entries are
        considered active; any extras are ignored.
    """
    active    = conditions[:n_conditions]
    passcoded = [c for c in active if "passcode" in c]

    # Invariant 1: Partial configuration - some but not all conditions define
    # a passcode.  Either every arm needs a passcode (passcode routing) or
    # none do (random routing).  A mixed state is always a mistake.
    if 0 < len(passcoded) < n_conditions:
        st.error(
            f"Passcode routing is partially configured: **{len(passcoded)}** of "
            f"**{n_conditions}** active conditions have a `\"passcode\"` field. "
            "Either add a `\"passcode\"` to every condition or remove them all."
        )
        st.stop()

    if len(passcoded) == n_conditions:
        # Invariant 2: No blank passcode strings.
        if any(not c["passcode"].strip() for c in active):
            st.error(
                "One or more condition `\"passcode\"` values are empty strings. "
                "Every passcode must contain at least one character."
            )
            st.stop()

        # Invariant 3: All passcodes must be unique (case-insensitive).
        passcodes = [c["passcode"].strip().lower() for c in active]
        if len(passcodes) != len(set(passcodes)):
            st.error(
                "Two or more conditions share the same `\"passcode\"` value. "
                "Every condition must have a unique passcode."
            )
            st.stop()


def build_api_messages(conversation: list, system_prompt: str) -> list:
    """
    Construct the message list to send to the LLM API for a single turn.

    The system prompt is inserted as a {"role": "system"} message at
    position 0.  Participants never see this text, but it defines the
    model's entire persona and behavioral instructions for the conversation.

    Only "role" and "content" are forwarded from the conversation history.
    The "timestamp" key is local-only metadata that the chat completions API
    does not accept and would cause a validation error if included.

    Parameters
    ----------
    conversation : list[dict]
        The current value of st.session_state["messages"].  Each element
        has "role" ("user" or "assistant"), "content", and "timestamp" keys.
    system_prompt : str
        The hidden system prompt from the active condition dict.

    Returns
    -------
    list[dict]
        A list of {"role": str, "content": str} dicts ready for the
        chat completions endpoint.
    """
    return (
        [{"role": "system", "content": system_prompt}]
        + [
            {"role": m["role"], "content": m["content"]}
            for m in conversation
        ]
    )

def request_api(condition, api_messages):
    """
    Send the current conversation turn to the appropriate API.

    This function chooses between two API paths based on the active
    experimental condition:

    - If condition["mode"] is "file_search", it uses the Responses API with
      the file_search tool enabled. The vector store ID comes from
      condition["context"], and the conversation is passed through the
      Responses API's `input` argument.

    - Otherwise, it uses the Chat Completions API. The conversation is passed
      through the Chat Completions API's `messages` argument.

    Parameters
    ----------
    condition : dict
        Dictionary describing the active condition. Expected keys are:

        - "mode": Determines which API path to use. The value "file_search"
          enables the Responses API with the file_search tool. Any other value
          uses Chat Completions.
        - "model": The OpenAI model name to use.
        - "context": The vector store ID to use for file_search mode.

    api_messages : list[dict]
        The prepared conversation history, usually produced by
        `build_api_messages(...)`. Each message should include only fields
        accepted by the API, such as "role" and "content".

    Returns
    -------
    generator
        A generator that yields plain text chunks from the model response.
        This return value is suitable for passing directly to
        `st.write_stream(...)`.

    Raises
    ------
    KeyError
        If required keys such as "mode", "model", or "context" are missing
        from `condition`.
    """
    if condition["mode"] == "file_search":
        reply = client.responses.create(
            model=condition["model"],
            input=api_messages,
            tools=[
                {
                    "type": "file_search",
                    "vector_store_ids": [condition["context"]],
                    "max_num_results": 2,
                    "ranking_options": {
                        "score_threshold": 0.35,
                    },
                }
            ],
            service_tier="priority",
            stream=True,
        )

    else:
        reply = client.chat.completions.create(
            model=condition["model"],
            messages=api_messages,
            stream=True,
        )

    return stream_openai_text(reply)

def stream_openai_text(reply):
    """
    Convert an OpenAI streaming response into plain text chunks.
    """
    for event in reply:
        if getattr(event, "type", None) == "response.output_text.delta":
            yield event.delta
        elif hasattr(event, "choices") and event.choices:
            content = getattr(event.choices[0].delta, "content", None)
            if content:
                yield content
    
def build_transcript(messages: list) -> dict:
    """
    Format the conversation history as the transcript object shown after chat ends.

    Returns a JSON-serialisable dict with a single "messages" key.  Each
    entry carries:
      - "role"      : "participant" (relabelled from "user") or "assistant"
      - "content"   : the full text of the message
      - "timestamp" : UTC ISO-8601 string, e.g. "2026-03-06T14:22:01+00:00"

    Design notes:
      - "user" is relabelled "participant" so researchers get a domain-
        appropriate label when parsing the transcript in Python or R.
      - Condition name and model are intentionally excluded.  In experiment
        mode, participants must not be able to infer their assigned condition
        from the transcript they read and manually copy back into the survey.
        Treatment assignment is recovered separately from the passcode stored
        in the survey platform's response data.
      - In survey mode (N_CONDITIONS = 1) there is only one condition, so
        excluding the name is a no-op, but it keeps the transcript format
        identical across both modes.

    Parameters
    ----------
    messages : list[dict]
        The current value of st.session_state["messages"].

    Returns
    -------
    dict
        Transcript object suitable for json.dumps(indent=2, ensure_ascii=False).
    """
    return {
        "messages": [
            {
                "role":      "participant" if m["role"] == "user" else "assistant",
                "content":   m["content"],
                "timestamp": m.get("timestamp", ""),
            }
            for m in messages
        ],
    }


# =============================================================================
#  PAGE & STYLE SETUP
# =============================================================================

st.set_page_config(
    page_title=STUDY_TITLE,
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Clean, minimal stylesheet - accent colors kept in sync with config.toml.
# Streamlit renders its own theme via CSS-in-JS and does not expose theme
# values as CSS custom properties, so we hardcode the palette here.
# If you change colors in .streamlit/config.toml, update these too:
#
#   PRIMARY   = #5C6C79   (primaryColor)             - borders, accents
#   TEXT      = #1F2429   (textColor)                - body text
#   BG_SEC    = #EFF1F3   (secondaryBackgroundColor)  - banner backgrounds
st.markdown("""
<style>
/* ── Typography ────────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

/* Apply Inter to the entire app, overriding Streamlit's default font. */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Chrome removal ─────────────────────────────────────────────────────────── */
/* Hide the Streamlit toolbar, footer, and hamburger menu so the page looks
   like a standalone app rather than a Streamlit dashboard. */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="collapsedControl"] { display: none; }

/* ── Page layout ────────────────────────────────────────────────────────────── */
/* Constrain to a readable column width and reduce default top padding. */
.block-container { max-width: 740px; padding-top: 2.25rem; padding-bottom: 1rem; }

/* ── Transcript code block ──────────────────────────────────────────────────── */
/* Wrap long JSON lines so participants on narrow screens can read everything
   without horizontal scrolling. */
.stCode pre { white-space: pre-wrap; word-break: break-word; }

/* ── App header ─────────────────────────────────────────────────────────────── */
/* A thin rule below the study title separates it visually from the chat. */
.app-header {
    border-bottom: 2px solid #5C6C79;
    padding-bottom: 0.65rem;
    margin-bottom: 1.5rem;
}
.app-title {
    font-size: 1.35rem;
    font-weight: 600;
    color: #1F2429;
    letter-spacing: -0.4px;
    margin: 0;
}

/* ── Welcome / instruction banner ───────────────────────────────────────────── */
/* Shown above the chat input when WELCOME_MESSAGE is non-empty.  The left
   accent border matches the primary color to tie it to the site palette. */
.welcome-banner {
    background: #EFF1F3;
    border-left: 4px solid #5C6C79;
    border-radius: 0 6px 6px 0;
    padding: 0.75rem 1rem;
    font-size: 0.9rem;
    color: #1F2429;
    margin-bottom: 1.25rem;
    line-height: 1.55;
}

/* ── Transcript panel ───────────────────────────────────────────────────────── */
/* Shown after the participant clicks End.  Slightly more prominent border
   than the welcome banner to draw attention to the copy instruction. */
.transcript-banner {
    background: #EFF1F3;
    border: 1px solid #e5e7eb;
    border-left: 4px solid #5C6C79;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1.1rem;
    font-size: 0.85rem;
    color: #1F2429;
    margin-bottom: 1.25rem;
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
#  ENVIRONMENT & CONFIGURATION VALIDATION
# =============================================================================

# Read the API key from the environment (populated from .env above).
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Fail fast with a clear, actionable error if the API key is missing or blank.
if not OPENAI_API_KEY or not OPENAI_API_KEY.strip():
    st.error(
        "**OPENAI_API_KEY not found or empty.**  "
        "Please add it to your `.env` file and restart the application.\n\n"
        "Example `.env`:\n```\nOPENAI_API_KEY=sk-...\n```"
    )
    st.stop()

# Validate researcher configuration - surfaces common setup mistakes early.
if N_CONDITIONS < 1:
    st.error(
        "`N_CONDITIONS` must be at least **1**. "
        "Please update the Researcher Configuration section."
    )
    st.stop()

if len(CONDITIONS) < N_CONDITIONS:
    st.error(
        f"`CONDITIONS` list has **{len(CONDITIONS)}** "
        f"entr{'y' if len(CONDITIONS) == 1 else 'ies'}, "
        f"but `N_CONDITIONS` is set to **{N_CONDITIONS}**. "
        "Please add more condition definitions or reduce `N_CONDITIONS`."
    )
    st.stop()

# Validate passcode-routing configuration when N > 1.
# Full logic is documented in validate_passcode_routing() above.
if N_CONDITIONS > 1:
    validate_passcode_routing(CONDITIONS, N_CONDITIONS)


# =============================================================================
#  SESSION STATE INITIALIZATION
# =============================================================================
#
#  Streamlit re-runs the entire script on every user interaction (button
#  click, chat message, page refresh).  Any Python variable assigned during
#  one run is lost on the next.  st.session_state is the mechanism for
#  persisting values across reruns within a single browser session.
#
#  Each `if … not in st.session_state` guard ensures values are initialised
#  exactly once - on the participant's very first page load - and left
#  unchanged on every subsequent rerun.

# ── Determine routing mode ────────────────────────────────────────────────────
#
#  Survey mode      →  N_CONDITIONS = 1.
#                      No routing step.  Condition index is always 0.
#                      Participant goes straight to the chat interface.
#
#  Passcode routing →  N > 1 AND every active condition defines a "passcode".
#                      The passcode entry gate is shown before the chat.
#                      The same passcode always resolves to the same condition
#                      index, so a participant who refreshes the page and
#                      re-enters their passcode lands on the same arm -
#                      without any server-side session storage.
#
#  Random routing   →  N > 1 BUT no conditions define a "passcode".
#                      Condition is drawn uniformly at random on first load.
#                      A page refresh draws a new condition, so this mode is
#                      only appropriate when refresh is unlikely or impossible
#                      (e.g. the survey platform embeds the link once).
_passcode_routing = N_CONDITIONS > 1 and all(
    "passcode" in CONDITIONS[i] for i in range(N_CONDITIONS)
)

# ── Assign condition index ────────────────────────────────────────────────────
#
#  For survey/random routing, assign immediately.
#  For passcode routing, defer until the participant enters their passcode;
#  assignment happens in the passcode-gate block below.
if not _passcode_routing and "condition_index" not in st.session_state:
    st.session_state["condition_index"] = (
        0 if N_CONDITIONS == 1 else random.randint(0, N_CONDITIONS - 1)
    )

# ── Per-session flags ─────────────────────────────────────────────────────────

# Whether the passcode gate has been passed.
# Initialised to True when no gate is needed (survey / random routing).
if "passcode_accepted" not in st.session_state:
    st.session_state["passcode_accepted"] = not _passcode_routing

# Whether the participant has ended the chat session.
# Flips to True when they confirm End; triggers the transcript panel.
if "chat_ended" not in st.session_state:
    st.session_state["chat_ended"] = False

# Two-step end-confirmation flag.
# First click on "End this chat" sets this to True (arming the confirmation).
# Second click on "✓ Confirm" sets chat_ended to True and shows the transcript.
# This prevents accidental chat termination and loss of the conversation.
if "confirm_end" not in st.session_state:
    st.session_state["confirm_end"] = False

# Flipped to True the moment the participant sends their first message.
# The End button is hidden until this is True to avoid showing a useless
# button before any conversation has happened.
if "has_sent_message" not in st.session_state:
    st.session_state["has_sent_message"] = False

# The full conversation history for this session.
# Each item is a dict: {"role": str, "content": str, "timestamp": str}.
# Grows by one entry per user message and one per assistant reply.
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ── LLM client ────────────────────────────────────────────────────────────────

@st.cache_resource
def get_client(api_key: str, base_url: str) -> OpenAI:
    """
    Create and cache a singleton LLM client.

    @st.cache_resource creates the object once, shares it across all reruns
    and browser sessions on the same server, and never serialises it to disk.
    This is the correct Streamlit pattern for connection-like objects.

    Parameters
    ----------
    api_key : str
        The API key read from the environment (OPENAI_API_KEY).
    base_url : str
        The API_BASE_URL set in the researcher configuration.

    Returns
    -------
    OpenAI
        A configured client instance.
    """
    return OpenAI(api_key=api_key, base_url=base_url)

client = get_client(OPENAI_API_KEY, API_BASE_URL)


# =============================================================================
#  MAIN CHAT INTERFACE
# =============================================================================
#
#  The interface is rendered in a single linear pass from top to bottom.
#  Streamlit's execution model means every widget call below is conditional
#  on session-state flags set during earlier runs; this drives the multi-step
#  participant flow:
#
#  Stage 1 - Passcode gate  (experiment mode with passcode routing only)
#    • Displayed when st.session_state["passcode_accepted"] is False.
#    • A form with a single text input collects the passcode.
#    • Valid entry maps to a condition index, sets passcode_accepted=True,
#      and triggers a full rerun so stage 1 is skipped on subsequent runs.
#    • Invalid entry shows an inline error; the gate remains visible.
#    • st.stop() at the end of stage 1 prevents any subsequent code from
#      running until the gate is passed - the chat UI is never rendered
#      even partially for unauthenticated participants.
#
#  Stage 2 - Active chat
#    • Displayed when chat_ended is False.
#    • The optional welcome banner is rendered first.
#    • All messages in st.session_state["messages"] are replayed in order
#      so the full conversation history is visible on every rerun.
#    • st.chat_input() blocks further execution until the participant sends
#      a message; the user message is appended, then the LLM is called.
#    • The response is streamed token-by-token via st.write_stream() to give
#      a natural, responsive feel even on slow connections.
#    • The End button appears in a right-aligned column after the first
#      exchange.  A two-step confirmation (End → Confirm) prevents
#      participants from accidentally discarding their conversation.
#
#  Stage 3 - Transcript panel
#    • Displayed when chat_ended is True.
#    • The transcript banner and JSON code block are rendered.
#    • Streamlit’s st.code() provides a built-in copy button in the
#      top-right corner of the block, requiring no custom JavaScript.
#    • The participant copies the JSON and pastes it back into Qualtrics
#      (or whichever survey tool they came from).
#
# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    f'<div class="app-header">'
    f'<div class="app-title">💬 {STUDY_TITLE}</div>'
    f'</div>',
    unsafe_allow_html=True,
)

# ── Passcode entry (experiment mode with passcode routing only) ─────────────
# Shown before the chat until the participant enters a valid passcode.
# On page refresh the gate reappears, but the same passcode always maps to the
# same condition, so assignment is stable across refreshes.
if not st.session_state["passcode_accepted"]:
    _passcode_map = {
        CONDITIONS[i]["passcode"].strip().lower(): i
        for i in range(N_CONDITIONS)
    }
    if WELCOME_MESSAGE:
        st.markdown(
            f'<div class="welcome-banner">{WELCOME_MESSAGE}</div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        f'<p style="margin-bottom:1rem;font-size:0.95rem;color:#1F2429">'
        f'{PASSCODE_ENTRY_PROMPT}</p>',
        unsafe_allow_html=True,
    )
    with st.form("key_form"):
        _code = st.text_input("Passcode", placeholder="e.g. BETA")
        _submitted = st.form_submit_button("Continue →", type="primary")
    if _submitted:
        _idx = _passcode_map.get(_code.strip().lower())
        if _idx is not None:
            st.session_state["condition_index"] = _idx
            st.session_state["passcode_accepted"] = True
            st.rerun()
        else:
            st.error("Code not recognised. Please check and try again.")
    st.stop()

# Passcode accepted (or not required) - condition is now resolved.
condition = CONDITIONS[st.session_state["condition_index"]]

# ── End Chat button - appears after the first exchange ────────────────────────
# Placed below the header so it does not compete with the title layout.
if not st.session_state["chat_ended"] and st.session_state["has_sent_message"]:
    _, end_col = st.columns([5, 1])
    with end_col:
        if not st.session_state["confirm_end"]:
            if st.button("End chat", use_container_width=True, type="secondary"):
                st.session_state["confirm_end"] = True
                st.rerun()
        else:
            # Second click required to confirm - prevents accidental endings
            if st.button("✓ Confirm", use_container_width=True, type="primary"):
                st.session_state["chat_ended"] = True
                st.rerun()

# ── Active chat ───────────────────────────────────────────────────────────────
if not st.session_state["chat_ended"]:

    # Optional welcome / instruction message - hidden once chatting has begun,
    # or immediately if the participant just passed through the passcode gate.
    if WELCOME_MESSAGE and not _passcode_routing and not st.session_state["has_sent_message"]:
        st.markdown(
            f'<div class="welcome-banner">{WELCOME_MESSAGE}</div>',
            unsafe_allow_html=True,
        )

    # Render conversation history.
    # Every message stored in st.session_state["messages"] is displayed on
    # each rerun, giving the participant a full view of the conversation.
    # st.chat_message() renders a colored avatar and indented bubble whose
    # style depends on the role: "user" gets a right-aligned bubble and
    # "assistant" a left-aligned one, matching familiar chat conventions.
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input - hidden once chat_ended is True
    if prompt := st.chat_input("Type your message here…"):

        prompt = prompt.strip()
        if not prompt:
            st.stop()

        # Append and immediately display the user's message
        st.session_state["messages"].append({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        st.session_state["has_sent_message"] = True  # reveal End button from now on
        with st.chat_message("user"):
            st.markdown(prompt)

        # Build the full message list for the API call.
        # See build_api_messages() in the HELPER FUNCTIONS section for details.
        api_messages = build_api_messages(
            st.session_state["messages"],
            condition["system_prompt"],
        )

        # Stream the model's response token-by-token for a natural feel
        with st.chat_message("assistant"):
            try:
                stream = request_api(condition, api_messages)
                # stream = client.chat.completions.create(
                #     model=condition["model"],
                #     messages=api_messages,
                #     stream=True,
                # )
                response = st.write_stream(stream)
            except Exception as e:
                response = None
                # Remove the user message we just appended - leaving it in
                # history without a paired assistant reply would send two
                # consecutive user turns to the API on the next message.
                st.session_state["messages"].pop()
                st.error(
                    f"**Could not reach the LLM.** "
                    f"Check your `API_BASE_URL` and `OPENAI_API_KEY`.\n\n"
                    f"Error: `{e}`"
                )

        # Save the completed assistant response to history
        if response:
            st.session_state["messages"].append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        # On the very first exchange, force a rerun so the header re-renders
        # and the End button becomes visible immediately.
        if len(st.session_state["messages"]) == 2:
            st.rerun()

# =============================================================================
#  POST-CHAT TRANSCRIPT
# =============================================================================
#
#  Shown after the participant clicks "End".
#  The transcript is a JSON object with one key:
#    - "messages": a list of turns, each with role/content/timestamp
#
#  Parsing in Python:
#      import json, pandas as pd
#      data = json.loads(transcript_string)
#      df   = pd.DataFrame(data["messages"])  # one row per turn
#
#  Parsing in R:
#      library(jsonlite)
#      data <- fromJSON(transcript_string)
#      df   <- as.data.frame(data$messages)   # one row per turn
#
#  Streamlit's st.code() block has a built-in copy button in the top-right
#  corner - one click copies everything to the clipboard.

else:
    st.markdown(
        '<div class="transcript-banner">'
        'Your chat has ended. Your transcript is below. '
        'Click the <strong>copy symbol in the top-right corner</strong> of the box, '
        'then paste it back into the survey.'
        '</div>',
        unsafe_allow_html=True,
    )

    # Build and render the transcript.
    # See build_transcript() in the HELPER FUNCTIONS section for design notes
    # on why condition name/model are excluded and how the role label works.
    transcript = build_transcript(st.session_state["messages"])

    st.code(json.dumps(transcript, indent=2, ensure_ascii=False), language="json")


# =============================================================================
#  STUDY DATA HANDLING NOTES  (for researchers)
# =============================================================================
#
#  These notes describe how to process the transcript data collected from
#  participants.  They are for researcher reference only and have no effect
#  on the running application.
#
#  PARSING THE TRANSCRIPT IN PYTHON
#  ---------------------------------
#  import json
#  import pandas as pd
#
#  raw = qualtrics_response_column   # string value from Qualtrics export
#  data = json.loads(raw)
#  df = pd.DataFrame(data["messages"])  # columns: role, content, timestamp
#
#  Useful derived columns:
#    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
#    df["turn"]      = range(len(df))
#    df["words"]     = df["content"].str.split().str.len()
#    df_user         = df[df["role"] == "participant"]
#    df_asst         = df[df["role"] == "assistant"]
#
#  PARSING THE TRANSCRIPT IN R
#  ----------------------------
#  library(jsonlite)
#
#  raw  <- qualtrics_response_column   # character vector from survey export
#  data <- fromJSON(raw)
#  df   <- as.data.frame(data$messages)  # columns: role, content, timestamp
#
#  Useful transformations:
#    df$timestamp <- as.POSIXct(df$timestamp, tz = "UTC", format = "%Y-%m-%dT%H:%M:%OS")
#    df_user <- subset(df, role == "participant")
#    df_asst <- subset(df, role == "assistant")
#
#  CONDITION ASSIGNMENT (EXPERIMENT MODE)
#  ----------------------------------------
#  Condition identity is NOT in the transcript.  Recover it from the survey
#  branching logic:
#    - In passcode routing: store the passcode shown to each participant in
#      a Qualtrics embedded data field.  Map passcode → condition name in R/Python.
#    - In random routing:   store the displayed arm label in an embedded data
#      field in the Qualtrics Survey Flow randomizer branch.
#
#  DATA QUALITY CHECKS
#  --------------------
#  Recommended minimum checks before analysis:
#    1. Verify json.loads() succeeds for every row (malformed pastes).
#    2. Drop sessions with fewer than 2 messages (participant sent nothing).
#    3. Check for unusually short response times (bot detection, inattention).
#    4. Review assistant messages flagged with "Error" (API failures mid-session).
#
# =============================================================================