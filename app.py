# =============================================================================
#  surveychat - Chatbot Surveys and Randomized Experiments
# =============================================================================
#
#  PURPOSE
#  -------
#  surveychat supports two use modes:
#
#  Survey mode  (N_CONDITIONS = 1)
#    Every participant talks to the same chatbot.  No randomization.
#    Access can optionally be gated with a shared passcode by adding a
#    "passcode" key to the single condition dict.  Omit it to let
#    participants go straight to the chat.
#    Use this for open-ended interviews, cognitive
#    interviewing, pilot testing, or any qualitative data collection that
#    benefits from a conversational format rather than a plain text box.
#    Examples: exploratory interviews, pilot testing, cognitive debriefs,
#    and any study where adaptive follow-up questions driven by participant
#    responses would produce richer data than a fixed question list.
#
#  Experiment mode  (N_CONDITIONS >= 2)
#    Participants are routed to one of N chatbot "conditions", each defined
#    by a unique system prompt and model choice.  Every condition must have
#    a unique passcode.  Your survey tool (e.g. Qualtrics) shows each
#    participant the right code before they open the chat link; entering it
#    routes them to exactly that condition.  Use this for A/B tests or
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
#      OpenRouter:        https://openrouter.ai/api/v1
#        (one key; access Claude, Gemini, Llama, Mistral, Grok, and 300+
#         models - set "model" to e.g. "anthropic/claude-3-5-sonnet",
#         "google/gemini-2.5-pro", "meta-llama/llama-3.3-70b-instruct")
#      Groq:              https://api.groq.com/openai/v1
#        (fast inference; model e.g. "llama-3.3-70b-versatile")
#      Mistral AI:        https://api.mistral.ai/v1
#        (model e.g. "mistral-large-latest")
#      Azure via LiteLLM: https://your-proxy.azurewebsites.net
#      Local (LM Studio, Ollama, llama.cpp, vLLM): http://localhost:1234/v1
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
import time
from datetime import datetime, timezone

# ── Third-party ───────────────────────────────────────────────────────────────
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv          # reads .env into os.environ automatically

# Load the .env file so that OPENAI_API_KEY is available via os.environ
# even when the app is run without pre-exporting it in the shell.
# override=True ensures .env always takes precedence over any stale value
# that may already be set in the process environment.
load_dotenv(override=True)


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  ✏️  RESEARCHER CONFIGURATION - edit this section to set up your study    ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

# ── LLM API settings ──────────────────────────────────────────────────────────
#
#  API_BASE_URL  The base URL for your LLM API endpoint.
#                - OpenAI (default):
#                    "https://api.openai.com/v1"
#                - OpenRouter (access Claude, Gemini, Llama, Mistral, Groq,
#                  and 300+ models with one API key):
#                    "https://openrouter.ai/api/v1"
#                - Anthropic (Claude) via OpenRouter:
#                    set API_BASE_URL = "https://openrouter.ai/api/v1"
#                    set "model" to e.g. "anthropic/claude-3-5-sonnet"
#                - Google Gemini via OpenRouter:
#                    set API_BASE_URL = "https://openrouter.ai/api/v1"
#                    set "model" to e.g. "google/gemini-2.5-pro"
#                - Groq (fast open-source models):
#                    "https://api.groq.com/openai/v1"
#                    set "model" to e.g. "llama-3.3-70b-versatile"
#                - Mistral AI:
#                    "https://api.mistral.ai/v1"
#                    set "model" to e.g. "mistral-large-latest"
#                - HuggingFace Inference API:
#                    "https://api-inference.huggingface.co/v1"
#                  (set OPENAI_API_KEY to your HuggingFace token;
#                   set "model" to the HF model ID, e.g.
#                   "meta-llama/Llama-3.3-70B-Instruct")
#                - Azure OpenAI via LiteLLM proxy:
#                    "https://your-proxy.azurewebsites.net"
#                - Local model (LM Studio, Ollama, llama.cpp, vLLM):
#                    "http://localhost:1234/v1"
#
#  The API key is read from OPENAI_API_KEY in the .env file - do not paste
#  keys directly here.  For OpenRouter, use your OpenRouter key here.
API_BASE_URL = "https://llmproxy.uva.nl/v1"

# ── How many chatbot conditions does your study have? ─────────────────────────
#
#   N_CONDITIONS = 1   →  Survey mode.  Every participant talks to the same
#                          chatbot.  No randomization.
#                          Add a "passcode" to CONDITIONS[0] to gate access
#                          with a shared entry code; omit it to let
#                          participants go straight to the chat.
#                          Use this for structured or semi-structured
#                          interviews, pilot testing, cognitive debriefs, or
#                          any study where a conversation replaces a plain
#                          text-entry question.
#
#   N_CONDITIONS = 2   →  Experiment mode, classic A/B test.
#                          Each condition requires a unique "passcode".
#                          Participants receive their code from the survey
#                          tool and enter it to reach their condition.
#
#   N_CONDITIONS = 3+  →  Experiment mode, multi-arm.
#                          Same as above: a unique passcode is required on
#                          every condition.
#
#   Default: 2
N_CONDITIONS = 2

# ── Define each chatbot condition ─────────────────────────────────────────────
#
#  Add one dictionary per condition.  You MUST have at least N_CONDITIONS
#  entries.  Any extra entries beyond N_CONDITIONS are silently ignored.
#
#  Fields per condition
#  --------------------
#  "name"           Short internal label used in log messages and debug info.
#                   Never shown to participants.  Keep it descriptive enough
#                   to identify the condition when reviewing data or logs.
#
#  "passcode"       Passcode that routes participants to this condition.
#                   Required in experiment mode (N_CONDITIONS > 1): every
#                   active condition must have a unique passcode.  Configure
#                   your survey tool to show each participant their code
#                   before they open the chat link.
#                   Optional in survey mode (N_CONDITIONS = 1): include a
#                   passcode to gate access with a shared entry code, or
#                   omit it to let participants go straight to the chat.
#                   Matching is case-insensitive ("alpha" == "ALPHA").
#
#  "system_prompt"  The hidden instruction sent to the model at the very
#                   start of every conversation.  Participants never see
#                   this text, but it defines the chatbot's entire persona,
#                   tone, and behavioral boundaries.
#
#                   In survey mode, treat this as an interviewer brief:
#                   describe the study topic, the interview style, how to
#                   handle off-topic responses, and when to wrap up.
#
#                   In experiment mode, make sure the prompts differ clearly
#                   between conditions so the manipulation is strong and its
#                   effects are detectable in your outcome measures.
#
#  "model"          The model identifier string for this condition.
#                   Common options (OpenAI):
#                     "gpt-4o"        - GPT-4o (capable, widely available)
#                     "gpt-4o-mini"   - GPT-4o Mini, faster and cheaper
#                     "gpt-4.1"       - GPT-4.1
#                     "o4-mini"       - OpenAI reasoning model
#                   Via OpenRouter (set API_BASE_URL to OpenRouter first):
#                     "anthropic/claude-sonnet-4-5"  - Claude Sonnet
#                     "anthropic/claude-haiku-3-5"   - Claude Haiku, fast/cheap
#                     "google/gemini-2.5-pro"         - Gemini 2.5 Pro
#                     "meta-llama/llama-3.3-70b-instruct" - Llama 3.3
#                   Via Groq (fast inference, set API_BASE_URL accordingly):
#                     "llama-3.3-70b-versatile"
#                   Different conditions can use different models if you want
#                   to directly compare model-level effects.
#
#  "initial_message"  (optional) The first message shown in the chat,
#                   sent by the assistant before the participant types
#                   anything.  Use this to open the conversation with a
#                   scripted question rather than waiting for the
#                   participant to initiate - common in structured and
#                   semi-structured interview protocols.
#                   Omit or set to "" for no opening message (participant
#                   types first).
#
#  "temperature"    (optional) Per-condition response randomness.
#                   Overrides the global TEMPERATURE setting for this
#                   condition only.  Useful in experiments that directly
#                   compare model consistency across arms.
#
#  "max_tokens"     (optional) Per-condition token cap.
#                   Overrides the global MAX_TOKENS setting for this
#                   condition only.
#
#  Tips
#  ----
#  - In experiment mode use short, neutral passcodes ("ALPHA"/"BETA",
#    colours, animals) that give participants no hint of their condition.
#  - System prompts work best when they specify tone, task, and limits all
#    at once.  Vague prompts produce inconsistent behavior across sessions.
#  - Test each condition manually before launching the study.
#
#  Survey mode example (N_CONDITIONS = 1):
#  ─────────────────────────────────────────────────────────────────────
#  CONDITIONS = [
#      {
#          "name":          "Social-media interview bot",
#          "passcode":      "STUDY2026",   # remove this line to skip the gate
#          "system_prompt": (
#              "You are a friendly research interviewer studying how people "
#              "use social media in their daily lives.  Ask one open-ended "
#              "question at a time, listen carefully, and ask follow-up "
#              "questions to explore the participant's experience in depth. "
#              "After 5-7 exchanges, thank the participant warmly and let "
#              "them know they can click End chat."
#          ),
#          "model": "gpt-4o",
#      },
#  ]

CONDITIONS = [

    # ── Condition A ───────────────────────────────────────────────────────────
    {
        "name":          "Condition A - Neutral",
        "passcode":      "ALPHA",      # routes participants to this condition
        "system_prompt": (
            "You are a neutral, information-focused research assistant participating "
            "in an academic study. Respond to the participant's messages in a clear, "
            "factual, and impersonal manner. "
            "Do not acknowledge feelings, offer encouragement, or use warm language. "
            "Do not say things like 'I understand', 'that makes sense', or 'thank you "
            "for sharing'. "
            "If the participant raises an emotional or personal topic, respond only to "
            "its factual content and do not comment on the emotional dimension. "
            "Keep every response to two or three sentences. Be direct and concise."
        ),
        "model": "gpt-4o",
        "initial_message": "Hello. I'm here to assist you as part of this study. What would you like to discuss?",
    },

    # ── Condition B ───────────────────────────────────────────────────────────
    {
        "name":          "Condition B - Empathetic",
        "passcode":      "BETA",       # routes participants to this condition
        "system_prompt": (
            "You are a warm, empathetic research assistant participating in an "
            "academic study. Your role is to make the participant feel genuinely "
            "heard and understood. "
            "Always begin your response by acknowledging the participant's feelings "
            "or perspective - reflect back what they said or validate their experience "
            "before you offer any information or ask a follow-up question. "
            "Use a caring, conversational tone throughout. When a participant shares "
            "something personal or emotionally significant, spend time on that before "
            "moving on - never rush past it. "
            "Ask one warm follow-up question at the end of each response to invite "
            "the participant to share more. "
            "Never sound clinical, detached, or bureaucratic."
        ),
        "model": "gpt-4o",
        "initial_message": "Hello! I'm really glad you're here. This is a space where you can share whatever's on your mind. What would you like to talk about today?",
    },

    # ── Add more conditions below by copying the block above ─────────────────
    # {
    #     "name":          "Condition C - Socratic",
    #     "passcode":      "GAMMA",
    #     "system_prompt": (
    #         "You are a Socratic research assistant participating in an academic "
    #         "study. Your role is to help the participant think through topics "
    #         "more deeply by asking carefully chosen questions rather than "
    #         "providing answers or information directly. "
    #         "Never volunteer your own opinion, conclusion, or recommendation. "
    #         "Instead, respond to each message by reflecting back what the "
    #         "participant seems to be assuming or implying, then posing one "
    #         "probing question that invites them to examine that assumption, "
    #         "consider a counterexample, or articulate their reasoning more "
    #         "precisely. "
    #         "Questions should be open-ended and genuinely exploratory - not "
    #         "leading questions that hint at a preferred answer. "
    #         "If the participant asks you a direct question, turn it back to them "
    #         "with a question that helps them work toward their own answer. "
    #         "Keep the conversational pressure gentle but persistent: always end "
    #         "your turn with exactly one question, never more. "
    #         "Do not summarise, conclude, or wrap up the conversation - your goal "
    #         "is continued, deepening inquiry."
    #     ),
    #     "model": "mistral-small-3.2",  # or any other model
    # },

]

# ── Optional model parameters ─────────────────────────────────────────────────
#
#  TEMPERATURE  Controls response randomness / creativity.
#               0.0  = deterministic, highly consistent across sessions
#               1.0  = default for most models, balanced
#               >1.0 = more varied / creative, less predictable
#               Set to None to use the model's default.
#               In survey research, values between 0.7 and 1.0 usually work
#               well.  Lower values reduce variability between participants,
#               which can be useful for standardised interview protocols.
TEMPERATURE = None   # e.g. 0.8, or None to use the model's default

#  MAX_TOKENS   Hard cap on the number of tokens the model generates per reply.
#               Prevents runaway responses and controls costs.
#               Set to None for no cap (model decides).
#               A typical assistant turn is 50-250 tokens; 512 is a safe cap
#               for most survey/interview use cases.
MAX_TOKENS = None    # e.g. 512, or None for no cap

#  MAX_EXCHANGES  Hard limit on the number of participant turns.
#                 When the participant sends their Nth message, the assistant
#                 replies as normal, and the transcript is then shown
#                 automatically - no "End chat" click required.
#                 Set to None for no limit (participant ends manually).
#                 Recommended for standardised interview protocols where all
#                 participants should receive the same number of exchanges.
#                 Example: MAX_EXCHANGES = 6 gives a 6-turn interview.
MAX_EXCHANGES = None   # e.g. 6, or None for no limit

# ── Study title (shown in the browser tab and as the page heading) ────────────
STUDY_TITLE = "surveychat"

# ── Welcome / instruction message shown above the chat input ─────────────────
#
#   Displayed in a shaded banner at the top of the chat interface.  Use it
#   to orient participants before they start typing.
#
#   Good uses:
#     - Task framing:  "In this part of the study you will discuss your
#       recent online shopping experiences with an AI assistant."
#     - Consent reminder:  "This conversation is recorded as part of a
#       research study and will be stored securely."
#     - Behavioural instruction:  "Please respond as you normally would.
#       There are no right or wrong answers."
#
#   Set to "" to show no banner - useful if your Qualtrics page already
#   provides full instructions before the participant opens the chat link.
#
#   HTML is supported - use <strong>, <em>, <br> etc. for emphasis.
#
#   Examples:
#
#       WELCOME_MESSAGE = ""   # no banner
#
#       WELCOME_MESSAGE = (
#           "Welcome. In this part of the study you will have a short "
#           "conversation with an AI assistant about climate change. "
#           "When you are done, click <strong>End chat</strong> "
#           "to copy your transcript and paste it into the survey."
#       )
#
#       WELCOME_MESSAGE = (
#           "This conversation is part of a research study on AI-assisted "
#           "decision-making.  Your responses are confidential and will only "
#           "be used for research purposes.<br><br>"
#           "When finished, click <strong>End chat</strong> to copy "
#           "your transcript and paste it into the survey."
#       )
WELCOME_MESSAGE = ""

# ── Prompt shown on the passcode entry screen ────────────────────────────────
#
#   Displayed above the passcode text box whenever all active conditions
#   define a "passcode".  Ignored when no conditions define a passcode.
PASSCODE_ENTRY_PROMPT = "Enter your passcode below to start the conversation."

# ── Configuration reference ───────────────────────────────────────────────────
#
#  Variable               Default           Description
#  ──────────────────────────────────────────────────────────────────────────────
#  API_BASE_URL           (OpenAI URL)      Base URL for the LLM API endpoint.
#  N_CONDITIONS           2                 1 = survey mode, 2 = A/B test,
#                                           3+ = multi-arm experiment.
#  CONDITIONS             [A, B]            List of condition dicts.  Each has
#                                           "name", "passcode" (required for
#                                           N > 1, optional for N = 1),
#                                           "system_prompt", and "model".
#  TEMPERATURE            None              Response randomness (0.0–2.0).
#                                           None = model default (usually 1.0).
#                                           Override per-condition with
#                                           "temperature" in condition dict.
#  MAX_TOKENS             None              Max tokens per assistant reply.
#                                           None = no cap.
#                                           Override per-condition with
#                                           "max_tokens" in condition dict.
#  MAX_EXCHANGES          None              Max participant turns before chat
#                                           auto-ends. None = no limit.
#  STUDY_TITLE            "surveychat"      Browser tab title and page heading.
#  WELCOME_MESSAGE        ""                Banner shown above the chat.
#                                           Set to "" to hide.
#  PASSCODE_ENTRY_PROMPT  (see app)         Text above the passcode box.
#                                           Shown whenever a passcode is set.
#  ──────────────────────────────────────────────────────────────────────────────
#
#  Routing behaviour summary
#  ─────────────────────────
#  N = 1, no passcode   Survey mode.  No gate, direct to chat.
#  N = 1, with passcode Survey mode with gate.  Shared passcode required.
#  N > 1                Experiment mode.  A unique passcode is required on
#                       every condition.  Each code maps to exactly one
#                       condition, stable across page refreshes.
#
# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  END OF RESEARCHER CONFIGURATION - no edits needed below this line        ║
# ╚═════════════════════════════════════════════════════════════════════════════╝


# =============================================================================
#  HELPER FUNCTIONS
# =============================================================================
#
#  Four helper functions used by the main interface below:
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
#
#  mask_unshared_messages(messages, unshared_indices)
#    Redacts participant-selected messages before transcript export, along
#    with the next assistant reply because it may quote or summarize the
#    hidden participant text.

def validate_passcode_routing(conditions: list, n_conditions: int) -> None:
    """
    Check passcode-routing configuration and halt the app on any inconsistency.

    Enforces three invariants:
      1. In experiment mode (n_conditions > 1), every active condition must
         define a "passcode" field.  Partial configuration (some but not
         all conditions have a passcode) is also rejected in any mode.
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

    # Invariant 1a: Experiment mode requires a passcode on every condition.
    # Participants must be traceable to a known arm, so random routing is
    # not supported when N > 1.
    if n_conditions > 1 and len(passcoded) == 0:
        st.error(
            f"Experiment mode requires a `\"passcode\"` on every condition, "
            f"but none of the **{n_conditions}** active conditions define one. "
            "Add a unique passcode to each condition in the CONDITIONS list."
        )
        st.stop()

    # Invariant 1b: Partial configuration is always an error.
    # Either every active condition has a passcode or none do.
    if 0 < len(passcoded) < n_conditions:
        st.error(
            f"Passcode configuration is incomplete: **{len(passcoded)}** of "
            f"**{n_conditions}** active conditions have a `\"passcode\"` field. "
            "Every active condition must have a passcode."
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
        has "role", "content", and "timestamp".
    system_prompt : str
        The hidden system prompt from the active condition dict.

    Returns
    -------
    list[dict]
        A list of {"role": str, "content": str} dicts ready for the chat
        completions endpoint.
    """
    messages = [{"role": "system", "content": system_prompt}]
    for m in conversation:
        messages.append({"role": m["role"], "content": m["content"]})
    return messages


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
    entries = []
    for m in messages:
        entries.append({
            "role":      "participant" if m["role"] == "user" else "assistant",
            "content":   m["content"],
            "timestamp": m.get("timestamp", ""),
        })
    return {"messages": entries}


def mask_unshared_messages(messages: list, unshared_indices: set[int]) -> list:
    """
    Redact participant-selected messages before transcript export.

    If a participant hides one of their own messages, also hide the
    immediately following assistant reply. Assistant replies often quote,
    summarize, or directly answer the previous participant turn, so leaving
    them visible could accidentally reveal the message the participant chose
    not to share.
    """
    hidden_assistant_indices = set()
    for idx in unshared_indices:
        next_idx = idx + 1
        if (
            next_idx < len(messages)
            and messages[next_idx].get("role") == "assistant"
        ):
            hidden_assistant_indices.add(next_idx)

    masked_messages = []
    for idx, message in enumerate(messages):
        masked = dict(message)
        if idx in unshared_indices:
            masked["content"] = "Message unshared by participant"
        elif idx in hidden_assistant_indices:
            masked["content"] = (
                "Assistant response hidden because the previous participant "
                "message was unshared"
            )
        masked_messages.append(masked)

    return masked_messages


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

/* ── Optional exclusion expander ───────────────────────────────────────────── */
/* Rendered above the copy button; kept visually quiet so participants
   focus on copying rather than on the optional exclusion option. */
[data-testid="stExpander"] details { border: none !important; background: transparent !important; }
[data-testid="stExpander"] summary { font-size: 0.8rem !important; color: #888 !important; padding-left: 0 !important; }
[data-testid="stExpander"] summary:hover { color: #555 !important; }

/* ── Welcome / instruction banner ───────────────────────────────────────────── */
/* Shown above the chat input when WELCOME_MESSAGE is non-empty.  The left
   accent border matches the primary color to tie it to the site palette. */
.welcome-banner {
    background: #EFF1F3;
    border-left: 4px solid #5C6C79;
    border-radius: 0 6px 6px 0;
    padding: 0.75rem 1rem;
    color: #1F2429;
    margin-bottom: 1.25rem;
    line-height: 1.55;
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

# Validate passcode-routing configuration (covers all modes including N = 1).
# Full logic is documented in validate_passcode_routing() above.
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
#  Survey (no gate) →  N_CONDITIONS = 1, no "passcode" on the condition.
#                      Condition index is always 0.  Participant goes
#                      straight to the chat interface.
#
#  Passcode routing →  Every active condition defines a "passcode".
#                      Covers survey mode with a shared gate (N = 1) and
#                      all experiment mode configurations (N > 1, required).
#                      The passcode entry screen is shown before the chat.
#                      The same passcode always maps to the same condition
#                      index, so routing is stable across page refreshes
#                      without any server-side session storage.
_passcode_routing = all(
    "passcode" in CONDITIONS[i] for i in range(N_CONDITIONS)
)

# ── Assign condition index ────────────────────────────────────────────────────
#
#  For ungated survey mode (N = 1, no passcode), assign immediately.
#  For passcode routing, defer until the participant enters their passcode;
#  assignment happens in the passcode-gate block below.
if not _passcode_routing and "condition_index" not in st.session_state:
    st.session_state["condition_index"] = 0  # only reachable for N = 1, no passcode

# ── Per-session flags ─────────────────────────────────────────────────────────

# Whether the passcode gate has been passed.
# Initialised to True only when no gate is needed (N = 1, no passcode).
if "passcode_accepted" not in st.session_state:
    st.session_state["passcode_accepted"] = not _passcode_routing

# Whether the participant has ended the chat session.
# Flips to True when they confirm End; triggers the transcript panel.
if "chat_ended" not in st.session_state:
    st.session_state["chat_ended"] = False

# Two-step end-confirmation flag.
# First click on "End chat" sets this to True (arming the confirmation).
# Second click on "✓ Confirm" sets chat_ended to True and shows the transcript.
# This prevents accidental chat termination and loss of the conversation.
if "confirm_end" not in st.session_state:
    st.session_state["confirm_end"] = False

# Flipped to True when the chat ends automatically because MAX_EXCHANGES was
# reached (as opposed to the participant clicking End chat manually).
# Used to show a brief completion message at the top of the transcript panel.
if "auto_ended" not in st.session_state:
    st.session_state["auto_ended"] = False

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
    # Explicitly set Authorization in default_headers in addition to passing
    # api_key.  Some versions of the openai SDK do not forward the auth header
    # reliably when base_url points to a non-OpenAI endpoint (e.g. OpenRouter,
    # LiteLLM proxies).  Setting it here guarantees the header is always sent.
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers={
            "Authorization": f"Bearer {api_key}",
        },
    )

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
#  Stage 1 - Passcode gate  (any mode where all conditions define a passcode)
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
#    • A JavaScript clipboard button copies the full transcript JSON with
#      one click, turning green once copied.
#    • A collapsed expander lets the participant optionally exclude
#      individual messages before copying (all included by default).
#    • The participant pastes the transcript into their survey tool.
#
# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    f'<div class="app-header">'
    f'<div class="app-title">💬 {STUDY_TITLE}</div>'
    f'</div>',
    unsafe_allow_html=True,
)

# ── Passcode entry (shown whenever all conditions define a passcode) ─────────
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
        f'<p style="margin-bottom:1rem;font-size:1rem;color:#1F2429">'
        f'{PASSCODE_ENTRY_PROMPT}</p>',
        unsafe_allow_html=True,
    )
    with st.form("key_form"):
        _code = st.text_input("Passcode", placeholder="Enter your passcode")
        _submitted = st.form_submit_button("Start →", type="primary")
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

# ── Seed initial assistant message if configured ──────────────────────────────
# If the condition defines an "initial_message", inject it as the first
# assistant turn so the bot opens the conversation before the participant
# types anything.  This is only done once: when messages is still empty.
# Any subsequent reruns skip this block because messages is non-empty.
_initial_msg = condition.get("initial_message", "").strip()
if _initial_msg and not st.session_state["messages"]:
    st.session_state["messages"].append({
        "role":      "assistant",
        "content":   _initial_msg,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

# ── End Chat button - appears after the first exchange ────────────────────────
# Placed below the header so it does not compete with the title layout.
if not st.session_state["chat_ended"] and st.session_state["has_sent_message"]:
    _, end_col = st.columns([4, 2])
    with end_col:
        if not st.session_state["confirm_end"]:
            if st.button("End chat", use_container_width=True, type="secondary"):
                st.session_state["confirm_end"] = True
                st.rerun()
        else:
            # Second click required to confirm - prevents accidental endings
            if st.button("✓ Confirm end", use_container_width=True, type="primary"):
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

        # Stream the model's response token-by-token.
        with st.chat_message("assistant"):
            try:
                # Per-condition temperature/max_tokens override the global
                # defaults, falling back to TEMPERATURE and MAX_TOKENS if the
                # condition dict does not define them.
                _call_kwargs = {
                    "model":    condition["model"],
                    "messages": api_messages,
                }
                _temp     = condition.get("temperature", TEMPERATURE)
                _max_tok  = condition.get("max_tokens",  MAX_TOKENS)
                if _temp is not None:
                    _call_kwargs["temperature"] = _temp
                if _max_tok is not None:
                    _call_kwargs["max_tokens"] = _max_tok

                _call_kwargs["stream"] = True
                stream   = client.chat.completions.create(**_call_kwargs)

                def _throttled(s):
                    for chunk in s:
                        yield chunk
                        time.sleep(0.05)

                response = st.write_stream(_throttled(stream))

                # Some proxy implementations return an empty stream instead of
                # raising an exception on error (e.g. rate-limit 429).  Treat
                # an empty response as a failure so the error handler fires.
                if not response:
                    raise RuntimeError(
                        "The model returned an empty response. "
                        "This may be a rate-limit or temporary API issue. "
                        "Please wait a moment and try again."
                    )

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

        # Save the completed assistant response to history.
        if response:
            st.session_state["messages"].append({
                "role":      "assistant",
                "content":   response,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        # Auto-end when MAX_EXCHANGES limit is reached.
        # Count only participant turns (role == "user") so that a seeded
        # initial_message (role == "assistant") does not affect the tally.
        if MAX_EXCHANGES is not None:
            _user_turns = sum(
                1 for m in st.session_state["messages"] if m["role"] == "user"
            )
            if _user_turns >= MAX_EXCHANGES:
                st.session_state["chat_ended"] = True
                st.session_state["auto_ended"] = True
                st.rerun()

        # On the very first user exchange, force a rerun so the End button
        # becomes visible immediately.  Count only user turns so this fires
        # correctly whether or not an initial_message is configured.
        _user_turns_now = sum(
            1 for m in st.session_state["messages"] if m["role"] == "user"
        )
        if _user_turns_now == 1:
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
else:
    # Show a brief completion notice when the chat was auto-ended by
    # MAX_EXCHANGES (as opposed to the participant clicking End chat).
    if st.session_state.get("auto_ended"):
        st.info(
            "The conversation is now complete. "
            "Please copy your transcript below and paste it into the survey."
        )

    # Collect participant message indices and build transcript first.
    # Checkbox state persists in st.session_state across rerenders, so the
    # JSON is always up-to-date even though the checkboxes render below.
    _participant_indices = []
    for _i, _msg in enumerate(st.session_state["messages"]):
        if _msg["role"] == "assistant":
            continue
        _participant_indices.append(_i)

    # Build transcript now, reflecting current checkbox states.
    _unshared = {
        _i for _i in _participant_indices
        if not st.session_state.get(f"share_msg_{_i}", True)
    }
    _msgs_for_transcript = mask_unshared_messages(
        st.session_state["messages"],
        _unshared,
    )
    _transcript_json = json.dumps(
        build_transcript(_msgs_for_transcript), indent=2, ensure_ascii=False
    )
    # Safe JS string literal. Escape closing script sequences because
    # participant-entered text can appear inside the JSON transcript.
    _js_str = json.dumps(_transcript_json).replace("</", "<\\/")

    with st.expander("Optional: exclude a message before sharing"):
        st.caption(
            "Uncheck any messages you'd prefer not to share. "
            "The next assistant reply will be hidden too."
        )
        for _i in _participant_indices:
            _msg = st.session_state["messages"][_i]
            _is_shared = st.session_state.get(f"share_msg_{_i}", True)
            _col_cb, _col_msg = st.columns([1, 11])
            with _col_cb:
                st.checkbox(
                    "include",
                    value=True,
                    key=f"share_msg_{_i}",
                    label_visibility="collapsed",
                )
            with _col_msg:
                with st.chat_message("user"):
                    if _is_shared:
                        st.markdown(_msg["content"])
                    else:
                        st.markdown(f"~~{_msg['content']}~~")

    st.html(
        f"""
        <style>
          #copy-btn {{
            width: 100%; padding: 0.55rem 1rem;
            font-size: 1rem; font-weight: 600;
            background: #ff4b4b; color: white;
            border: none; border-radius: 0.5rem; cursor: pointer;
          }}
          #copy-btn:hover {{ background: #e03535; }}
          #copy-btn:disabled {{ background: #21c354; cursor: default; }}
          #fallback {{ display: none; margin-top: 0.75rem; font-size: 0.85rem; color: #555; }}
          #fallback textarea {{
            width: 100%; height: 80px; font-size: 0.75rem;
            font-family: monospace; margin-top: 0.25rem;
          }}
        </style>
        <button id="copy-btn">
          &#10003;&nbsp; Copy your conversation transcript
        </button>
        <div id="fallback">
          <p>Automatic copy failed. Select all and copy manually:</p>
          <textarea id="fallback-ta" readonly></textarea>
        </div>
        <script>
        (function() {{
          var btn = document.getElementById('copy-btn');
          var fb = document.getElementById('fallback');
          var ta = document.getElementById('fallback-ta');
          var text = {_js_str};
          btn.addEventListener('click', function() {{
            function onSuccess() {{
              btn.textContent = '\u2713 Copied! Paste it in the below question to proceed.';
              btn.disabled = true;
            }}
            function onFail() {{
              btn.style.display = 'none';
              ta.value = text;
              fb.style.display = 'block';
              ta.focus(); ta.select();
            }}
            if (navigator.clipboard && window.isSecureContext) {{
              navigator.clipboard.writeText(text).then(onSuccess, onFail);
            }} else {{
              try {{
                var ta2 = document.createElement('textarea');
                ta2.value = text;
                ta2.style.cssText = 'position:fixed;left:-9999px';
                document.body.appendChild(ta2);
                ta2.focus(); ta2.select();
                document.execCommand('copy');
                document.body.removeChild(ta2);
                onSuccess();
              }} catch(e) {{ onFail(); }}
            }}
          }});
        }})();
        </script>
        """,
        unsafe_allow_javascript=True,
    )


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
#    - Store the passcode shown to each participant in a Qualtrics embedded
#      data field.  Map passcode → condition name in R/Python during analysis.
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
