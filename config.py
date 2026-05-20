# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  ✏️  RESEARCHER CONFIGURATION - edit this section to set up your study    ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

# ── LLM API settings ──────────────────────────────────────────────────────────
#
#  API_BASE_URL  The base URL for your LLM API endpoint.
#                - Azure LiteLLM proxy (default):
#                    "https://ai-research-proxy.azurewebsites.net"
#                - OpenAI:
#                    "https://api.openai.com/v1"
#                - OpenRouter:
#                    "https://openrouter.ai/api/v1"
#                - HuggingFace Inference API:
#                    "https://api-inference.huggingface.co/v1"
#                  (set OPENAI_API_KEY to your HuggingFace token;
#                   set "model" to the HF model ID, e.g.
#                   "meta-llama/Llama-3.3-70B-Instruct")
#
#  The API key is read from OPENAI_API_KEY in the .env file - do not paste
#  keys directly here.
API_BASE_URL = "https://api.openai.com/v1"

# ── How many chatbot conditions does your study have? ─────────────────────────
#
#   N_CONDITIONS = 1   →  Survey mode.  Every participant talks to the same
#                          chatbot.  No passcodes or randomization needed.
#                          The passcode gate screen is suppressed entirely;
#                          participants go straight to the chat.
#                          Use this for structured or semi-structured
#                          interviews, pilot testing, cognitive debriefs, or
#                          any study where a conversation replaces a plain
#                          text-entry question.
#
#   N_CONDITIONS = 2   →  Experiment mode, classic A/B test.
#                          Participants are split ~50 / 50 across two
#                          conditions and enter a passcode to reach their arm.
#
#   N_CONDITIONS = 3+  →  Experiment mode, multi-arm.
#                          Participants are split as evenly as possible across
#                          all conditions.
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
#                   Only needed in experiment mode (N_CONDITIONS > 1).
#                   Assign one unique passcode per condition and configure
#                   your survey tool to display the correct passcode to each
#                   participant before they open the chat link.
#                   Matching is case-insensitive ("alpha" == "ALPHA").
#                   Omit this field entirely when N_CONDITIONS = 1.
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
#                   Common options:
#                     "gpt-oss-120b"  - large open-weights model (default proxy)
#                     "gpt4o"         - GPT-4o via OpenAI / Azure
#                     "gpt4o-mini"    - GPT-4o Mini, faster and cheaper
#                   Different conditions can use different models if you want
#                   to directly compare model-level effects.
#
#  "context_mode"   This signify if additional context will be sent to the model.
#                   Common options:
#                       "none"        - this runs the model without any supplement information
#                       "file_search" - OpenAI Responses API's file search tool, supply vector store ID
#                       "text"        - contexts stored as a text file, supply file name.
#
#  "context"        This defines the context.
#
#  Tips
#  ----
#  - In experiment mode use short, neutral passcodes ("ALPHA"/"BETA",
#    colours, animals) that give participants no hint of their condition.
#  - System prompts work best when they specify tone, task, and limits all
#    at once.  Vague prompts produce inconsistent behavior across sessions.
#  - Test each condition manually before launching the study.
#
#  Survey mode example (N_CONDITIONS = 1, no "passcode" field needed):
#  ─────────────────────────────────────────────────────────────────────
#  CONDITIONS = [
#      {
#          "name":          "Social-media interview bot",
#          "system_prompt": (
#              "You are a friendly research interviewer studying how people "
#              "use social media in their daily lives.  Ask one open-ended "
#              "question at a time, listen carefully, and ask follow-up "
#              "questions to explore the participant's experience in depth. "
#              "After 5-7 exchanges, thank the participant warmly and let "
#              "them know they can click End this chat."
#          ),
#          "model": "gpt-oss-120b",
#           "mode": "file_search",
#        "context": "vs_35u92340934173412"
#      },
#  ]

CONDITIONS = [

    # ── Condition A ───────────────────────────────────────────────────────────
    {
        "name":          "Condition A - Neutral",
        "passcode":      "ALPHA",      # routes participants to this condition
        "system_prompt": (
            '''
            You are an AI chatbot that helps voters make their minds up about Dutch municipal elections 2026. The context are party platforms for the major parties. 
            Please be helpful. Be as accurate as possible. Do not include any persuasive arguments for the other side. 
            Your goal is not to convince people of one side or the other. Your goal is to help people make up their minds. 
            If the discussion about an issue gets stale, ask follow up questions guiding the user to the next issue. 
            Return the information in an accessible reading level (high school). Always end in complete sentences. 
            Use details from conversation history to seem more personable. 
            This chat is for people on the go. Be a cool person who is affable and nice. 
            Do not seem overly chat-botty. If anyone says anything harmful, please tell them to seek professional help. Respond in 2-3 sentences. 
            Pay attention to the query. Reject irrelevant context. 
            Important: Do not always repeat the same thing. Vary the language a bit every time. 
            Pretend you are a neutral expert having a natural conversation. 
            If the user asks which party is closest to their views, use the chat history to determine which party aligns most closely. 
            Do your best to guess which party they would most likely support.
            '''
        ),
        "model": "gpt-5.4",
        "mode": "file_search",
        "context": "vs_69aa9208eda881918e99651877600b08"
    },

    # ── Condition B ───────────────────────────────────────────────────────────
    {
        "name":          "Condition B - Empathetic",
        "passcode":      "BETA",       # routes participants to this condition
        "system_prompt": (
            '''
            You are an AI chatbot that helps voters make their minds up about Dutch municipal elections 2026. The context are party platforms for the major parties. 
            Please be helpful. Be as accurate as possible. Do not include any persuasive arguments for the other side. 
            Your goal is not to convince people of one side or the other. Your goal is to help people make up their minds. 
            If the discussion about an issue gets stale, ask follow up questions guiding the user to the next issue. 
            Return the information in an accessible reading level (high school). Always end in complete sentences. 
            Use details from conversation history to seem more personable. 
            This chat is for people on the go. Be a cool person who is affable and nice. 
            Do not seem overly chat-botty. If anyone says anything harmful, please tell them to seek professional help. Respond in 2-3 sentences. 
            Pay attention to the query. Reject irrelevant context. 
            Important: Do not always repeat the same thing. Vary the language a bit every time. 
            Pretend you are a neutral expert having a natural conversation. 
            If the user asks which party is closest to their views, use the chat history to determine which party aligns most closely. 
            Do your best to guess which party they would most likely support.
            '''
        ),
        "model": "gpt-5.4",
        "mode": "vanilla", 
        "context": "null"
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
    #     "model": "gpt-5.4",
    #     "mode": "vanilla", 
    #     "context": "null"
    # },

]

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
#           "When you are done, click <strong>End this chat</strong> "
#           "to receive your transcript."
#       )
#
#       WELCOME_MESSAGE = (
#           "This conversation is part of a research study on AI-assisted "
#           "decision-making.  Your responses are confidential and will only "
#           "be used for research purposes.<br><br>"
#           "When finished, click <strong>End this chat</strong> to copy "
#           "your transcript and paste it into the survey."
#       )
WELCOME_MESSAGE = (
    "You are about to have a short conversation with an AI assistant. "
    "When you are finished, click the <strong>End chat</strong> button to receive your transcript, "
    "then paste it back into the survey."
)

# ── Prompt shown on the passcode entry screen (passcode routing only) ──────
#
#   Displayed above the passcode text box when N > 1 and all conditions define
#   a "passcode".  Ignored when N = 1.
PASSCODE_ENTRY_PROMPT = (
    "Please enter the passcode you received in the survey to begin chatting."
)

# ── Configuration reference ───────────────────────────────────────────────────
#
#  Variable               Default           Description
#  ──────────────────────────────────────────────────────────────────────────────
#  API_BASE_URL           (proxy URL)       Base URL for the LLM API endpoint.
#  N_CONDITIONS           2                 1 = survey mode, 2 = A/B test,
#                                           3+ = multi-arm experiment.
#  CONDITIONS             [A, B]            List of condition dicts.  Each has
#                                           "name", optional "passcode",
#                                           "system_prompt", and "model".
#  STUDY_TITLE            "surveychat"      Browser tab title and page heading.
#  WELCOME_MESSAGE        (default string)  Banner shown above the chat.
#                                           Set to "" to hide.
#  PASSCODE_ENTRY_PROMPT  (default string)  Text above the passcode box.
#                                           Only shown in experiment mode.
#  ──────────────────────────────────────────────────────────────────────────────
#
#  Routing behaviour summary
#  ─────────────────────────
#  N = 1                Survey mode.  No gate, no passcode, direct to chat.
#  N > 1, no passcode   Random routing.  Condition drawn at random on load.
#  N > 1, with passcode Passcode routing.  Same passcode → same condition,
#                       stable across page refreshes.
#
# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  END OF RESEARCHER CONFIGURATION - no edits needed below this line        ║
# ╚═════════════════════════════════════════════════════════════════════════════╝