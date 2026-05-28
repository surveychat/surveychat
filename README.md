# 💬 surveychat

[![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)
![GitHub License](https://img.shields.io/github/license/surveychat/surveychat)

surveychat lets you embed an AI chatbot directly inside your Qualtrics survey. Participants chat, click **End chat**, and paste a transcript back into your survey. No separate website or app to open. You set up everything by editing a single text file; no programming experience needed.

**Demo:** [surveychat.invisible.info](https://surveychat.invisible.info) - use code `ALPHA` (neutral chatbot) or `BETA` (empathetic chatbot).

| What you need | Which mode to use |
|---|---|
| Everyone gets the same chatbot (e.g. an interview or debriefing task) | **Survey mode**: set `N_CONDITIONS = 1` |
| Different participants get different chatbot versions (e.g. a neutral vs. empathetic bot) | **Experiment mode**: set `N_CONDITIONS = 2` or more |

| | | |
|---|---|---|
| ![Passcode entry](paper/surveychat-interface-1.png) | ![Chat interface](paper/surveychat-interface-2.png) | ![Transcript export](paper/surveychat-interface-3.png) |

---

## Setup

**You will need:**
- Python 3.10 or newer (download from [python.org](https://www.python.org/downloads/) if you don't have it)
- An API key from your AI provider (e.g. OpenAI, or your institution's LLM service)

**Steps:**

```bash
git clone https://github.com/YOUR_USERNAME/surveychat.git
cd surveychat
pip install -r requirements.txt
cp .env.example .env          # then open .env and paste in your API key
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501). You should see the chatbot.

> This address only works on your own computer. To share the chatbot with participants, see [Deployment](#deployment).

---

## Configuration

Open `app.py` in any text editor (Notepad, TextEdit, VS Code, etc.) and find the section that starts with:

```
# ╔══════════ RESEARCHER CONFIGURATION ═══════════╗
```

Everything between that line and the matching closing line is yours to edit. Nothing else in the file needs to be touched.

---

### 1. Choose your mode

Set `N_CONDITIONS` to the number of different chatbot versions you need:

```python
N_CONDITIONS = 1   # everyone gets the same chatbot
N_CONDITIONS = 2   # two versions (A/B)
N_CONDITIONS = 3   # three versions, and so on
```

---

### 2. Write your chatbot instructions

The `CONDITIONS` list defines each chatbot version. Each version is a set of settings inside curly braces `{ }`.

```python
CONDITIONS = [
    {
        "name":          "Neutral",       # your internal label; participants never see this
        "passcode":      "ALPHA",         # required in experiment mode; optional in survey mode
        "system_prompt": "...",           # the hidden instructions that tell the chatbot how to behave
        "model":         "gpt-4o",        # which AI model to use
    },
]
```

| Setting | When to include | What it does |
|---|---|---|
| `name` | Always | A label for your own records. Participants never see it. |
| `passcode` | Required in experiment mode; optional in survey mode | The code a participant enters to start. In experiment mode, each condition gets a different code, which is how you control which version each participant sees. In survey mode, you can include a shared code to restrict access, or leave it out entirely. |
| `system_prompt` | Always | The hidden instructions that define how the chatbot behaves: its tone, task, and boundaries. Participants never see this text. |
| `model` | Always | Which AI model to use, e.g. `"gpt-4o"` or `"gpt-4o-mini"`. |
| `initial_message` | Optional | A scripted first message the chatbot sends before the participant types anything. Useful if you want the bot to open with a fixed question. |
| `temperature` | Optional | How varied the chatbot's replies are. Overrides the global setting for this condition only. |
| `max_tokens` | Optional | Maximum length of each chatbot reply. Overrides the global setting for this condition only. |

---

### 3. Other settings

| Setting | Default | What it does |
|---|---|---|
| `API_BASE_URL` | OpenAI | The web address of the AI service you are using. Change this if you use a different provider (see table below). |
| `TEMPERATURE` | Model default | How varied the chatbot's replies are. `0` = very consistent, `1` = more natural variation. Leave as `None` to use the AI provider's default. |
| `MAX_TOKENS` | No limit | Maximum length of each chatbot reply. Set a number (e.g. `512`) to keep replies concise and control costs. |
| `MAX_EXCHANGES` | No limit | Maximum number of messages a participant can send. When this limit is reached, the chat ends automatically and the transcript appears. Set e.g. `6` for a fixed-length interview. |
| `STUDY_TITLE` | `"surveychat"` | The name shown in the browser tab and at the top of the page. |
| `WELCOME_MESSAGE` | *(none)* | A short instruction or welcome note shown at the top of the chat. Leave as `""` to show nothing (useful when Qualtrics already shows instructions above the chatbot). |
| `PASSCODE_ENTRY_PROMPT` | see app.py | The instruction text shown above the passcode box. |

**Which AI service address (`API_BASE_URL`) should I use?**

| Provider | Address |
|---|---|
| OpenAI | `"https://api.openai.com/v1"` |
| University of Amsterdam | `"https://llmproxy.uva.nl/v1"` |
| OpenRouter (access many models with one key) | `"https://openrouter.ai/api/v1"` |
| Mistral AI | `"https://api.mistral.ai/v1"` |
| HuggingFace | `"https://api-inference.huggingface.co/v1"` |
| Local model (LM Studio, Ollama, vLLM) | `"http://localhost:1234/v1"` |

---

## Transcript format

When a participant clicks **End chat**, they receive a transcript they can copy and paste into your survey. The transcript is structured text (JSON) that looks like this:

```json
{
  "messages": [
    {"role": "participant", "content": "Hello!",    "timestamp": "2026-03-06T14:22:01+00:00"},
    {"role": "assistant",   "content": "Hi there!", "timestamp": "2026-03-06T14:22:03+00:00"}
  ]
}
```

Each message records who said it (`participant` or `assistant`), what was said, and when. You can import this into a spreadsheet using Python or R:

Parse in **Python:**
```python
import json, pandas as pd
df = pd.DataFrame(json.loads(transcript_string)["messages"])
```

Parse in **R:**
```r
df <- as.data.frame(jsonlite::fromJSON(transcript_string)$messages)
```

---

## Deployment

To let participants access the chatbot, you need to host it somewhere publicly accessible. Choose the option that fits your situation:

| Option | Best for | How |
|---|---|---|
| **Streamlit Cloud** (recommended) | Most researchers. Free, no server needed. | Push your repo to GitHub → [share.streamlit.io](https://share.streamlit.io) → add `OPENAI_API_KEY` under Advanced settings → Secrets → Deploy |
| **Local** | Testing on your own computer only | `streamlit run app.py` (not accessible to participants) |
| **Docker** | Researchers comfortable with the command line | `docker compose up --build` (reads your `.env` automatically) |
| **Cloud server** | Large studies or custom infrastructure | `streamlit run app.py --server.port 80 --server.headless true` |

> **Qualtrics iFrame note:** Embedding the chatbot inside a Qualtrics page requires your app to be served over HTTPS (i.e. a `https://` address). Streamlit Cloud and most cloud providers handle this automatically.

---

## Qualtrics integration

**Survey mode**

In your Qualtrics survey, add a **Text / Graphic** block and paste this HTML (replacing the URL with your own). Then add a **Text Entry** question immediately after it.

```html
<p>Chat with the assistant below. When you're done, click <strong>End chat</strong> to get a button to copy your transcript.</p>
<div style="border: 1px solid #d4d4d4; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.06); margin: 16px 0;">
  <iframe src="https://your.app.url/" width="100%" height="700" style="display: block; border: none;"></iframe>
</div>
```

**Experiment mode**

In **Survey Flow → Randomizer**, create one branch per condition. In each branch:

1. Set an embedded data field, e.g. `passcode = ALPHA` for one branch, `passcode = BETA` for another.
2. Add a **Text / Graphic** block with this HTML (replacing the URL):

```html
<p style="font-size:16px;">Your passcode is: <strong>${e://Field/passcode}</strong></p>
<br>
<p>Enter your assigned passcode in the chatbot below to start. When you want to finish speaking, click <strong>End chat</strong>, and you will be shown a button to copy your conversation transcript.</p>
<div style="border: 1px solid #d4d4d4; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.06); margin: 16px 0;">
  <iframe src="https://your.app.url/" width="100%" height="700" style="display: block; border: none;"></iframe>
</div>
```

In both modes, add a **Text Entry** question after the chatbot block: *"After you have finished chatting, paste your copied transcript here."*

---

## Troubleshooting

| What you see | What to do |
|---|---|
| "Code not recognised" | The passcode entered doesn't match what's set in `app.py`. Double-check for typos. Passcodes are not case-sensitive. |
| "OPENAI_API_KEY not found" | Open your `.env` file and make sure it contains `OPENAI_API_KEY=sk-...` with no spaces around the `=` sign. |
| Chat returns an error message | Check that `API_BASE_URL` in `app.py` is correct for your provider, and that your API key is valid. |
| The page doesn't load | Go to http://localhost:8501 directly. If it says "connection refused", the app may have crashed. Check the terminal window for error messages. |
| Port 8501 already in use | Run `pkill -f "streamlit run"` to stop any existing instance, or start on a different port: `streamlit run app.py --server.port 8502` |
| You edited `app.py` but nothing changed | Streamlit usually reloads automatically. If it doesn't, press **R** in the terminal, or stop and restart with `streamlit run app.py`. |

