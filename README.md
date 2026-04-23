# 💬 surveychat

[![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)
![GitHub License](https://img.shields.io/github/license/surveychat/surveychat)

`surveychat` is an open-source web application that enables researchers to administer surveys and conduct randomized experiments involving large language model (LLM)-based conversational agents, without the need to develop custom web application code. The system supports two primary operational modes: (i) **survey mode**, in which all participants interact with an identical chatbot configuration, and (ii) **experiment mode**, in which participants are randomly assigned to one of multiple chatbot conditions, each defined by a researcher-specified persona and language model. Upon completion of the interaction, participants receive an anonymized JSON transcript that contains only the role, content, and timestamp of each message. This transcript can be copied back into the parent survey platform (such as Qualtrics), within which the chatbot interface itself can also be directly embedded. The frontend of `surveychat` is implemented using Streamlit, and the entire application is configured via a single Python file. The system does not persist conversation data on its server and is compatible with any chat-completions-compatible API endpoint - including locally hosted models - thereby allowing researchers to retain full control over model selection, API usage, data jurisdiction, and adherence to ethical and regulatory requirements.

> Simple setup instructions [here](https://surveychat.github.io/).

> **Demo:** Try an experimental setup [here](https://surveychat.invisible.info) — use code ALPHA for a neutral chatbot, and BETA for an empathetic chatbot. This demo uses the open-source model `gpt-oss-120b`.

surveychat works in two modes:

- **Survey mode** - every participant talks to the same chatbot. Good for open-ended interviews, pilot testing, or replacing a plain text-entry question with a richer conversation.
- **Experiment mode** - participants are automatically assigned to different chatbot versions (e.g. neutral vs. empathetic vs. socratic). Use this when you want to *compare* how different chatbot styles/models/prompts affect responses.

In both modes the participant chats, clicks **End chat**, and copies a text transcript back into your survey tool (e.g. Qualtrics). No coding experience beyond editing a text file is required - no server to manage, no database to set up.



Entering a passcode (experiment mode only):

![Passcode entry](paper/surveychat-interface-1.png)

Chatting with the bot:

![Chat interface](paper/surveychat-interface-2.png)

Copying the transcript when done:

![Transcript export](paper/surveychat-interface-3.png)

---

## Before you start

You will need:

- **Python 3.10 or newer.** Check by running `python3 --version` in your terminal. If you don't have Python, download it from [python.org](https://www.python.org/downloads/).
- **An API key and endpoint.** surveychat uses a large language model (LLM) to power the chatbot. You will need an API key from any compatible provider (OpenAI, Azure, OpenRouter, or a local proxy). The key is stored in the `OPENAI_API_KEY` environment variable by convention. You will also set `API_BASE_URL` in `app.py` to point to your provider's chat-completions endpoint (see [Step 3 - Optional settings](#step-3---optional-settings)).
- **A terminal.** On macOS/Linux open **Terminal**. On Windows open **Command Prompt** or **PowerShell**.

---

## Quick start

**Step 1 - Fork the repo**

Click **Fork** at the top right of this GitHub page. This creates your own copy of surveychat under your GitHub account, which you can edit and deploy freely.

**Step 2 - Clone your fork to your computer**

Replace `YOUR_USERNAME` with your GitHub username:

```bash
git clone https://github.com/YOUR_USERNAME/surveychat.git
cd surveychat
pip install -r requirements.txt
cp .env.example .env
```

> **Don't have git?** Download it from [git-scm.com](https://git-scm.com/downloads) (free). On macOS it may already be installed - check with `git --version` in your terminal.

Now open the file called `.env` in any text editor and replace `your-key-here` with your actual API key:

```
OPENAI_API_KEY=sk-...
```

Save the file, then start the app by running this command in the terminal:

```bash
streamlit run app.py
```

Your browser will open automatically at http://localhost:8501. You should see the chatbot interface.

> **Note:** This URL only works on your own computer. To let participants access the chatbot, you will need to deploy it — see [Deployment](#deployment).

---

## Configuration

All settings live at the top of the file `app.py` inside a clearly marked section. You do not need to touch any other part of the file.

Open `app.py` in a text editor and find the block that begins:

```
# ╔══════════ RESEARCHER CONFIGURATION ═══════════╗
```

Everything you need to change is between that line and the matching closing line.

---

### Step 1 - Choose your mode

Set `N_CONDITIONS` to the number of different chatbot versions you need:

```python
N_CONDITIONS = 1   # survey mode  - one chatbot for everyone
N_CONDITIONS = 2   # experiment mode - A/B test (two versions)
N_CONDITIONS = 3   # experiment mode - three versions, and so on
```

---

### Step 2 - Write your chatbot instructions

The `CONDITIONS` list defines each chatbot version. Each version is a block of settings inside curly braces `{ }`.

**Survey mode example** (`N_CONDITIONS = 1`):

```python
CONDITIONS = [
    {
        "name":          "Interview bot",
        "system_prompt": "You are a friendly research interviewer. Ask one open-ended question at a time about the participant's social media habits. After 5–6 exchanges, thank them and let them know they can click End this chat.",
        "model":         "gpt-oss-120b",
    },
]
```

**Experiment mode example** (`N_CONDITIONS = 2`):

```python
CONDITIONS = [
    {
        "name":          "Condition A - Neutral",
        "passcode":      "ALPHA",
        "system_prompt": "You are a neutral research assistant. Answer questions clearly and factually without expressing opinions.",
        "model":         "gpt-oss-120b",
    },
    {
        "name":          "Condition B - Empathetic",
        "passcode":      "BETA",
        "system_prompt": "You are a warm, empathetic research assistant. Acknowledge the participant's feelings before responding.",
        "model":         "gpt-oss-120b",
    },
]
```

**What each field means:**

| Field | Required? | What it does |
|---|---|---|
| `"name"` | Always | A label for your own reference. Participants never see this. |
| `"passcode"` | Experiment mode only | The code a participant enters to reach this condition. Case-insensitive (`"alpha"` and `"ALPHA"` are the same). Leave this out when `N_CONDITIONS = 1`. |
| `"system_prompt"` | Always | The hidden instruction that tells the chatbot how to behave. Participants never see this text. |
| `"model"` | Always | Which AI model to use. Ask your lab coordinator which model name to use. |

---

### Step 3 - Optional settings

```python
API_BASE_URL = "https://api.openai.com/v1"
# The base URL for your LLM provider's chat-completions endpoint.
# Common values:
#   OpenAI:       "https://api.openai.com/v1"
#   OpenRouter:   "https://openrouter.ai/api/v1"
#   HuggingFace:  "https://api-inference.huggingface.co/v1"
#                 (set OPENAI_API_KEY to your HF token; set "model" to
#                  the HF model ID, e.g. "meta-llama/Llama-3.3-70B-Instruct")
#   Local model:  "http://localhost:1234/v1"
# Any chat-completions-compatible endpoint will work.

STUDY_TITLE = "surveychat"
# The name shown in the browser tab and at the top of the page.
# Change this to your study name, e.g. "Climate Attitudes Study".

WELCOME_MESSAGE = (
    "You are about to have a short conversation with an AI assistant. "
    "When you are finished, click the <strong>End chat</strong> button to receive your transcript, "
    "then paste it back into the survey."
)
# A message shown to participants before they start chatting.
# Leave as "" for no message.

PASSCODE_ENTRY_PROMPT = "Please enter the passcode you received in the survey to begin chatting."
# The instruction shown above the passcode box (experiment mode only).
```

---

## Transcript format

```json
{
  "messages": [
    {
      "role": "participant", 
      "content": "Hello!", 
      "timestamp": "2026-03-06T14:22:01+00:00"
    },
    {
      "role": "assistant",
      "content": "Hi there! How can I help you today?", 
      "timestamp": "2026-03-06T14:22:03+00:00"
    }
  ]
}
```

Each message has:
- `role` - either `"participant"` (what the person typed) or `"assistant"` (the chatbot's reply)
- `content` - the full text of the message
- `timestamp` - when the message was sent (UTC time)


**Note:** The JSON transcript can be parsed in Python or R using standard libraries. Each message becomes a row in a dataframe, with columns for `role`, `content`, and `timestamp`.

Parse in Python:
```python
import json, pandas as pd
data = json.loads(transcript_string)   # transcript_string is the text they pasted
df   = pd.DataFrame(data["messages"]) # one row per message
```

Parse in R:
```r
library(jsonlite)
data <- fromJSON(transcript_string)
df   <- as.data.frame(data$messages)
```

---

## Deployment

### Option 1 - Run locally

Good for small surveys on your own computer or trying things out before you deploy online.

```bash
streamlit run app.py
```

Streamlit will show two URLs:
- `http://localhost:8501` — works only on your computer.
- `http://192.168.x.x:8501` — works on other devices on the same local network (e.g. your home or office Wi-Fi), but only if your firewall allows connections on port 8501.

Neither URL is accessible from the internet, so this option is not suitable for sharing with participants remotely. For a publicly accessible URL, use one of the options below.

### Option 2 - Streamlit Community Cloud (free for public repos)

This gives you a permanent public URL with no server to manage.

1. Push your repo to GitHub (your `.env` file is excluded automatically)
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your GitHub repo
3. Under **Advanced settings → Secrets**, add: `OPENAI_API_KEY = "sk-..."`
4. Click **Deploy** - you get a public URL to share with participants

### Option 3 - Docker (for researchers comfortable with the command line)
The repo ships with a `Dockerfile` and `docker-compose.yml`.

**Quick start (recommended):**
```bash
docker compose up --build
```
Open [http://localhost:8501](http://localhost:8501). The container reads your `.env` file automatically.

**Without Compose:**
```bash
docker build -t surveychat .
docker run --rm -p 8501:8501 --env-file .env surveychat
```

**Production tips:**
- Remove the `volumes:` bind-mount in `docker-compose.yml` so the image is fully self-contained.
- Serve HTTPS via a reverse proxy (Caddy, nginx) in front of port 8501 - required for Qualtrics iFrame embeds.
- On a cloud VM, add `--server.port 80` to the `ENTRYPOINT` in the `Dockerfile` if you expose port 80 directly.

### Option 4 - Cloud VM (advanced)

For a permanent public deployment on a cloud server (e.g. Azure, AWS, DigitalOcean), set up a VM with Python and Docker, clone your repo, and run the app with:

```bash
pip install -r requirements.txt
streamlit run app.py --server.port 80 --server.headless true
```

---

## Integrating with Qualtrics

surveychat works well when embedded directly inside your Qualtrics survey using an **iFrame** - a standard way to show one website inside another. Participants stay on the Qualtrics page the whole time: the chatbot loads right there, they chat, and then paste their transcript into the next question without ever opening a separate tab. In experiment mode, Qualtrics shows each participant their passcode just above the iFrame so they can enter it to start.

To embed, add a **Text / Graphic** block in Qualtrics and paste this HTML, replacing the URL with your own:

```html
<iframe
  src="https://your.apps.url/"
  width="100%"
  height="600"
  frameborder="0"
  allow="clipboard-write"
></iframe>
```

The `allow="clipboard-write"` attribute lets the built-in copy button work inside the iFrame.

**Survey mode (N = 1):**
1. Add a **Text / Graphic** block containing the iFrame above (or just a plain link if you prefer)
2. After that block, add a **Text Entry** question: *"Paste your chat transcript here"*
3. Export responses and parse the JSON from that column

**Experiment mode (N > 1):**
1. Use Qualtrics **Survey Flow - Randomizer** to split participants into arms
2. In each arm's branch, display the matching passcode (e.g. *"Your passcode is: ALPHA"*) and embed the iFrame below it
3. After the iFrame block, add a **Text Entry** question: *"Paste your chat transcript here"*
4. Export responses - you know which condition each participant was in from which Qualtrics branch they went through

---

## Troubleshooting

**"Code not recognised"**
The passcode the participant typed does not match any `"passcode"` value in `CONDITIONS`. Check for typos in `app.py`. Passcode matching is case-insensitive, so `"alpha"` and `"ALPHA"` both work.

**"OPENAI_API_KEY not found"**
Make sure the `.env` file exists in the project folder and contains a line exactly like this (no spaces around `=`):
```
OPENAI_API_KEY=sk-your-key-here
```

**Chat returns an error message**
Verify that `API_BASE_URL` in `app.py` is set to the correct URL for your API provider, and that your key is valid and has not expired.

**The app does not open in the browser**
Manually open http://localhost:8501. If you see "connection refused", the app may have crashed - check the terminal for error messages.

**"Port 8501 already in use"**
Another instance of the app is already running. Stop it with:
```bash
pkill -f "streamlit run"
```
Or start the app on a different port:
```bash
streamlit run app.py --server.port 8502
```

**I edited `app.py` but nothing changed**
Streamlit usually reloads automatically when you save the file. If it does not, press **R** in the terminal where the app is running, or stop and restart with `streamlit run app.py`.
