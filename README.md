# Knowledge Nexus v3.0

Multi-agent supply chain system built for CMU's Agentic AI course (Spring 2026).

## Features

- **Chat Interface**: Conversational interaction with a multi-agent pipeline (Planner, Executor, Critic)
- **Email Inbox**: Fetch and process real Gmail messages through the agent
- **Evaluation Dashboard**: Automated 5-case test harness with groundedness, adherence, and task completion metrics
- **Agent Logs**: Real-time log viewer for pipeline activity
- **Adaptive Control Loop**: OODA-based observe-plan-execute-evaluate cycle with automatic retry/replan

## Architecture

```
User Message
    |
[Retrieval Agent] -> VectorDB search
    |
[Planner Agent]   -> Intent classification + plan
    |
[Executor Agent]  -> Tool use (PO gen, email, search)
    |
[Critic Agent]    -> Groundedness + adherence scoring
    |
[Orchestrator]    -> Retry / Replan / Approve decision
```

## Running Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create a .env file with your API keys
cp secrets.toml.example .env
# Edit .env and fill in your keys

# 3. Run the app
streamlit run app.py
```

## Deploying on Streamlit Cloud

1. Fork or push this repo to your GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo and select `app.py` as the main file
4. Add your secrets in **Settings > Secrets** using the format from `secrets.toml.example`
5. Deploy

## Required Secrets

| Key | Description |
|-----|-------------|
| `GEMINI_API_KEY` | Google Gemini API key |
| `EMAIL_ADDRESS` | Gmail address for inbox features |
| `EMAIL_PASSWORD` | Gmail App Password (not account password) |
| `SLACK_BOT_TOKEN` | Slack bot token (optional) |
| `SLACK_APP_TOKEN` | Slack app token (optional) |
