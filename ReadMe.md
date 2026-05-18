
# LiveKit Meeting Agent

A LiveKit-powered meeting agent that uses LangGraph to drive conversation flow with avatar support.

## Prerequisites

- Python 3.11–3.13
- [UV](https://docs.astral.sh/uv/) package manager
- [LiveKit CLI](https://docs.livekit.io/home/cli/cli-setup/) (`lk`)

## Setup

### 1. Authenticate with LiveKit Cloud

```powershell
lk cloud auth
```

### 2. Install dependencies

```powershell
uv sync
```

### 3. Configure environment variables

Generate the `.env.local` file with LiveKit API keys:

```powershell
lk app env -w
```

Then add API keys for the other services (Deepgram, Cartesia, OpenAI, Anam, etc.) to `.env.local`.

### 4. Download model files

Required for Silero VAD, turn detector, and noise cancellation plugins:

```powershell
uv run agent.py download-files
```

## Running the Agent

### Run in console mode (for testing)

```powershell
uv run agent.py console
```

### Run as a LiveKit sandbox (production)

```powershell
uv run agent.py dev

Now browse livekit sandbox url - 
https://agents-playground.livekit.io/
```



## Project Setup from Scratch

If you need to recreate the project from scratch:

```powershell
uv init livekit-meeting-agent --bare
cd livekit-meeting-agent

uv add "livekit-agents[deepgram,openai,cartesia,silero,turn-detector]~=1.4" "livekit-plugins-noise-cancellation~=0.2" "python-dotenv"
uv add python-dotenv langgraph langchain-core langchain-openai langchain-community langchain-chroma chromadb
uv add "livekit-plugins-langchain~=1.1"
uv add pypdf
```