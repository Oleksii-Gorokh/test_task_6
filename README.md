# Confidence Voice Agent

Python voice agent built with LiveKit Agents. It uses Deepgram for STT, OpenAI for LLM, ElevenLabs for TTS, and passes STT confidence metadata into the LLM context before every response.

## What It Does

Pipeline:

```text
User speech -> Deepgram STT -> confidence extraction -> LLM context -> OpenAI LLM -> ElevenLabs TTS -> user
```

Behavior:

- High confidence, `> 0.8`: answer normally.
- Low confidence, `< 0.6`: ask the user to repeat or clarify.
- Medium confidence, `0.6..0.8`: answer cautiously and clarify when needed.
- Missing or invalid confidence: ask the user to repeat.

## Requirements

- Python 3.12
- uv
- Docker, only if you want to run LiveKit locally
- API keys for Deepgram, OpenAI, and ElevenLabs

## Setup

Install dependencies:

```bash
uv sync
```

Create local environment file:

```bash
cp .env.example .env
```

Fill in:

```dotenv
DEEPGRAM_API_KEY=...
OPENAI_API_KEY=...
ELEVENLABS_API_KEY=...
```

For local LiveKit, `.env.example` already uses:

```dotenv
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret
```

If you use LiveKit Cloud, replace those values with project credentials from the LiveKit Cloud dashboard.

## API Keys

Deepgram:

1. Create an account at https://deepgram.com/.
2. Open the Deepgram console.
3. Create an API key.
4. Set `DEEPGRAM_API_KEY`.

OpenAI:

1. Create an API key in the OpenAI dashboard.
2. Set `OPENAI_API_KEY`.
3. Optional: change `OPENAI_MODEL`.

ElevenLabs:

1. Create an account at https://elevenlabs.io/.
2. Create an API key.
3. Set `ELEVENLABS_API_KEY`.
4. Optional: change `ELEVENLABS_VOICE_ID`.

LiveKit:

- Cloud: create a free LiveKit Cloud project and copy `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`.
- Local: run the dev server shown below.

## Run LiveKit Locally

```bash
docker compose up livekit
```

LiveKit dev mode uses:

```dotenv
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret
```

## Run The Agent

Development mode:

```bash
uv run python -m confidence_voice_agent.app dev
```

Production-style mode:

```bash
uv run python -m confidence_voice_agent.app start
```

Console mode for quick local testing:

```bash
uv run python -m confidence_voice_agent.app console
```

## Verify Manually

1. Start LiveKit locally or use LiveKit Cloud.
2. Start the agent in `dev` mode.
3. Join the same room from LiveKit Agents Playground or another LiveKit client.
4. Speak clearly and verify a normal answer.
5. Speak quietly or with background noise and verify the agent asks you to repeat.

High-confidence example:

```text
User said: "I need help with my order"
STT confidence: 0.92
Agent: "Sure. What seems to be the issue with your order?"
```

Low-confidence example:

```text
User said: "I need help with my order"
STT confidence: 0.35
Agent: "Sorry, I may have misunderstood you. Could you please repeat that?"
```

## How Confidence Reaches The LLM

`ConfidenceAwareAgent.stt_node` keeps the default LiveKit STT pipeline, but inspects final and preflight `SpeechEvent` values from Deepgram. `ConfidenceExtractor` converts the event into a typed `TranscriptionTurn` with transcript, confidence, request id, language, and confidence band.

Before the LLM runs, `ConfidenceAwareAgent.llm_node` copies the current `ChatContext` and appends a system message like:

```text
Speech recognition metadata for the latest user turn:
User said: "I need help with my order"
STT confidence: 0.35
Confidence band: low
Instruction: Ask the user to repeat or clarify instead of answering the presumed request.
```

The original chat context is not mutated.

## Tests And Checks

Run tests:

```bash
uv run pytest
```

Run lint:

```bash
uv run ruff check .
```

Run formatting check:

```bash
uv run ruff format --check .
```

Run type checks:

```bash
uv run mypy src
```

The unit tests do not call Deepgram, OpenAI, ElevenLabs, or LiveKit Cloud. They cover config validation, confidence thresholds, bad provider confidence, prompt context propagation, and agent-level behavior around final vs interim transcripts.

## Troubleshooting

Missing API key:

- Check `.env`.
- Make sure the variable name matches `.env.example`.

No LiveKit connection:

- For local mode, check `docker compose up livekit`.
- For Cloud mode, check `LIVEKIT_URL`, `LIVEKIT_API_KEY`, and `LIVEKIT_API_SECRET`.

No audio response:

- Check ElevenLabs key and voice id.
- Check that the selected room has an agent participant.

Low-confidence behavior does not trigger:

- Verify logs contain the confidence value and `confidence_band`.
- Temporarily raise `LOW_CONFIDENCE_THRESHOLD` in `.env` to make the behavior easier to reproduce.
