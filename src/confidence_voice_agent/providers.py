from livekit.agents import AgentSession
from livekit.plugins import deepgram, elevenlabs, openai, silero

from confidence_voice_agent.config import Settings


def create_agent_session(settings: Settings) -> AgentSession:
    return AgentSession(
        stt=create_stt(settings),
        llm=create_llm(settings),
        tts=create_tts(settings),
        vad=silero.VAD.load(),
    )


def create_stt(settings: Settings) -> deepgram.STT:
    return deepgram.STT(
        model=settings.deepgram_model,
        language=settings.deepgram_language,
        interim_results=True,
        api_key=settings.deepgram_api_key.get_secret_value(),
    )


def create_llm(settings: Settings) -> openai.LLM:
    return openai.LLM(
        model=settings.openai_model,
        api_key=settings.openai_api_key.get_secret_value(),
        temperature=0.2,
    )


def create_tts(settings: Settings) -> elevenlabs.TTS:
    return elevenlabs.TTS(
        voice_id=settings.elevenlabs_voice_id,
        model=settings.elevenlabs_model,
        api_key=settings.elevenlabs_api_key.get_secret_value(),
    )
