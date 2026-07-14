import pytest
from livekit.agents import ChatContext, stt

from confidence_voice_agent.agent import ConfidenceAwareAgent
from confidence_voice_agent.confidence import ConfidenceBand
from tests.test_confidence import speech_event
from tests.test_config import valid_settings


def test_agent_records_final_stt_confidence() -> None:
    agent = ConfidenceAwareAgent(valid_settings())

    turn = agent.record_stt_event(speech_event(confidence=0.35))

    assert turn is not None
    assert turn.confidence == 0.35
    assert turn.band is ConfidenceBand.LOW


def test_agent_ignores_interim_stt_for_llm_context() -> None:
    agent = ConfidenceAwareAgent(valid_settings())
    agent.record_stt_event(speech_event(confidence=0.92))

    before = agent.build_llm_chat_context(ChatContext())
    agent.record_stt_event(
        speech_event(confidence=0.2, event_type=stt.SpeechEventType.INTERIM_TRANSCRIPT)
    )
    after = agent.build_llm_chat_context(ChatContext())

    assert "STT confidence: 0.92" in str(before.to_dict())
    assert "STT confidence: 0.92" in str(after.to_dict())
    assert "STT confidence: 0.20" not in str(after.to_dict())


def test_agent_clears_previous_confidence_when_new_speech_starts() -> None:
    agent = ConfidenceAwareAgent(valid_settings())
    agent.record_stt_event(speech_event(confidence=0.92))

    before = agent.build_llm_chat_context(ChatContext())
    agent.record_stt_event(
        speech_event(confidence=0.2, event_type=stt.SpeechEventType.START_OF_SPEECH)
    )
    after = agent.build_llm_chat_context(ChatContext())

    assert "STT confidence: 0.92" in str(before.to_dict())
    assert "STT confidence" not in str(after.to_dict())


def test_agent_adds_confidence_metadata_to_llm_context() -> None:
    agent = ConfidenceAwareAgent(valid_settings())
    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="I need help with my order")
    agent.record_stt_event(speech_event(confidence=0.35))

    enriched = agent.build_llm_chat_context(chat_ctx)
    context_text = str(enriched.to_dict())

    assert "I need help with my order" in context_text
    assert "STT confidence: 0.35" in context_text
    assert "Confidence band: low" in context_text
    assert "Ask the user to repeat or clarify" in context_text
    assert len(enriched.items) == len(chat_ctx.items) + 1


def test_agent_does_not_mutate_original_chat_context() -> None:
    agent = ConfidenceAwareAgent(valid_settings())
    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="Hello")
    agent.record_stt_event(speech_event(confidence=0.92))

    agent.build_llm_chat_context(chat_ctx)

    assert len(chat_ctx.items) == 1


@pytest.mark.asyncio
async def test_agent_passes_confidence_context_to_llm_and_clears_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from typing import Any

    from livekit.agents import ModelSettings

    agent = ConfidenceAwareAgent(valid_settings())
    agent.record_stt_event(speech_event(confidence=0.92))
    captured: dict[str, ChatContext] = {}

    async def response_stream() -> Any:
        yield "response"

    def mock_llm_node(_agent: Any, chat_ctx: ChatContext, *_: Any) -> Any:
        captured["chat_ctx"] = chat_ctx
        return response_stream()

    monkeypatch.setattr("livekit.agents.Agent.default.llm_node", mock_llm_node)

    assert agent._latest_turn is not None
    response = agent.llm_node(ChatContext(), [], ModelSettings())
    assert agent._latest_turn is None
    assert "STT confidence: 0.92" in str(captured["chat_ctx"].to_dict())
    assert [chunk async for chunk in response] == ["response"]
