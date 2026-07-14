import logging
from collections.abc import AsyncIterable
from typing import Any

from livekit.agents import Agent, ChatContext, ModelSettings, llm, stt

from confidence_voice_agent.confidence import (
    ConfidenceExtractor,
    ConfidencePolicy,
    TranscriptionTurn,
)
from confidence_voice_agent.config import Settings
from confidence_voice_agent.context import ConfidenceContextBuilder

logger = logging.getLogger(__name__)


class ConfidenceAwareAgent(Agent):
    def __init__(self, settings: Settings) -> None:
        self._extractor = ConfidenceExtractor(
            ConfidencePolicy(
                low_threshold=settings.low_confidence_threshold,
                high_threshold=settings.high_confidence_threshold,
            )
        )
        self._context_builder = ConfidenceContextBuilder()
        self._latest_turn: TranscriptionTurn | None = None
        super().__init__(
            instructions=(
                "You are a concise support voice agent. Use the speech recognition metadata "
                "provided in the latest system message. If confidence is low or unknown, ask "
                "the user to repeat or clarify instead of answering a guessed request."
            )
        )

    def record_stt_event(self, event: stt.SpeechEvent) -> TranscriptionTurn | None:
        if event.type is stt.SpeechEventType.START_OF_SPEECH:
            self._latest_turn = None
            return None

        turn = self._extractor.extract_turn(event)
        if turn is None:
            return None

        self._latest_turn = turn
        logger.info(
            "recognized user speech",
            extra={
                "request_id": turn.request_id,
                "confidence": turn.confidence,
                "confidence_band": turn.band.value,
                "has_bad_provider_confidence": turn.has_bad_provider_confidence,
            },
        )
        return turn

    def build_llm_chat_context(self, chat_ctx: ChatContext) -> ChatContext:
        if self._latest_turn is None:
            return chat_ctx

        enriched_context = chat_ctx.copy()
        enriched_context.add_message(
            role="system",
            content=self._context_builder.build(self._latest_turn),
        )
        return enriched_context

    async def stt_node(
        self,
        audio: AsyncIterable[Any],
        model_settings: ModelSettings,
    ) -> AsyncIterable[stt.SpeechEvent | str]:
        async for event in Agent.default.stt_node(self, audio, model_settings):
            if isinstance(event, stt.SpeechEvent):
                self.record_stt_event(event)
            yield event

    async def llm_node(
        self,
        chat_ctx: ChatContext,
        tools: list[llm.Tool],
        model_settings: ModelSettings,
    ) -> Any:
        return Agent.default.llm_node(
            self,
            self.build_llm_chat_context(chat_ctx),
            tools,
            model_settings,
        )
