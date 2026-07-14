import pytest
from livekit.agents import NOT_GIVEN, stt
from livekit.agents.types import TimedString

from confidence_voice_agent.confidence import (
    ConfidenceBand,
    ConfidenceExtractor,
    ConfidencePolicy,
)


def speech_event(
    *,
    text: str = "I need help with my order",
    confidence: float | object = 0,
    words: list[TimedString] | None = None,
    event_type: stt.SpeechEventType = stt.SpeechEventType.FINAL_TRANSCRIPT,
) -> stt.SpeechEvent:
    return stt.SpeechEvent(
        type=event_type,
        request_id="request-1",
        alternatives=[
            stt.SpeechData(
                language="en",  # type: ignore[arg-type]
                text=text,
                confidence=confidence,  # type: ignore[arg-type]
                words=words,
            )
        ],
    )


@pytest.mark.parametrize(
    ("confidence", "band"),
    [
        (0.35, ConfidenceBand.LOW),
        (0.6, ConfidenceBand.MEDIUM),
        (0.7, ConfidenceBand.MEDIUM),
        (0.8, ConfidenceBand.MEDIUM),
        (0.92, ConfidenceBand.HIGH),
        (None, ConfidenceBand.UNKNOWN),
    ],
)
def test_confidence_policy_classifies_thresholds(
    confidence: float | None, band: ConfidenceBand
) -> None:
    assert ConfidencePolicy().classify(confidence) is band


def test_extractor_reads_segment_confidence() -> None:
    turn = ConfidenceExtractor(ConfidencePolicy()).extract_turn(speech_event(confidence=0.92))

    assert turn is not None
    assert turn.text == "I need help with my order"
    assert turn.confidence == 0.92
    assert turn.band is ConfidenceBand.HIGH
    assert turn.request_id == "request-1"


def test_extractor_uses_segment_confidence_before_word_confidence() -> None:
    words = [
        TimedString(text="hello", confidence=0.2),
        TimedString(text="there", confidence=0.3),
    ]

    turn = ConfidenceExtractor(ConfidencePolicy()).extract_turn(
        speech_event(confidence=0.91, words=words)
    )

    assert turn is not None
    assert turn.confidence == 0.91
    assert turn.band is ConfidenceBand.HIGH


def test_extractor_falls_back_to_word_average_when_segment_confidence_is_missing() -> None:
    words = [
        TimedString(text="hello", confidence=0.5),
        TimedString(text="there", confidence=0.7),
    ]

    turn = ConfidenceExtractor(ConfidencePolicy()).extract_turn(
        speech_event(confidence=NOT_GIVEN, words=words)
    )

    assert turn is not None
    assert turn.confidence == 0.6
    assert turn.band is ConfidenceBand.MEDIUM
    assert turn.has_bad_provider_confidence is False


def test_extractor_ignores_empty_transcript() -> None:
    assert ConfidenceExtractor(ConfidencePolicy()).extract_turn(speech_event(text="   ")) is None


def test_extractor_ignores_interim_transcript() -> None:
    assert (
        ConfidenceExtractor(ConfidencePolicy()).extract_turn(
            speech_event(event_type=stt.SpeechEventType.INTERIM_TRANSCRIPT)
        )
        is None
    )


@pytest.mark.parametrize("confidence", [-0.1, 1.4])
def test_extractor_marks_invalid_provider_confidence_as_unknown(confidence: float) -> None:
    turn = ConfidenceExtractor(ConfidencePolicy()).extract_turn(speech_event(confidence=confidence))

    assert turn is not None
    assert turn.confidence is None
    assert turn.band is ConfidenceBand.UNKNOWN
    assert turn.has_bad_provider_confidence is True
