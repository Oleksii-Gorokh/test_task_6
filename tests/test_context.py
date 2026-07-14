from confidence_voice_agent.confidence import ConfidenceBand, TranscriptionTurn
from confidence_voice_agent.context import ConfidenceContextBuilder


def turn(band: ConfidenceBand, confidence: float | None) -> TranscriptionTurn:
    return TranscriptionTurn(
        text='I need help with my "order"\nplease',
        confidence=confidence,
        band=band,
    )


def test_context_contains_transcript_confidence_and_band() -> None:
    context = ConfidenceContextBuilder().build(turn(ConfidenceBand.LOW, 0.35))

    assert 'User said: "I need help with my \\"order\\" please"' in context
    assert "STT confidence: 0.35" in context
    assert "Confidence band: low" in context


def test_low_confidence_context_instructs_clarification() -> None:
    context = ConfidenceContextBuilder().build(turn(ConfidenceBand.LOW, 0.35))

    assert "Ask the user to repeat or clarify" in context
    assert "presumed request" in context


def test_high_confidence_context_allows_normal_response() -> None:
    context = ConfidenceContextBuilder().build(turn(ConfidenceBand.HIGH, 0.92))

    assert "Respond normally" in context
    assert "Ask the user to repeat or clarify" not in context


def test_unknown_confidence_context_does_not_hide_missing_metadata() -> None:
    context = ConfidenceContextBuilder().build(turn(ConfidenceBand.UNKNOWN, None))

    assert "STT confidence: unknown" in context
    assert "confidence is unavailable" in context
