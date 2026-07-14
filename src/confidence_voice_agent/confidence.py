from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from livekit.agents import stt
from livekit.agents.types import NotGiven


class ConfidenceBand(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ConfidencePolicy:
    low_threshold: float = 0.6
    high_threshold: float = 0.8

    def classify(self, confidence: float | None) -> ConfidenceBand:
        if confidence is None or not _is_valid_confidence(confidence):
            return ConfidenceBand.UNKNOWN
        if confidence < self.low_threshold:
            return ConfidenceBand.LOW
        if confidence > self.high_threshold:
            return ConfidenceBand.HIGH
        return ConfidenceBand.MEDIUM


@dataclass(frozen=True, slots=True)
class TranscriptionTurn:
    text: str
    confidence: float | None
    band: ConfidenceBand
    request_id: str = ""
    language: str | None = None
    has_bad_provider_confidence: bool = False


class ConfidenceExtractor:
    def __init__(self, policy: ConfidencePolicy) -> None:
        self._policy = policy

    def extract_turn(self, event: stt.SpeechEvent) -> TranscriptionTurn | None:
        if event.type not in {
            stt.SpeechEventType.PREFLIGHT_TRANSCRIPT,
            stt.SpeechEventType.FINAL_TRANSCRIPT,
        }:
            return None
        if not event.alternatives:
            return None

        alternative = event.alternatives[0]
        text = alternative.text.strip()
        if not text:
            return None

        confidence, has_bad_provider_confidence = self._extract_confidence(alternative)
        return TranscriptionTurn(
            text=text,
            confidence=confidence,
            band=self._policy.classify(confidence),
            request_id=event.request_id,
            language=str(alternative.language) if alternative.language else None,
            has_bad_provider_confidence=has_bad_provider_confidence,
        )

    def _extract_confidence(self, alternative: stt.SpeechData) -> tuple[float | None, bool]:
        segment_confidence = _normalize_confidence(alternative.confidence)
        if segment_confidence.is_present:
            return segment_confidence.value, segment_confidence.is_bad
        if segment_confidence.is_bad:
            return None, True

        word_confidences = []
        has_bad_provider_confidence = False
        for word in alternative.words or []:
            word_confidence = _normalize_confidence(getattr(word, "confidence", None))
            if word_confidence.is_bad:
                has_bad_provider_confidence = True
            if word_confidence.is_present and word_confidence.value is not None:
                word_confidences.append(word_confidence.value)

        if not word_confidences:
            return None, has_bad_provider_confidence

        return sum(word_confidences) / len(word_confidences), has_bad_provider_confidence


@dataclass(frozen=True, slots=True)
class _NormalizedConfidence:
    value: float | None
    is_present: bool
    is_bad: bool


def _normalize_confidence(value: Any) -> _NormalizedConfidence:
    if value is None or isinstance(value, NotGiven):
        return _NormalizedConfidence(value=None, is_present=False, is_bad=False)
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return _NormalizedConfidence(value=None, is_present=False, is_bad=True)
    if not _is_valid_confidence(confidence):
        return _NormalizedConfidence(value=None, is_present=False, is_bad=True)
    return _NormalizedConfidence(value=confidence, is_present=True, is_bad=False)


def _is_valid_confidence(confidence: float) -> bool:
    return 0 <= confidence <= 1
