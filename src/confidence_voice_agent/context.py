from dataclasses import dataclass

from confidence_voice_agent.confidence import ConfidenceBand, TranscriptionTurn


@dataclass(frozen=True, slots=True)
class ConfidenceContextBuilder:
    def build(self, turn: TranscriptionTurn) -> str:
        confidence = "unknown" if turn.confidence is None else f"{turn.confidence:.2f}"
        instruction = self._instruction_for(turn.band)
        return "\n".join(
            [
                "Speech recognition metadata for the latest user turn:",
                f'User said: "{self._safe_text(turn.text)}"',
                f"STT confidence: {confidence}",
                f"Confidence band: {turn.band.value}",
                f"Instruction: {instruction}",
            ]
        )

    def _instruction_for(self, band: ConfidenceBand) -> str:
        if band is ConfidenceBand.LOW:
            return "Ask the user to repeat or clarify instead of answering the presumed request."
        if band is ConfidenceBand.UNKNOWN:
            return "Ask the user to repeat because speech recognition confidence is unavailable."
        if band is ConfidenceBand.MEDIUM:
            return (
                "Answer cautiously and ask a short clarifying question if the intent is ambiguous."
            )
        return "Respond normally to the user's request."

    def _safe_text(self, text: str) -> str:
        return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
