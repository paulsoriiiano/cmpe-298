"""FakeModelClient: a ModelClient implementation returning scripted responses, so tests
never make a real network call. Substitute it into models._CLIENT_CACHE (see test_run.py /
test_resumption.py) rather than the real Anthropic/OpenAI clients.
"""
from dataclasses import dataclass, field

from ..models import ModelResponse


class FakeModelClient:
    """responder(system, user) -> ModelResponse | Exception. Every call is recorded in
    .calls so tests can assert exactly what was (or wasn't) sent, e.g. to verify a resumed
    A_IE run does not re-issue a completed translation call."""

    def __init__(self, responder):
        self.responder = responder
        self.calls: list[tuple[str, str]] = []

    def complete(self, *, system: str, user: str, config) -> ModelResponse:
        self.calls.append((system, user))
        result = self.responder(system, user)
        if isinstance(result, Exception):
            raise result
        return result


def canned_response(
    text: str, finish_reason: str = "stop", input_tokens: int = 10, output_tokens: int = 10,
) -> ModelResponse:
    return ModelResponse(
        text=text, finish_reason=finish_reason, input_tokens=input_tokens,
        output_tokens=output_tokens, raw=None,
    )


# One scripted response per FailureType branch, for grading.classify_result() coverage.
CANNED = {
    "correct_number": canned_response("Reasoning...\n<answer>109</answer>"),
    "correct_letter": canned_response("Reasoning...\n<answer>A</answer>"),
    "wrong_number": canned_response("Reasoning...\n<answer>42</answer>"),
    "invalid_format": canned_response("Reasoning...\n<answer>maybe purple</answer>"),
    "missing_answer": canned_response("I thought about it but here's my reasoning only."),
    "truncated": canned_response("Reasoning that got cut off mid-sen", finish_reason="length"),
    "refusal": canned_response("I cannot help with that request."),
    "repetitive": canned_response(" ".join(["loop token repeat"] * 40)),
    "translation_ok": canned_response("This is the translated English question text."),
    "translation_solved_instead": canned_response("The translation is done.\n<answer>109</answer>"),
    "translation_empty": canned_response(""),
}
