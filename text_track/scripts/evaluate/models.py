"""Model registry and provider clients.

Every provider implements the ModelClient protocol: complete(system, user, config) ->
ModelResponse. This is the seam tests substitute a FakeModelClient into (see tests/fakes.py).
"""
import dataclasses
import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ModelConfig:
    key: str
    display_name: str
    provider: str                 # "anthropic" | "openai" | "openai_router" | "hpc"
    model_id: str                  # the SERVED alias/name sent as the API "model" field
    endpoint: str | None = None
    temperature: float = 0.0
    supports_temperature: bool = True   # reasoning models often reject a non-default temperature
    max_output_tokens: int = 2048
    seed: int | None = None
    max_retries: int = 3
    # Reproducibility metadata for the run manifest (see storage.write_run_manifest) and the
    # resumption config fingerprint (see config_fingerprint()). Hosted APIs
    # (Anthropic/OpenAI/HF router) don't expose most of these to clients, so they stay None
    # for those providers rather than guessed. For HPC/vLLM models, values not known at
    # registry-definition time can be supplied via env vars at manifest-build time instead
    # of hardcoded here — see manifest_settings_for().
    hf_repo: str | None = None              # actual underlying HF repo, if different from model_id
    checkpoint_commit_sha: str | None = None
    revision: str | None = None
    tokenizer_repo: str | None = None
    context_length: int | None = None
    thinking_mode: str | None = None
    precision: str | None = None
    vllm_version: str | None = None
    flashinfer_sampler: bool | None = None


@dataclass
class ModelResponse:
    text: str
    finish_reason: str | None
    input_tokens: int | None
    output_tokens: int | None
    raw: object = None


class ModelClient(Protocol):
    def complete(self, *, system: str, user: str, config: ModelConfig) -> ModelResponse: ...


class AnthropicClient:
    """Wraps anthropic.Anthropic(). Client construction is lazy so importing this module
    never requires ANTHROPIC_API_KEY to be set (needed for HPC-only / offline test runs)."""

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(max_retries=3)
        return self._client

    def complete(self, *, system: str, user: str, config: ModelConfig) -> ModelResponse:
        client = self._get_client()
        response = client.messages.create(
            model=config.model_id,
            max_tokens=config.max_output_tokens,
            temperature=config.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(getattr(b, "text", "") for b in response.content if b.type == "text")
        if not text:
            text = f"[{(response.stop_reason or 'EMPTY').upper()}] Claude returned no text content."
        usage = getattr(response, "usage", None)
        return ModelResponse(
            text=text,
            finish_reason=response.stop_reason,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            raw=response,
        )


class OpenAICompatibleClient:
    """Wraps an OpenAI-compatible chat.completions endpoint. Used for the HF router
    (Llama), direct OpenAI (ChatGPT), and HPC-hosted vLLM (Qwen models) — these differ only
    in base_url/api_key, which come from the ModelConfig/env, not from separate classes.
    vLLM's OpenAI-compatible server generally doesn't validate the API key, so a missing env
    var falls back to the "EMPTY" placeholder vLLM's own docs recommend, rather than failing
    client construction outright."""

    def __init__(self, base_url: str | None, api_key_env: str):
        self._base_url = base_url
        self._api_key_env = api_key_env
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self._base_url,
                api_key=os.environ.get(self._api_key_env) or "EMPTY",
                max_retries=3,
                timeout=60.0,
            )
        return self._client

    def complete(self, *, system: str, user: str, config: ModelConfig) -> ModelResponse:
        client = self._get_client()
        kwargs = dict(
            model=config.model_id,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=config.max_output_tokens,
        )
        if config.supports_temperature:
            kwargs["temperature"] = config.temperature
        response = client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        return ModelResponse(
            text=choice.message.content or "",
            finish_reason=choice.finish_reason,
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            raw=response,
        )


# ---------------- Registry ---------------- #

MODEL_REGISTRY: dict[str, ModelConfig] = {
    "claude_sonnet_4_6": ModelConfig(
        key="claude_sonnet_4_6",
        display_name="Claude Sonnet 4.6",
        provider="anthropic",
        model_id="claude-sonnet-4-6",
        context_length=200_000,
    ),
    "llama_3_8b": ModelConfig(
        key="llama_3_8b",
        display_name="Llama 3 8B",
        provider="openai_router",
        model_id="meta-llama/Meta-Llama-3-8B-Instruct",
        endpoint="https://router.huggingface.co/v1",
        context_length=8192,
    ),
    "gpt_5_2_thinking": ModelConfig(
        key="gpt_5_2_thinking",
        display_name="ChatGPT 5.2 Thinking (OpenAI direct API)",
        provider="openai",
        model_id="gpt-5.2-thinking",
        # Reasoning models on the OpenAI API generally reject a non-default temperature.
        supports_temperature=False,
    ),
    # HPC-hosted via vLLM's OpenAI-compatible server. Model weights are still being staged
    # on the HPC as of this writing — endpoint comes from HPC_VLLM_BASE_URL (see .env);
    # complete() raises a clear error if that isn't set yet rather than guessing a default.
    "qwen_3_6_27b": ModelConfig(
        key="qwen_3_6_27b",
        display_name="Qwen3.6-27B (HPC via vLLM)",
        provider="hpc",
        model_id="qwen3.6-27b",
    ),
    "qwen_sealion_v4_5_27b_it": ModelConfig(
        key="qwen_sealion_v4_5_27b_it",
        display_name="Qwen-SEA-LION-v4.5-27B-IT (HPC via vLLM)",
        provider="hpc",
        model_id="qwen-sealion-v4.5-27b-it",
    ),
}

# Lets HPC-served models' reproducibility metadata be supplied at manifest-build time via
# env vars (the actual vLLM config lives outside this repo, in whatever SLURM script the
# user submits) rather than hardcoded into the registry ahead of time. Each entry is
# (env_var_name, parser) — parser converts the raw string env value to the field's real
# type, since every os.environ value is a str by construction.
def _parse_bool_env(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on", "enabled")


_HPC_ENV_OVERRIDES: dict[str, tuple[str, type]] = {
    "hf_repo": ("HPC_HF_REPO", str),
    "checkpoint_commit_sha": ("HPC_CHECKPOINT_COMMIT_SHA", str),
    "revision": ("HPC_MODEL_REVISION", str),
    "tokenizer_repo": ("HPC_TOKENIZER_REPO", str),
    "context_length": ("HPC_CONTEXT_LENGTH", int),
    "thinking_mode": ("HPC_THINKING_MODE", str),
    "precision": ("HPC_PRECISION", str),
    "vllm_version": ("HPC_VLLM_VERSION", str),
    "flashinfer_sampler": ("HPC_FLASHINFER_SAMPLER", _parse_bool_env),
}


def manifest_settings_for(model_key: str) -> dict:
    """Reproducibility metadata for one model, for embedding in the run manifest. Env-var
    overrides are parsed into their proper types (int/bool), not left as raw strings."""
    config = MODEL_REGISTRY[model_key]
    settings = dataclasses.asdict(config)
    if config.provider == "hpc":
        for field_name, (env_name, parser) in _HPC_ENV_OVERRIDES.items():
            raw_value = os.environ.get(env_name)
            if raw_value is not None:
                settings[field_name] = parser(raw_value)
    return settings


def config_fingerprint(model_key: str, tokenizer_revision: str | None = None) -> str:
    """A short, stable hash of everything that determines this model's actual behavior:
    exact checkpoint, precision, sampling settings, serving config, and tokenizer revision.
    Included in every ResultRecord and the resume key (see storage.make_resume_key) so a
    changed model configuration — e.g. a different vLLM precision, or a checkpoint update —
    can never be silently treated as equivalent to an older run's results."""
    settings = manifest_settings_for(model_key)
    settings["tokenizer_revision"] = tokenizer_revision
    canonical = json.dumps(settings, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


_CLIENT_CACHE: dict[str, ModelClient] = {}


def get_client(model_key: str) -> ModelClient:
    if model_key in _CLIENT_CACHE:
        return _CLIENT_CACHE[model_key]

    config = MODEL_REGISTRY[model_key]
    if config.provider == "anthropic":
        client: ModelClient = AnthropicClient()
    elif config.provider == "openai_router":
        client = OpenAICompatibleClient(base_url=config.endpoint, api_key_env="HF_TOKEN")
    elif config.provider == "openai":
        client = OpenAICompatibleClient(base_url=config.endpoint, api_key_env="OPENAI_API_KEY")
    elif config.provider == "hpc":
        base_url = config.endpoint or os.environ.get("HPC_VLLM_BASE_URL")
        if not base_url:
            raise RuntimeError(
                f"HPC endpoint not configured for {model_key!r}. Set HPC_VLLM_BASE_URL in "
                f".env once the vLLM server is running (e.g. http://<hpc-node>:8000/v1)."
            )
        client = OpenAICompatibleClient(base_url=base_url, api_key_env="HPC_VLLM_API_KEY")
    else:
        raise ValueError(f"Unknown provider: {config.provider!r}")

    _CLIENT_CACHE[model_key] = client
    return client


def complete_with_retry(model_key: str, *, system: str, user: str) -> tuple[ModelResponse | None, Exception | None, int, float]:
    """Call the model, catching exceptions so callers can classify infrastructure failures
    without a bare try/except at every call site. Returns (response, exception, retry_count,
    latency_ms). Actual retry/backoff is delegated to each SDK's own max_retries."""
    config = MODEL_REGISTRY[model_key]
    client = get_client(model_key)
    start = time.monotonic()
    try:
        response = client.complete(system=system, user=user, config=config)
        latency_ms = (time.monotonic() - start) * 1000
        return response, None, 0, latency_ms
    except Exception as e:  # noqa: BLE001 - deliberately broad: any failure here is INFRASTRUCTURE_API_FAILURE
        latency_ms = (time.monotonic() - start) * 1000
        return None, e, 0, latency_ms
