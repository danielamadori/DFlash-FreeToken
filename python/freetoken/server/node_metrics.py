"""``/props`` and ``/metrics``, the two endpoints a llama.cpp-shaped client asks for.

WHY THIS EXISTS. The BOS device client watches a node by fetching the engine's ``/props`` (what
is it running) and ``/metrics`` (how is it going), both at the engine's root rather than under
``/v1``. Those are llama-server's endpoints, and this engine is not llama-server: it answered
404 to both, so a node running it reported a heartbeat and never one metric. Measured on
thething the 2026-09-14: 5530 heartbeats accepted, ``last_metrics_at`` empty since the asset
was created.

The numbers were never missing -- ``/v1/stats`` has all of them. What was missing was the two
shapes the watcher reads. So this serves the same figures twice rather than collecting them
twice: one source, two dialects, and no second place to go stale.

The metric names follow llama.cpp's ``llamacpp:`` exposition wherever the concept is the same,
because dashboards keyed on those names already exist and a node that renamed them would be
invisible in a different way. Where this engine has something llama-server does not -- the GDN
state slots of a hybrid model -- the name says ``freetoken:``, so nobody reads it as something
llama.cpp also reports.
"""

from __future__ import annotations

from typing import Any, Callable

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

from .generation import ACCEPTED_CONTENT_PART_TYPES

# OpenAI names a content part by its ``type``; llama-server's ``/props`` names the same
# abilities as modalities. One map, so the two vocabularies cannot drift apart.
_PART_TYPE_TO_MODALITY = {
    "image_url": "vision",
    "video_url": "video",
    "input_audio": "audio",
}


def _ctx_per_slot(card: dict) -> int:
    """The longest prompt one caller can send, which is what this field is read for.

    It used to divide the total by the slot count, copied from llama-server where `-c 32768
    -np 2` really does partition the cache and each slot holds 16384. **This engine has no such
    partition**: the KV cache is one pool allocated page by page, and any caller may take up to
    the model ceiling. Importing another engine's semantics understated this node by 4x, and the
    two halves did not even agree with each other -- 4 x 16384 is 65536 while the pool actually
    held 122351 tokens.

    Measured on thething, 2026-09-15, config `max_running_req=4`, `max_seq_len=65536`:

        one caller           59428 tokens of prompt: fine; 68289: refused, "> 65536 maximum"
        two callers at 59k   20.7 s against 20.5 s for one -- they run together, free
        three                41.5 s; four 61.9 s -- queued, never refused
        KV pool              122351 tokens, so exactly two full-context callers fit

    So the ceiling a caller meets is `max_seq_len`, and reporting the division would have a
    router turn away a 20k prompt this node answers without noticing. The sharing limit -- two
    at full size, the rest queued -- is a different fact, and llama-server's shape has no field
    for it: it belongs in the backlog, not in an invented key.
    """
    return int(card.get("ctx") or 0)


def _ctx_effettivo(card: dict, doc: dict) -> int:
    """The ceiling the engine ACTUALLY enforces: the lower of the model ceiling and the pool.

    ``max_seq_len`` is a ceiling on what a caller may ASK for; the KV pool is how much the card
    could hold. They are sized independently -- the pool comes from the free-memory ratio -- so
    raising the ceiling does not grow the pool, and the node would then advertise a number it
    refuses. Measured on thething 2026-09-15: ceiling raised 65536 -> 128000, pool stayed at
    121899, and a 127711-token prompt came back

        prompt is too long: 127711 tokens > 121899 maximum

    The engine already refuses at the true number. Reporting the other one would make the node
    the only party in the chain that believes the wrong figure -- which is the shape of defect
    this file has already produced twice today.
    """
    tetto = _ctx_per_slot(card)
    pagine = int(((doc.get("kv") or {}).get("total_pages")) or 0)
    if tetto and pagine:
        return min(tetto, pagine)
    return tetto or pagine


# The ServerArgs field that holds how many requests run at once. Spelled out here, once,
# because getattr with a wrong name does not raise: it takes the default and the endpoint
# reports a plausible number that is not the engine's. That is what happened -- this asked for
# "max_running_requests", which no version of ServerArgs has ever had, so a node configured for
# 4 told BOS it had 1 slot and, since ctx_per_slot divides by it, claimed 65536 of context per
# caller instead of 16384. Both are stored on the asset, so the fleet page was wrong about the
# two facts this endpoint exists to carry, and the tests passed because the fixture used the
# same wrong name. test_the_slot_field_is_the_one_ServerArgs_has guards the spelling now.
_SLOTS_FIELD = "max_running_req"


def _slots(config: Any) -> int:
    """How many requests this engine serves at once, never fewer than 1."""
    return int(getattr(config, _SLOTS_FIELD, 0) or 1)


def build_props(state: Any, doc: dict, version: str) -> dict:
    """The ``/props`` body, in llama-server's spelling because that is what is read.

    Only the fields the watcher takes (``_PROPS_FIELDS`` in the client's configuration.py) plus
    the nested ``default_generation_settings.n_ctx``. Anything else here would be noise nobody
    reads, and a field with this engine's own meaning under llama-server's name would be worse
    than absent.
    """
    config = getattr(state, "config", None)
    card = doc.get("model") or {}
    slots = _slots(config)
    return {
        "model_alias": card.get("id"),
        "model_path": getattr(config, "model_path", None),
        "model_ftype": card.get("quant"),
        "total_slots": slots,
        "build_info": f"freetoken-{version}",
        # A state, not a configuration, and this engine never idles its weights out.
        "is_sleeping": False,
        "endpoint_metrics": True,
        "modalities": _modalities(),
        "default_generation_settings": {"n_ctx": _ctx_effettivo(card, doc)},
    }


def _modalities() -> dict:
    """What a caller may SEND this engine, in llama-server's spelling.

    A cluster that routes by model name alone sends a photo wherever that name is served and
    finds out it does not fit only when the node refuses it -- which is what happens today:
    the request crosses the hub, is leased as a job, reaches the engine and dies there, and
    the caller reads a refusal instead of being routed somewhere that could have answered.
    A router can only do better if every node SAYS what it takes, so this says it.

    Derived from ``ACCEPTED_CONTENT_PART_TYPES``, the tuple the front door actually checks,
    and never from the model's own abilities: a checkpoint with a vision tower loaded would
    still be refused an image by this API, and advertising vision because the weights are
    there would be a node promising what it then rejects. When the API learns a part type,
    this follows without anyone remembering to update it.
    """
    accettate = {_PART_TYPE_TO_MODALITY[t] for t in ACCEPTED_CONTENT_PART_TYPES
                 if t in _PART_TYPE_TO_MODALITY}
    return {nome: nome in accettate for nome in ("vision", "video", "audio")}


def _linee(doc: dict, slots: int) -> list[str]:
    """The exposition, one metric at a time, each with the HELP and TYPE Prometheus wants."""
    kv = doc.get("kv") or {}
    mamba = doc.get("mamba") or {}
    req = doc.get("requests") or {}
    thr = doc.get("throughput") or {}
    pagine = int(kv.get("total_pages") or 0)
    usate = int(kv.get("used_pages") or 0)

    def metrica(nome: str, aiuto: str, tipo: str, valore: float) -> list[str]:
        return [f"# HELP {nome} {aiuto}", f"# TYPE {nome} {tipo}",
                f"{nome} {valore:g}"]

    out: list[str] = []
    out += metrica("llamacpp:prompt_tokens_total", "Number of prompt tokens processed.",
                   "counter", req.get("prompt_tokens_total") or 0)
    out += metrica("llamacpp:tokens_predicted_total", "Number of generation tokens processed.",
                   "counter", req.get("completion_tokens_total") or 0)
    out += metrica("llamacpp:prompt_tokens_seconds", "Average prompt throughput in tokens/s.",
                   "gauge", thr.get("prefill_tps") or 0)
    out += metrica("llamacpp:predicted_tokens_seconds",
                   "Average generation throughput in tokens/s.",
                   "gauge", thr.get("decode_tps") or 0)
    out += metrica("llamacpp:kv_cache_usage_ratio", "KV-cache usage. 1 means 100 percent usage.",
                   "gauge", (usate / pagine) if pagine else 0.0)
    out += metrica("llamacpp:kv_cache_tokens", "KV-cache tokens.",
                   "gauge", usate * int(kv.get("page_size") or 1))
    out += metrica("llamacpp:requests_processing", "Number of requests processing.",
                   "gauge", req.get("active") or 0)
    out += metrica("llamacpp:requests_deferred", "Number of requests deferred.", "gauge", 0)
    # This engine's own, under its own prefix: a hybrid model's recurrent-state slots are a
    # second budget that runs out on its own schedule, and llama.cpp has no such thing.
    if mamba:
        out += metrica("freetoken:state_slots_used",
                       "Recurrent-state slots in use by a hybrid model.",
                       "gauge", mamba.get("used_slots") or 0)
        out += metrica("freetoken:state_slots_total", "Recurrent-state slots configured.",
                       "gauge", mamba.get("total_slots") or 0)
    out += metrica("freetoken:vram_bytes", "Device memory held by this engine.",
                   "gauge", doc.get("vram_bytes") or 0)
    out += metrica("freetoken:slots_total", "Concurrent requests this engine is configured for.",
                   "gauge", slots)
    out += metrica("freetoken:uptime_seconds", "Seconds since this engine became ready.",
                   "counter", doc.get("uptime_s") or 0)
    return out


def register_node_metrics_routes(
    app: FastAPI,
    get_state: Callable[[], Any],
    build_doc: Callable[[], dict],
    version: str,
) -> None:
    """Register ``/props`` and ``/metrics`` at the root, where the watcher looks for them."""

    @app.get("/props")
    async def props():
        return build_props(get_state(), build_doc(), version)

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics():
        doc = build_doc()
        config = getattr(get_state(), "config", None)
        slots = _slots(config)
        return "\n".join(_linee(doc, slots)) + "\n"


__all__ = ["register_node_metrics_routes", "build_props"]
