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


def _ctx_per_slot(card: dict, slots: int) -> int:
    """Context one caller gets, which is what the watcher records.

    llama-server reports this in ``default_generation_settings.n_ctx``: started with -c 32768
    -np 2 it says 16384, and the client relies on that being per-slot -- the difference between
    "this node holds 32k" and "this node holds 32k, twice over, for two callers". This engine
    configures the total, so it divides.
    """
    ctx = int(card.get("ctx") or 0)
    return ctx // slots if slots > 0 and ctx > 0 else ctx


def build_props(state: Any, doc: dict, version: str) -> dict:
    """The ``/props`` body, in llama-server's spelling because that is what is read.

    Only the fields the watcher takes (``_PROPS_FIELDS`` in the client's configuration.py) plus
    the nested ``default_generation_settings.n_ctx``. Anything else here would be noise nobody
    reads, and a field with this engine's own meaning under llama-server's name would be worse
    than absent.
    """
    config = getattr(state, "config", None)
    card = doc.get("model") or {}
    slots = int(getattr(config, "max_running_requests", 0) or 1)
    return {
        "model_alias": card.get("id"),
        "model_path": getattr(config, "model_path", None),
        "model_ftype": card.get("quant"),
        "total_slots": slots,
        "build_info": f"freetoken-{version}",
        # A state, not a configuration, and this engine never idles its weights out.
        "is_sleeping": False,
        "endpoint_metrics": True,
        "default_generation_settings": {"n_ctx": _ctx_per_slot(card, slots)},
    }


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
        slots = int(getattr(config, "max_running_requests", 0) or 1)
        return "\n".join(_linee(doc, slots)) + "\n"


__all__ = ["register_node_metrics_routes", "build_props"]
