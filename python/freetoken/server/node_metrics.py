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

# The engine names a served modality "image"; llama-server's ``/props`` calls the same
# ability "vision". One map, so the two vocabularies cannot drift apart.
_ENGINE_MODALITY_TO_PROPS = {
    "image": "vision",
    "video": "video",
    "audio": "audio",
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


def _ctx_effettivo(card: dict, doc: dict, page_size: int) -> int:
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

    AND THE POOL IS COUNTED IN PAGES, NOT TOKENS. ``total_pages`` is the manager's
    ``num_pages``; a page holds ``page_size`` of them, which is why every token-valued
    quantity next to it multiplies: ``available_size`` is
    ``evictable + len(free_slots) * page_size``, and the prefill adder's
    ``_kv_reservation_size`` (upstream #367) returns, in its own words, «the
    token-equivalent cost of the additional KV pages». Comparing a token ceiling with a
    page count is right only where ``--page-size`` is 1 -- which is what thething and the
    Dell run, so the measurement above stands. On a node started with ``--page-size 16``
    it would advertise a SIXTEENTH of the context it holds, and a router would turn away
    prompts the node answers: the same defect this function exists to remove, pointing the
    other way.

    The min is KEPT and the disagreement is not swallowed with it: ``_divergenza_ctx`` publishes
    both halves and the reason beside this number whenever the two differ.
    """
    tetto = _ctx_per_slot(card)
    fondo = _fondo_kv(doc, page_size)
    if tetto and fondo:
        return min(tetto, fondo)
    return tetto or fondo


def _fondo_kv(doc: dict, page_size: int) -> int:
    """The KV pool in TOKENS: ``total_pages`` is a page count and a page holds ``page_size``."""
    pagine = int(((doc.get("kv") or {}).get("total_pages")) or 0)
    return pagine * max(int(page_size or 1), 1)


def _divergenza_ctx(card: dict, doc: dict, page_size: int) -> dict:
    """The two halves of the ``min`` above, published whenever they disagree -- and nothing
    when they agree.

    The ``min`` is right: the engine refuses at the lower number, and advertising the other
    would leave this node the only party in the chain believing a figure it will not serve.
    But taking the lower number and saying nothing else DELETES THE DISAGREEMENT, and that is
    what cost the days. The Dell published 27931 and no one -- not the hub, not the node's own
    agent -- could say where it came from: it is not a knob anybody set, it is the pool the
    free VRAM happened to buy at startup, so it lands somewhere new after every restart while
    ``max_seq_len`` sits unchanged in the config that is supposed to explain it.

    So the winner stays in ``n_ctx``, where the watcher reads it, and the loser is published
    beside it with the reason. The shape is the one the chain already knows: the hub answers
    ``/v1/models`` with ``n_ctx`` + ``n_ctx_observed`` + ``n_ctx_differs_because`` exactly when
    two numbers disagree, and drops the extra fields when they do not. The names here are this
    engine's own because the pair is a different pair -- the hub's is declared-vs-engine, this
    one is ceiling-vs-pool -- and reusing a name for another meaning is the defect this file
    removes elsewhere.
    """
    tetto = _ctx_per_slot(card)
    fondo = _fondo_kv(doc, page_size)
    if not tetto or not fondo or tetto == fondo:
        return {}
    return {
        "n_ctx_configured": tetto,
        "n_ctx_kv_pool": fondo,
        "n_ctx_differs_because": (
            f"max_seq_len is configured at {tetto} while the KV pool allocated at startup "
            f"holds {fondo} tokens. The engine refuses at the lower of the two, so n_ctx is "
            f"{min(tetto, fondo)}. The pool is sized from the VRAM free when this process "
            "started, not from the ceiling, so it changes across restarts while the "
            "configured ceiling does not."
        ),
    }


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
    page_size = getattr(config, "page_size", 1)
    props = {
        "model_alias": card.get("id"),
        "model_path": getattr(config, "model_path", None),
        "model_ftype": _model_ftype(card, config),
        "total_slots": slots,
        "build_info": f"freetoken-{version}",
        # A state, not a configuration, and this engine never idles its weights out.
        "is_sleeping": False,
        "endpoint_metrics": True,
        "modalities": _modalities(config),
        "default_generation_settings": {
            "n_ctx": _ctx_effettivo(card, doc, page_size)
        },
    }
    # Only when they disagree, so a reader who sees the keys knows something is being resolved
    # and a reader who does not see them knows nothing is.
    props.update(_divergenza_ctx(card, doc, page_size))
    return props


_FTYPE_PER_DTYPE = {
    "bfloat16": "BF16",
    "float16": "F16",
    "half": "F16",
    "float32": "F32",
    "float": "F32",
}


# llama.cpp's ``llama_ftype`` -> the spelling ``llama_ftype_name`` prints, so a GGUF node
# names its quantisation the way the llama.cpp nodes beside it on the fleet page do.
# Generated from llama.h (the enum values) and llama-model-loader.cpp (the strings);
# an ftype absent here is left unnamed rather than approximated.
_FTYPE_PER_GGUF = {
    0: "all F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    7: "Q8_0",
    8: "Q5_0",
    9: "Q5_1",
    10: "Q2_K - Medium",
    11: "Q3_K - Small",
    12: "Q3_K - Medium",
    13: "Q3_K - Large",
    14: "Q4_K - Small",
    15: "Q4_K - Medium",
    16: "Q5_K - Small",
    17: "Q5_K - Medium",
    18: "Q6_K",
    19: "IQ2_XXS - 2.0625 bpw",
    20: "IQ2_XS - 2.3125 bpw",
    21: "Q2_K - Small",
    22: "IQ3_XS - 3.3 bpw",
    23: "IQ3_XXS - 3.0625 bpw",
    24: "IQ1_S - 1.5625 bpw",
    25: "IQ4_NL - 4.5 bpw",
    26: "IQ3_S - 3.4375 bpw",
    27: "IQ3_S mix - 3.66 bpw",
    28: "IQ2_S - 2.5 bpw",
    29: "IQ2_M - 2.7 bpw",
    30: "IQ4_XS - 4.25 bpw",
    31: "IQ1_M - 1.75 bpw",
    32: "BF16",
    36: "TQ1_0 - 1.69 bpw ternary",
    37: "TQ2_0 - 2.06 bpw ternary",
    38: "MXFP4 MoE",
    39: "NVFP4",
    40: "Q1_0",
    41: "Q2_0",
}


def _codice_ftype_gguf(model_path: Any) -> int | None:
    """The checkpoint's own ``general.file_type``, or None when there is no GGUF to ask.

    Imported inside the function: ``/props`` must answer on a node whose model is not a
    GGUF, and on one where gguf-py is not importable, without either turning into a 500.
    """
    if not model_path:
        return None
    try:
        from freetoken.models.gguf.reader import gguf_file_type

        return gguf_file_type(str(model_path))
    except Exception:
        return None


def _model_ftype(card: dict, config: Any) -> str | None:
    """How the weights are stored, in llama-server's spelling.

    Three sources, in the order of how directly each one knows the answer.

    ``card["quant"]`` is what the model card states outright, and nothing overrides it.

    Then the checkpoint's own ``general.file_type``. This engine never fills ``quant``,
    so before this a GGUF node fell through to the dtype below and answered BF16 for a
    Q4_K_S file: the weights are four-bit on disk and bfloat16 only once dequantised on
    the way to the kernels. The fleet page then read "Quantization: BF16" for a node
    serving Q4_K_S -- a wrong statement in the field a router compares, which is worse
    than the empty field it replaced, and exactly the guesswork the note below forbids.

    Last the engine's dtype, which is the right answer for a safetensors checkpoint:
    there the weights ARE held at it, and a node that reported nothing now reports BF16
    like the quantised nodes report theirs.

    An unrecognised dtype is left ABSENT rather than guessed: "not reported" is true,
    and a made-up spelling in a field other tools read by name is not.
    """
    dichiarato = card.get("quant")
    if dichiarato:
        return str(dichiarato)
    codice = _codice_ftype_gguf(getattr(config, "model_path", None))
    if codice is not None:
        # A GGUF that declares an ftype this table cannot name is left ABSENT. Falling
        # through to the dtype here would put back the very claim this branch removes:
        # the weights are quantised, whatever the engine computes in.
        return _FTYPE_PER_GGUF.get(int(codice))
    dtype = getattr(config, "dtype", None)
    if dtype is None:
        return None
    # torch.bfloat16 renders as "torch.bfloat16"; a plain string is taken as it comes.
    nome = str(dtype).rsplit(".", 1)[-1].strip().lower()
    return _FTYPE_PER_DTYPE.get(nome)


def _modalities(config: Any) -> dict:
    """What a caller may SEND this engine, in llama-server's spelling.

    A cluster that routes by model name alone sends a photo wherever that name is served and
    finds out it does not fit only when the node refuses it: the request crosses the hub, is
    leased as a job, reaches the engine and dies there, and the caller reads a refusal instead
    of being routed somewhere that could have answered. A router can only do better if every
    node SAYS what it takes, so this says it.

    Read from ``config.served_modalities``, which is the union of the encoder towers this
    PROCESS actually built -- the family registers them, the checkpoint config must carry the
    section, and ``--mm-disable`` can switch one off. The same field decides whether an image
    is accepted: ``mm/media.image_reject_reason`` refuses when "image" is not in it, and
    ``/v1/stats`` publishes ``input_modalities`` from it. Three statements, one source, so a
    node cannot advertise an ability its own front door refuses.

    Never from the model's abilities on paper. A GGUF of a vision-capable checkpoint is the
    case that makes the difference: the header carries no vision section, so no tower is built
    and no image can be answered, however multimodal the original release is.
    """
    servite = getattr(config, "served_modalities", None) or frozenset()
    nomi = {_ENGINE_MODALITY_TO_PROPS[m] for m in servite if m in _ENGINE_MODALITY_TO_PROPS}
    return {nome: nome in nomi for nome in ("vision", "video", "audio")}


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
