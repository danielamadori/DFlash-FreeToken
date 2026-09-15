"""What a llama.cpp-shaped watcher reads off this engine, and why it is shaped that way.

The BOS device client fetches ``/props`` and ``/metrics`` at the engine's ROOT. Those are
llama-server's endpoints; this engine answered 404 to both, so a node running it reported
heartbeats and never one metric -- 5530 accepted heartbeats against an empty last_metrics_at,
measured on thething the 2026-09-14. The numbers were never missing: /v1/stats had them all.
"""

from __future__ import annotations

import json

from types import SimpleNamespace

from freetoken.server.node_metrics import build_props, _ctx_per_slot, _linee


def _doc(ctx=65536, used=10, total=100, slots_used=1, slots_total=25):
    return {
        "model": {"id": "Qwen3.8-27B-UD-Q4_K_S.gguf", "ctx": ctx, "quant": "Q4_K_S"},
        "kv": {"used_pages": used, "total_pages": total, "page_size": 1},
        "mamba": {"used_slots": slots_used, "total_slots": slots_total},
        "uptime_s": 3600,
        "vram_bytes": 22670213120,
        "throughput": {"decode_tps": 105.4, "prefill_tps": 0.0},
        "requests": {"active": 2, "completed": 6, "prompt_tokens_total": 1000,
                     "completion_tokens_total": 500},
    }


def _state(slots=1, model_path="/models/qwen.gguf"):
    # The field name is imported, not written out: a fixture that spells it itself is green
    # whatever the engine calls it, which is exactly how a node came to report 1 slot while
    # serving 4.
    from freetoken.server.node_metrics import _SLOTS_FIELD

    return SimpleNamespace(config=SimpleNamespace(**{_SLOTS_FIELD: slots},
                                                  model_path=model_path))


def test_props_reports_the_context_of_ONE_slot():
    """The watcher records this as ctx_per_slot, which is the difference between 'this node
    holds 64k' and 'this node holds 64k, twice over, for two callers'. llama-server divides,
    so an engine that reported the total would overstate every multi-slot node."""
    assert _ctx_per_slot({"ctx": 65536}, 1) == 65536
    assert _ctx_per_slot({"ctx": 65536}, 2) == 32768
    assert _ctx_per_slot({"ctx": 65536}, 4) == 16384


def test_props_survives_an_engine_that_has_not_reported_a_context_yet():
    """Before the readiness meta arrives the card is empty, and a watcher asking then must get
    an answer rather than a traceback."""
    assert _ctx_per_slot({}, 4) == 0
    assert _ctx_per_slot({"ctx": 1024}, 0) == 1024


def test_props_carries_exactly_the_fields_the_watcher_takes():
    """Named as llama-server spells them, because that is what is read. A field carrying this
    engine's own meaning under llama-server's name would be worse than an absent one."""
    p = build_props(_state(slots=2), _doc(), "9.9.9")
    assert p["model_alias"] == "Qwen3.8-27B-UD-Q4_K_S.gguf"
    assert p["model_path"] == "/models/qwen.gguf"
    assert p["total_slots"] == 2
    assert p["default_generation_settings"]["n_ctx"] == 32768
    assert p["endpoint_metrics"] is True
    assert p["is_sleeping"] is False
    assert p["build_info"].startswith("freetoken-")


def test_the_exposition_is_parseable_and_every_metric_is_declared():
    """Prometheus needs a HELP and a TYPE before each sample, and BOS counts the samples it
    parses: an exposition it cannot read pushes zero and looks exactly like no engine."""
    righe = _linee(_doc(), 1)
    nomi = [r.split()[0] for r in righe if not r.startswith("#")]
    assert nomi, "an exposition with no samples is indistinguishable from a dead engine"
    testo = "\n".join(righe)
    for nome in nomi:
        assert f"# HELP {nome} " in testo
        assert f"# TYPE {nome} " in testo


def test_the_names_are_llama_cpp_where_the_concept_is_the_same():
    """Dashboards keyed on those names already exist; a node that renamed them would be
    invisible in a new way. The hybrid state slots get this engine's own prefix, because
    llama.cpp has no such budget and nobody should read it as one it reports."""
    righe = _linee(_doc(), 1)
    testo = "\n".join(righe)
    for nome in ("llamacpp:prompt_tokens_total", "llamacpp:tokens_predicted_total",
                 "llamacpp:kv_cache_usage_ratio", "llamacpp:requests_processing"):
        assert f"\n{nome} " in "\n" + testo
    assert "freetoken:state_slots_used" in testo
    assert "llamacpp:state_slots_used" not in testo


def test_the_kv_ratio_is_a_ratio_and_does_not_divide_by_zero():
    """A cold engine has no pages at all, and a watcher polling it must not get a 500."""
    testo = "\n".join(_linee(_doc(used=25, total=100), 1))
    assert "llamacpp:kv_cache_usage_ratio 0.25" in testo
    vuoto = _doc()
    vuoto["kv"] = {"used_pages": 0, "total_pages": 0, "page_size": 1}
    assert "llamacpp:kv_cache_usage_ratio 0" in "\n".join(_linee(vuoto, 1))


def test_a_non_hybrid_model_reports_no_state_slots_rather_than_zero():
    """Zero slots and no slot budget are different facts, and a dashboard that saw 0 would
    draw a line that means 'exhausted' for a model that has no such budget."""
    d = _doc()
    d["mamba"] = None
    testo = "\n".join(_linee(d, 1))
    assert "freetoken:state_slots" not in testo


def test_props_declares_what_the_front_door_accepts() -> None:
    """A cluster that routes by model name alone sends a photo wherever that name is
    served and learns it does not fit only when the node refuses it. The router can do
    better only if the node says what it takes."""
    from freetoken.server.node_metrics import build_props

    props = build_props(_state(), _doc(), "0.1.2")

    assert props["modalities"] == {"vision": False, "video": False, "audio": False}


def test_the_advertisement_follows_the_tuple_the_api_checks() -> None:
    """Never from the model's own abilities: a checkpoint with a vision tower loaded is
    still refused an image by this API, so advertising vision because the weights are
    there would be a node promising what it then rejects. Teaching the API a part type
    must move the advertisement with it, without anyone remembering to."""
    import freetoken.server.node_metrics as nm

    originale = nm.ACCEPTED_CONTENT_PART_TYPES
    try:
        nm.ACCEPTED_CONTENT_PART_TYPES = ("text", "image_url")
        assert nm.build_props(_state(), _doc(), "0.1.2")["modalities"]["vision"] is True
    finally:
        nm.ACCEPTED_CONTENT_PART_TYPES = originale

    assert nm.build_props(_state(), _doc(), "0.1.2")["modalities"]["vision"] is False


def test_the_slot_field_is_the_one_ServerArgs_has() -> None:
    """The one assertion the fixture cannot make for itself.

    ``getattr(config, "wrong_name", 0)`` does not raise: it takes the default, and the endpoint
    answers a plausible number that is not the engine's. This node reported 1 slot while serving
    4, and 65536 of context per caller instead of 16384, because the name asked for here never
    existed -- with every test green, since the fixture built a namespace around the same wrong
    name. Comparing against the real ServerArgs is what closes that loop.
    """
    from freetoken.server.args import ServerArgs
    from freetoken.server.node_metrics import _SLOTS_FIELD

    assert hasattr(ServerArgs, _SLOTS_FIELD), (
        f"node_metrics reads config.{_SLOTS_FIELD}, which ServerArgs does not have: "
        f"/props would answer the default instead of the engine's real slot count"
    )


def test_a_response_names_the_model_that_answered_not_the_one_asked_for() -> None:
    """A misroute must be visible in the answer.

    This engine accepts any model name: measured on thething 2026-09-15, `gpt-4` and
    `nome-inventato` both returned 200. Echoing the requested name back meant a request that
    landed on the wrong node came home with a confident answer wearing the right label -- the
    client asked X, the body said X, the text came from Y, and no point in the chain showed the
    swap. Naming the model that actually ran does not make the misroute correct; it makes it
    detectable, which is the whole difference.
    """
    from types import SimpleNamespace

    import freetoken.server.api_server as api_server
    from freetoken.server.openai_api import _answering_model

    originale = api_server._GLOBAL_STATE
    try:
        api_server._GLOBAL_STATE = SimpleNamespace(
            config=SimpleNamespace(served_model_name="Qwen3.8-27B"))
        assert _answering_model(SimpleNamespace(model="gpt-4")) == "Qwen3.8-27B"
        assert _answering_model(SimpleNamespace(model="Qwen3.8-27B")) == "Qwen3.8-27B"
        # Un server che non sa dire il proprio nome ricade sul richiesto, non su None.
        api_server._GLOBAL_STATE = SimpleNamespace(config=SimpleNamespace(served_model_name=None))
        assert _answering_model(SimpleNamespace(model="gpt-4")) == "gpt-4"
    finally:
        api_server._GLOBAL_STATE = originale


def test_a_model_this_node_does_not_serve_is_refused() -> None:
    """No fallback: Daniel's call, 2026-09-15.

    Answering any name meant a misrouted request came back as a normal answer from whatever
    model happened to be loaded. Naming the real model in the reply made the swap visible but
    did not stop it, and a plausible text from the wrong model is indistinguishable from a right
    one to anything downstream not reading that field. The refusal names what IS served, because
    an error that does not say what would have worked costs a round trip to find out.
    """
    from types import SimpleNamespace

    from freetoken.server.openai_api import _model_refusal

    state = SimpleNamespace(config=SimpleNamespace(served_model_name="Qwen3.8-27B",
                                                   model_path="/m/q.gguf"))

    assert _model_refusal(SimpleNamespace(model="Qwen3.8-27B"), state) is None

    rifiuto = _model_refusal(SimpleNamespace(model="gpt-4"), state)
    assert rifiuto is not None and rifiuto.status_code == 404
    corpo = json.loads(bytes(rifiuto.body).decode())
    assert corpo["error"]["code"] == "model_not_found"
    assert "gpt-4" in corpo["error"]["message"]
    assert "Qwen3.8-27B" in corpo["error"]["message"], "the refusal must name what IS served"


def test_a_server_that_cannot_name_itself_refuses_nothing() -> None:
    """Refusing on an unknown served name would turn one missing field into a node that
    answers nobody. Absent is not a mismatch."""
    from types import SimpleNamespace

    from freetoken.server.openai_api import _model_refusal

    cieco = SimpleNamespace(config=SimpleNamespace(served_model_name=None, model_path=None))
    assert _model_refusal(SimpleNamespace(model="qualunque"), cieco) is None
