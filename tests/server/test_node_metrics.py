"""What a llama.cpp-shaped watcher reads off this engine, and why it is shaped that way.

The BOS device client fetches ``/props`` and ``/metrics`` at the engine's ROOT. Those are
llama-server's endpoints; this engine answered 404 to both, so a node running it reported
heartbeats and never one metric -- 5530 accepted heartbeats against an empty last_metrics_at,
measured on thething the 2026-09-14. The numbers were never missing: /v1/stats had them all.
"""

from __future__ import annotations

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
    return SimpleNamespace(config=SimpleNamespace(max_running_requests=slots,
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
