"""The draft-graph switches default off and parse like the other speculative flags.

Both are read once at import through the FREETOKEN_ prefix, so a typo in the name or a
default that flips to True would enable the graph (or the per-block shadow sync) in every
production run without any log line saying so. Host-side only.
"""

from __future__ import annotations

import pytest

from freetoken.env import ENV, EnvBool, EnvVar

FLAGS = ["SPEC_DRAFT_GRAPH", "SPEC_DRAFT_GRAPH_SHADOW"]


@pytest.mark.parametrize("name", FLAGS)
def test_draft_graph_flags_default_off(name):
    flag = getattr(ENV, name)
    assert isinstance(flag, EnvVar)
    assert flag.fn is ENV.SPEC_VERIFY_GRAPH.fn
    assert not flag
    assert flag.value is False


@pytest.mark.parametrize("name", FLAGS)
@pytest.mark.parametrize(
    "raw, expected",
    [("1", True), ("true", True), ("YES", True), ("0", False), ("false", False), ("", False)],
)
def test_draft_graph_flags_parse_like_the_verify_graph_flag(monkeypatch, name, raw, expected):
    monkeypatch.setenv(f"FREETOKEN_{name}", raw)
    flag = EnvBool(False)
    flag._init(f"FREETOKEN_{name}")
    assert bool(flag) is expected


@pytest.mark.parametrize("name", FLAGS)
def test_draft_graph_flags_ignore_an_unset_variable(monkeypatch, name):
    monkeypatch.delenv(f"FREETOKEN_{name}", raising=False)
    flag = EnvBool(False)
    flag._init(f"FREETOKEN_{name}")
    assert not flag


def test_shadow_is_independent_of_the_graph_flag(monkeypatch):
    # The shadow switch is only consulted once a graph exists; parsing it must not depend on
    # SPEC_DRAFT_GRAPH, so a shadow-only environment stays a plain eager run at the flag level.
    monkeypatch.setenv("FREETOKEN_SPEC_DRAFT_GRAPH_SHADOW", "1")
    monkeypatch.delenv("FREETOKEN_SPEC_DRAFT_GRAPH", raising=False)
    graph, shadow = EnvBool(False), EnvBool(False)
    graph._init("FREETOKEN_SPEC_DRAFT_GRAPH")
    shadow._init("FREETOKEN_SPEC_DRAFT_GRAPH_SHADOW")
    assert not graph and shadow
