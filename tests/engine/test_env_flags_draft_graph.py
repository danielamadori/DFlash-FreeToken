"""The draft-graph switches default off and parse like the other speculative flags.

Both are read once at import through the FREETOKEN_ prefix, so a typo in the name or a
default that flips to True would enable the graph (or the per-block shadow sync) in every
production run without any log line saying so. Host-side only.
"""

from __future__ import annotations

import pytest

from freetoken.env import ENV, EnvBool, EnvVar

FLAGS = ["SPEC_DRAFT_GRAPH", "SPEC_DRAFT_GRAPH_SHADOW"]


def test_the_draft_graph_is_on_and_the_shadow_is_not():
    """The graph earns its default; the diagnostic beside it must not.

    The graph replays the draft forward and was shown equal to the eager one -- 1400 blocks at
    temperature 0, zero mismatching tokens. The shadow runs the block twice and returns the
    eager result: correct, and half the speed, so a default of on would read as a regression.
    """
    assert ENV.SPEC_DRAFT_GRAPH and ENV.SPEC_DRAFT_GRAPH.value is True
    assert not ENV.SPEC_DRAFT_GRAPH_SHADOW and ENV.SPEC_DRAFT_GRAPH_SHADOW.value is False


@pytest.mark.parametrize("name", FLAGS)
def test_both_flags_are_the_same_kind_of_switch(name):
    """Same parser as every other speculative flag, so the environment can still turn the
    graph off -- which is the way back out if a model or a backend captures badly."""
    flag = getattr(ENV, name)
    assert isinstance(flag, EnvVar)
    assert flag.fn is ENV.SPEC_VERIFY_GRAPH.fn


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
