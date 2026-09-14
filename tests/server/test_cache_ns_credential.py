"""The prefix-cache namespace rides on the credential, and why it has to.

The chain in front of this engine is client -> proxy -> a compacting middlebox -> here.
Measured on that chain (Agents docs/backlog.md (42)): a marker placed in the request body does
not survive, because compaction rewrites the messages; a custom header does not survive, because
the middlebox builds its own and its extra-headers setting is static rather than per request.
Authorization survives verbatim -- it has to, or the request does not authenticate.

So the namespace is a suffix on the key. These tests pin the parsing, which is the part that
must not be clever: it decides whether a request is authenticated at all.
"""

from __future__ import annotations

from freetoken.server.api_server import _split_cache_ns

CHIAVE = "abc123"


def test_a_plain_key_carries_no_namespace():
    """The shared tree stays the default: a deployment with one tenant changes nothing."""
    assert _split_cache_ns(CHIAVE, CHIAVE) == (CHIAVE, None)


def test_the_suffix_comes_off_and_the_key_still_matches():
    assert _split_cache_ns(f"{CHIAVE}.sessione-7", CHIAVE) == (CHIAVE, "sessione-7")


def test_a_wrong_key_is_left_alone_so_it_still_fails():
    """Nothing here may turn a bad credential into a good one. A wrong key with a dot in it
    must come back unchanged, so the comparison that follows rejects it exactly as before."""
    assert _split_cache_ns("wrong.sessione-7", CHIAVE) == ("wrong.sessione-7", None)
    assert _split_cache_ns(f"{CHIAVE}x.sessione", CHIAVE) == (f"{CHIAVE}x.sessione", None)


def test_a_key_that_contains_dots_is_not_mistaken_for_a_namespace():
    """Only a dot immediately after the whole configured key counts as the separator."""
    punteggiata = "ab.c1.23"
    assert _split_cache_ns(punteggiata, punteggiata) == (punteggiata, None)
    assert _split_cache_ns(f"{punteggiata}.tenant", punteggiata) == (punteggiata, "tenant")


def test_an_empty_namespace_reads_as_none():
    """'key.' must not drop the caller into a nameless partition where everyone meets."""
    assert _split_cache_ns(f"{CHIAVE}.", CHIAVE) == (f"{CHIAVE}.", None)


def test_no_configured_key_means_no_namespace_parsing():
    """With auth off there is nothing to anchor the suffix to, and splitting on a bare dot
    would namespace by whatever the caller typed."""
    assert _split_cache_ns("qualunque.cosa", "") == ("qualunque.cosa", None)
