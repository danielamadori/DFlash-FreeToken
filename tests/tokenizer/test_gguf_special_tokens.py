"""A GGUF's CONTROL/USER_DEFINED tokens must arrive whole, not as text pieces.

Measured on thething, 2026-09-14, Qwen3.8-27B-UD-Q4_K_S: the converter registered 4
special tokens out of the 33 the GGUF marks, so `<think>` -- which the qwen3 chat
template appends to EVERY prompt -- was tokenized `'<th'`, `'ink'`, `'>'`. The model
then sat off-distribution at the exact position where it chooses what to emit, and
answered 9 short instructions in 10 by closing the turn straight after `</think>` with
no text. llama.cpp on the same file answered every time.

These tests use the metadata shape, not a real GGUF: what is being checked is the rule
(token_type 3 and 4 get registered, ids never move), which a fixture can state exactly.
"""
from __future__ import annotations

import pytest

from freetoken.models.gguf.tokenizer import _register_gguf_special_tokens


class _Tokenizer:
    """Enough of a HF fast tokenizer to exercise the rule: it knows which names it has
    already registered, and it grows only when asked for a name it does not hold."""

    def __init__(self, vocabolario: dict[str, int], registrati: dict[str, int]) -> None:
        self._vocabolario = dict(vocabolario)
        self._registrati = dict(registrati)
        self.aggiunti: list[str] = []

    def get_added_vocab(self) -> dict[str, int]:
        return dict(self._registrati)

    def __len__(self) -> int:
        return len(self._vocabolario)

    def add_tokens(self, tokens: list, special_tokens: bool = False) -> int:
        for t in tokens:
            nome = str(t)
            self.aggiunti.append(nome)
            if nome in self._vocabolario:
                self._registrati[nome] = self._vocabolario[nome]
            else:
                self._vocabolario[nome] = len(self._vocabolario)
        return len(tokens)


_VOCABOLARIO = {"ciao": 0, "<|im_start|>": 1, "<|im_end|>": 2, "<think>": 3, "</think>": 4}
_TIPI = {"ciao": 1, "<|im_start|>": 3, "<|im_end|>": 3, "<think>": 4, "</think>": 4}


def _dizionario(vocabolario: dict[str, int] = _VOCABOLARIO) -> dict[str, object]:
    nomi = sorted(vocabolario, key=vocabolario.get)
    return {"tokens": nomi, "token_type": [_TIPI[n] for n in nomi]}


def test_control_and_user_defined_tokens_are_registered() -> None:
    """Type 3 is CONTROL and type 4 is USER_DEFINED; the model saw both as one unit."""
    tok = _Tokenizer(_VOCABOLARIO, registrati={"<|im_start|>": 1, "<|im_end|>": 2})

    _register_gguf_special_tokens(tok, _dizionario())

    assert sorted(tok.aggiunti) == ["</think>", "<think>"]
    assert "<think>" in tok.get_added_vocab()


def test_ordinary_tokens_are_left_alone() -> None:
    """Registering the whole vocabulary would make every word a special token."""
    tok = _Tokenizer(_VOCABOLARIO, registrati={})

    _register_gguf_special_tokens(tok, _dizionario())

    assert "ciao" not in tok.aggiunti


def test_the_vocabulary_never_grows() -> None:
    """The ids exist already: this names them. A vocabulary that grows means the
    conversion and the metadata disagree, and every later id would shift under the
    embedding table -- so it must raise, not proceed."""
    mancante = dict(_VOCABOLARIO)
    del mancante["<think>"]
    tok = _Tokenizer(mancante, registrati={})

    with pytest.raises(ValueError, match="grew the vocabulary"):
        _register_gguf_special_tokens(tok, _dizionario())


def test_metadata_without_token_type_is_a_no_op() -> None:
    """Not every GGUF carries token_type; one that does not must still load."""
    tok = _Tokenizer(_VOCABOLARIO, registrati={})

    _register_gguf_special_tokens(tok, {"tokens": sorted(_VOCABOLARIO, key=_VOCABOLARIO.get)})

    assert tok.aggiunti == []


def test_a_truncated_token_type_list_is_not_trusted() -> None:
    """Zipping a short type list against the vocabulary would mark the wrong tokens."""
    d = _dizionario()
    d["token_type"] = d["token_type"][:2]
    tok = _Tokenizer(_VOCABOLARIO, registrati={})

    _register_gguf_special_tokens(tok, d)

    assert tok.aggiunti == []
