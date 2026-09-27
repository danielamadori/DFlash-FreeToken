"""A failed start must not leave the spawned workers holding VRAM.

MEASURED ON THE DELL, twice. After a failed FreeToken start, three multiprocessing.spawn
workers with a dead parent, one holding 3982 MiB of a 6001 MiB card -- enough to make the
NEXT attempt fail with "Not enough memory for KV cache", a false fault standing in for the
real one that sends the diagnosis somewhere else. An earlier attempt left 10.9 GB.

Until this guard existed none of the three teardown paths ran in that case: shutdown() is
the orderly stop, the SIGTERM/SIGHUP handler needs a signal, and the shell's finally needs
the shell. A traceback out of run_api_server ran none of them.

What it does NOT cover is asserted too, because a guard whose limit is not written down gets
trusted past it: SIGKILL of the parent and an OOM kill run no code at all.
"""
from __future__ import annotations

import os
import sys
from typing import Any

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PY = os.path.join(_ROOT, "python")
if _PY not in sys.path:
    sys.path.insert(0, _PY)

from freetoken.server import api_server  # noqa: E402


class FakeWorker:
    def __init__(self) -> None:
        self.terminated = False
        self.killed = False

    def is_alive(self) -> bool:
        return not self.killed

    def terminate(self) -> None:
        self.terminated = True

    def join(self, timeout: float | None = None) -> None:
        return None

    def kill(self) -> None:
        self.killed = True


@pytest.fixture()
def tre_worker(monkeypatch: pytest.MonkeyPatch) -> list[FakeWorker]:
    """Three workers on the global state, as the Dell found them."""
    operai = [FakeWorker(), FakeWorker(), FakeWorker()]
    stato = type("S", (), {"backend_processes": operai})()
    monkeypatch.setattr(api_server, "_GLOBAL_STATE", stato)
    api_server._SHUTTING_DOWN.clear()
    yield operai
    api_server._SHUTTING_DOWN.clear()


def _corri_e_falli(monkeypatch: pytest.MonkeyPatch, errore: BaseException) -> None:
    """The tail of run_api_server, reproduced: uvicorn raises where it would really raise."""
    def boom(*_a: Any, **_k: Any) -> None:
        raise errore

    monkeypatch.setattr(api_server.uvicorn, "run", boom)
    with pytest.raises(type(errore)):
        try:
            api_server.uvicorn.run(None, host="127.0.0.1", port=1)
        except BaseException:
            if not api_server._SHUTTING_DOWN.is_set():
                api_server._SHUTTING_DOWN.set()
                api_server._terminate_backend_workers(
                    api_server._GLOBAL_STATE.backend_processes)
                api_server._reap_backend_workers(
                    api_server._GLOBAL_STATE.backend_processes)
            raise


def test_un_errore_all_avvio_abbatte_tutti_e_tre_i_worker(
        monkeypatch: pytest.MonkeyPatch, tre_worker: list[FakeWorker]) -> None:
    _corri_e_falli(monkeypatch, RuntimeError("cannot bind"))
    assert all(o.terminated for o in tre_worker)


def test_anche_un_KeyboardInterrupt_durante_il_caricamento(
        monkeypatch: pytest.MonkeyPatch, tre_worker: list[FakeWorker]) -> None:
    """BaseException e non Exception: un ^C mentre i pesi caricano lascia gli stessi orfani."""
    _corri_e_falli(monkeypatch, KeyboardInterrupt())
    assert all(o.terminated for o in tre_worker)


def test_uno_spegnimento_ORDINATO_non_passa_da_qui(
        monkeypatch: pytest.MonkeyPatch, tre_worker: list[FakeWorker]) -> None:
    """La meta' onesta: se _SHUTTING_DOWN e' gia' alzato qualcuno ha gia' abbattuto i worker,
    e rifarlo qui li terminerebbe due volte mascherando chi l'ha fatto per primo."""
    api_server._SHUTTING_DOWN.set()
    _corri_e_falli(monkeypatch, SystemExit(0))
    assert not any(o.terminated for o in tre_worker)


def test_la_guardia_e_nel_sorgente_e_dichiara_cosa_NON_copre() -> None:
    """Una guardia il cui limite non e' scritto viene creduta oltre quel limite. SIGKILL e OOM
    non eseguono nessun codice: chiuderli vuole PR_SET_PDEATHSIG o un Job Object, non questo."""
    import inspect

    fonte = inspect.getsource(api_server.run_api_server)
    assert "except BaseException" in fonte
    assert "SIGKILL" in fonte
    assert "PR_SET_PDEATHSIG" in fonte
