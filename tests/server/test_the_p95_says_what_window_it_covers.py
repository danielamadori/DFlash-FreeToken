"""A p95 without its window cannot be read, and this one's window is not what anyone guesses.

MEASURED ON THETHING, 2026-09-28. /v1/stats reported p95_ms = 541.9 s while a real request
answered in 55 ms -- a factor of ten thousand -- because 27 of the 73 requests in the ring were
one agent's measurement campaign and twelve of those ran over 300 seconds. The number was not
wrong; it answered a different question from the one the reader was asking.

And the window is worse than either obvious guess. It is not «since start» and not «the last
five minutes»: it is THE LAST 512 REQUESTS, so how long it covers depends entirely on the
traffic. On a busy engine that is a minute, on a quiet one the whole uptime.

The sharpest part is that the same field, read three days earlier during a real incident, said
18.7 hours and was CORRECT -- the queue truly was the state. Same field, same reading, opposite
validity, and nothing declaring which. So the window now travels beside the number. That does
not resolve the ambiguity, only a reader can; it gives the reader what they need to.
"""
from __future__ import annotations

import datetime
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PY = os.path.join(_ROOT, "python")
if _PY not in sys.path:
    sys.path.insert(0, _PY)

from freetoken.server.request_ring import RequestRecord, RequestRing  # noqa: E402
from freetoken.server.stats import build_stats  # noqa: E402


def _record(secondi_fa: int, durata_ms: int) -> RequestRecord:
    quando = datetime.datetime(2026, 9, 28, 12, 0, 0) - datetime.timedelta(seconds=secondi_fa)
    return RequestRecord(ts=quando.isoformat(), method="POST", path="/v1/chat/completions",
                         status=200, model="m", duration_ms=durata_ms, ttft_ms=None,
                         prompt_tokens=None, completion_tokens=None, stream=False, error=None)


def test_la_finestra_dice_QUANTE_richieste_e_su_quanti_secondi() -> None:
    anello = RequestRing(capacity=512)
    for i in range(10):
        anello.add(_record(secondi_fa=100 - i * 10, durata_ms=50))
    quante, secondi = anello.p95_window()
    assert quante == 10
    assert secondi == 90          # dal piu' vecchio al piu' recente


def test_un_anello_vuoto_dichiara_zero_e_non_finge() -> None:
    assert RequestRing(capacity=8).p95_window() == (0, 0)


def test_una_sola_richiesta_da_una_finestra_di_zero_secondi() -> None:
    """Un solo punto non ha ampiezza: dirlo e' meglio che dire un numero qualsiasi."""
    anello = RequestRing(capacity=8)
    anello.add(_record(secondi_fa=0, durata_ms=42))
    assert anello.p95_window() == (1, 0)


def test_la_finestra_SEGUE_lo_sfratto_del_ring() -> None:
    """Il punto per cui la finestra non e' «dall'avvio»: il ring tiene 512 record e i piu'
    vecchi escono, quindi la finestra si accorcia da sola mentre il traffico cresce."""
    anello = RequestRing(capacity=3)
    for i in range(6):
        anello.add(_record(secondi_fa=100 - i * 10, durata_ms=50))
    quante, secondi = anello.p95_window()
    assert quante == 3            # gli altri tre sono stati sfrattati
    assert secondi == 20          # e la finestra e' quella dei tre rimasti


def test_il_documento_di_stato_porta_la_finestra_ACCANTO_al_numero() -> None:
    """La forma che rende leggibile il campo: chi legge 541900 ms deve poter vedere subito che
    copre 73 richieste su 17 ore, e non cinque minuti.

    Verificato sulla FIRMA e sul corpo di build_stats invece che costruendo uno stato finto
    completo: quella funzione legge la configurazione del modello vera, e un doppio che la
    imitasse tutta verificherebbe soprattutto il doppio. Il punto qui e' che i due campi
    esistano accanto a p95_ms e vengano dal parametro, non da un calcolo interno.
    """
    import inspect

    from freetoken.server import stats

    firma = inspect.signature(stats.build_stats).parameters
    assert "p95_window" in firma
    assert firma["p95_window"].default == (0, 0), "il valore di ripiego non deve mentire"

    fonte = inspect.getsource(stats.build_stats)
    assert '"p95_window_requests": p95_window[0]' in fonte
    assert '"p95_window_seconds": p95_window[1]' in fonte
    # e stanno ACCANTO al numero che qualificano, non in un blocco separato
    assert fonte.index('"p95_ms"') < fonte.index('"p95_window_requests"') < fonte.index('"ttft_mean_ms"')


def test_i_due_chiamanti_passano_la_finestra_e_non_il_ripiego() -> None:
    """Il ripiego (0, 0) esiste per non rompere un chiamante vecchio, ma i due veri devono
    passare quella misurata: altrimenti il campo direbbe «zero richieste» per sempre e sarebbe
    peggio della sua assenza."""
    import inspect

    from freetoken.server import api_server, control_api

    for modulo in (api_server, control_api):
        fonte = inspect.getsource(modulo)
        assert "request_ring.requests_p95_window()" in fonte, modulo.__name__
