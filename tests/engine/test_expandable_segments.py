"""The allocator setting must REPORT what happened, not what was asked for.

Why this file exists. `_ensure_expandable_segments()` called
`torch.cuda.memory._set_allocator_settings("expandable_segments:True")` and then logged
"Enabled expandable_segments". But that call is a PARSER: it accepts the string and makes
no promise that the allocator will honour it. Measured on the Windows node of the cluster,
two consecutive lines of one startup log:

    INFO  Enabled expandable_segments (override via PYTORCH_ALLOC_CONF)
    UserWarning: expandable_segments not supported on this platform
                                               (c10/cuda/CUDAAllocatorConfig.h)

So every Windows startup asserted the opposite of what torch had just said, on the line
about the memory allocator. Someone chasing a memory failure on such a node would read
that the segments were on while they were off -- and that log line would be the most
trustworthy-looking thing they had.

The refusal arrives as a WARNING, not as an exception, which is exactly why the `except`
that was already there never caught it. These tests pin both halves: the refusal is read,
and the warnings that have nothing to do with it are handed back instead of swallowed.
"""
from __future__ import annotations

import warnings


class _Avviso:
    """Un avviso catturato, nella forma che `warnings.catch_warnings(record=True)` da'."""

    def __init__(self, messaggio: str, categoria=UserWarning):
        self.message = categoria(messaggio)
        self.category = categoria
        self.filename = "prova.py"
        self.lineno = 1


#: Il testo esatto misurato sul nodo Windows. Non parafrasato: se torch cambia la
#: formulazione questo test diventa rosso, ed e' giusto -- vuol dire che il
#: riconoscimento va rifatto sulla nuova, non che il problema e' sparito.
RIFIUTO_VERO = (
    "expandable_segments not supported on this platform "
    "(function operator ()) (c10/cuda/CUDAAllocatorConfig.h)"
)


def test_il_rifiuto_della_piattaforma_viene_letto():
    from freetoken.engine.engine import _rifiuto_degli_expandable_segments

    assert _rifiuto_degli_expandable_segments([_Avviso(RIFIUTO_VERO)]) == RIFIUTO_VERO


def test_il_rifiuto_si_riconosce_anche_in_mezzo_ad_altri_avvisi():
    from freetoken.engine.engine import _rifiuto_degli_expandable_segments

    avvisi = [
        _Avviso("qualcosa di deprecato", DeprecationWarning),
        _Avviso(RIFIUTO_VERO),
        _Avviso("un altro avviso qualunque"),
    ]
    assert _rifiuto_degli_expandable_segments(avvisi) == RIFIUTO_VERO


def test_senza_rifiuto_non_si_inventa_un_rifiuto():
    """Su una piattaforma che li supporta il risultato deve essere None.

    E' la meta' che impedisce la cura opposta: un riconoscimento troppo largo
    (per esempio la sola parola "expandable_segments") direbbe "non attivi" su
    Linux, dove invece lo sono, e spegnerebbe una riga di log vera.
    """
    from freetoken.engine.engine import _rifiuto_degli_expandable_segments

    assert _rifiuto_degli_expandable_segments([]) is None
    assert _rifiuto_degli_expandable_segments([_Avviso("un avviso qualunque")]) is None
    # nomina la cosa ma NON e' un rifiuto
    assert _rifiuto_degli_expandable_segments(
        [_Avviso("expandable_segments enabled for this process")]
    ) is None


def test_gli_avvisi_estranei_non_vengono_ingoiati():
    """Catturare per leggerne uno e poi buttarli tutti e' lo stesso difetto piu' in la'.

    Qui si esercita il giro completo -- cattura, riconoscimento, ri-emissione -- su
    un elenco che contiene sia il rifiuto sia un avviso che non c'entra.
    """
    from freetoken.engine.engine import _rifiuto_degli_expandable_segments

    catturati = [_Avviso(RIFIUTO_VERO), _Avviso("attenzione, cosa non correlata")]
    rifiuto = _rifiuto_degli_expandable_segments(catturati)

    with warnings.catch_warnings(record=True) as riemessi:
        warnings.simplefilter("always")
        for avviso in catturati:
            if str(avviso.message) != rifiuto:
                warnings.warn_explicit(
                    avviso.message, avviso.category, avviso.filename, avviso.lineno
                )

    testi = [str(a.message) for a in riemessi]
    assert testi == ["attenzione, cosa non correlata"], (
        "l'avviso estraneo doveva tornare a chi lo aspettava, e il rifiuto no: "
        "quello lo riporta il logger con la sua spiegazione"
    )


def test_il_log_non_dichiara_attivo_cio_che_la_piattaforma_ha_rifiutato(monkeypatch, caplog):
    """Il giro intero: piattaforma che rifiuta -> il log NON deve dire "Enabled".

    E' questo il test che descrive il difetto invece della cura. Sulla versione
    precedente diventa rosso dicendo che il log annuncia "Enabled expandable_segments"
    mentre l'avviso appena emesso dice il contrario -- che e' esattamente cio' che si
    leggeva nel log del nodo Windows, due righe una sotto l'altra.
    """
    import logging

    from freetoken.engine import engine as motore

    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)

    def _piattaforma_che_rifiuta(_impostazione: str) -> None:
        warnings.warn(RIFIUTO_VERO, UserWarning, stacklevel=2)

    monkeypatch.setattr(
        motore.torch.cuda.memory, "_set_allocator_settings", _piattaforma_che_rifiuta
    )

    with caplog.at_level(logging.INFO):
        motore._ensure_expandable_segments()

    detto = "\n".join(r.getMessage() for r in caplog.records)
    assert "Enabled expandable_segments" not in detto, (
        "il log dichiara attivi gli expandable segments subito dopo che la piattaforma "
        f"li ha rifiutati. Log:\n{detto}"
    )
    assert "NOT enabled" in detto and "refused" in detto, (
        f"il rifiuto non e' stato riportato affatto. Log:\n{detto}"
    )


def test_su_una_piattaforma_che_li_accetta_il_log_lo_dice(monkeypatch, caplog):
    """L'altra meta': dove funzionano, la riga vera deve restare."""
    import logging

    from freetoken.engine import engine as motore

    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.setattr(
        motore.torch.cuda.memory, "_set_allocator_settings", lambda _impostazione: None
    )

    with caplog.at_level(logging.INFO):
        motore._ensure_expandable_segments()

    detto = "\n".join(r.getMessage() for r in caplog.records)
    assert "Enabled expandable_segments" in detto
    assert "NOT enabled" not in detto


def test_la_riemissione_non_si_rialimenta(monkeypatch):
    """Re-emitting inside the recording block never ends, and it cost a production node.

    What happened on 2026-09-24. The first version of this function re-emitted the
    unrelated warnings from INSIDE ``with warnings.catch_warnings(record=True) as avvisi``
    and iterated ``avvisi`` itself. Every re-emitted warning was recorded straight back
    into that same list, so the loop grew its own iterable by one on each pass and never
    terminated -- allocating as it went. Measured on the engine's loading child: **2 GiB
    every 6 seconds, without ever stopping**, VRAM flat at 1.3 GiB the whole time, until
    the machine ran out and the OOM killer took the engine. Five separate hypotheses were
    chased before ``py-spy dump`` put the process exactly on that line.

    The test runs the function in a thread and fails if it has not returned in five
    seconds: a hang is the failure being pinned, so it has to be bounded rather than
    asserted on a value. It also checks the honest half -- that the unrelated warning
    still reaches the caller exactly ONCE, because "never re-emit" would pass the timeout
    and silently swallow what the caller was waiting for.
    """
    import threading

    from freetoken.engine import engine as motore

    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)

    def _emette_un_avviso_estraneo(_impostazione: str) -> None:
        warnings.warn("una cosa che non c'entra", UserWarning, stacklevel=2)

    monkeypatch.setattr(
        motore.torch.cuda.memory, "_set_allocator_settings", _emette_un_avviso_estraneo
    )

    fuori: list = []

    def _gira() -> None:
        with warnings.catch_warnings(record=True) as riemessi:
            warnings.simplefilter("always")
            motore._ensure_expandable_segments()
            fuori.extend(str(a.message) for a in riemessi)

    filo = threading.Thread(target=_gira, daemon=True)
    filo.start()
    filo.join(timeout=5.0)

    assert not filo.is_alive(), (
        "_ensure_expandable_segments non e' tornata in 5 secondi: la ri-emissione sta "
        "dentro il blocco che cattura, quindi il ciclo si rialimenta e non termina"
    )
    assert fuori == ["una cosa che non c'entra"], (
        f"l'avviso estraneo doveva tornare al chiamante una volta sola, e' tornato {fuori}"
    )
