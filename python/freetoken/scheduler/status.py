from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from freetoken.core import Batch


@dataclass
class SchedulerStatusReporter:
    log: Callable[[str], None]
    clock: Callable[[], float] = time.perf_counter
    decode_log_interval: int = 40
    _last_prefill_time: float = field(init=False)
    _last_decode_time: float = field(init=False)
    _decode_forward_count: int = field(default=0, init=False)
    _decode_generated_tokens: int = field(default=0, init=False)
    # Contatori della speculazione. Non esistevano, e senza non si puo' dire se un cambiamento
    # alla speculazione migliori o peggiori: llama.cpp riporta "draft acceptance = 0.375
    # (90 accepted / 240 generated), mean len = 3.43" per slot, noi non riportavamo niente e il
    # confronto del 2026-09-10 ha dovuto DEDURRE il nostro valore dal throughput -- ottenendo un
    # numero che non tornava con l'aritmetica della banda.
    _spec_drafted: int = field(default=0, init=False)     # candidati proposti
    _spec_accepted: int = field(default=0, init=False)    # candidati sopravvissuti alla verifica
    _spec_blocks: int = field(default=0, init=False)      # blocchi verificati
    _spec_drafted_tot: int = field(default=0, init=False)  # cumulativi, per la vita del processo
    _spec_accepted_tot: int = field(default=0, init=False)
    _spec_blocks_tot: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        now = self.clock()
        self._last_prefill_time = now
        self._last_decode_time = now
        self.decode_log_interval = max(1, self.decode_log_interval)

    def record_speculation(self, drafted: int, accepted: int) -> None:
        """Un blocco verificato: quanti candidati proposti e quanti accettati.

        Il blocco commette ``accepted + 1`` token -- gli accettati piu' il token bonus che il
        target produce comunque -- ed e' quel +1 che rende la lunghezza media confrontabile con
        il "mean len" di llama.cpp.
        """
        self._spec_drafted += drafted
        self._spec_accepted += accepted
        self._spec_blocks += 1
        self._spec_drafted_tot += drafted
        self._spec_accepted_tot += accepted
        self._spec_blocks_tot += 1

    def speculation_totals(self) -> dict:
        """I cumulativi, per chi li vuole leggere da fuori invece che dal log."""
        b = self._spec_blocks_tot
        return {
            "drafted": self._spec_drafted_tot,
            "accepted": self._spec_accepted_tot,
            "blocks": b,
            "acceptance": self._spec_accepted_tot / self._spec_drafted_tot if self._spec_drafted_tot else 0.0,
            "mean_len": (self._spec_accepted_tot + b) / b if b else 0.0,
        }

    def report_batch(
        self,
        batch: Batch,
        *,
        running_reqs: int,
        queue_reqs: int,
        kv_used_pages: int,
        kv_total_pages: int,
        page_size: int,
        mamba_slots: tuple[int, int] | None = None,
        swa_tokens: tuple[int, int] | None = None,
    ) -> None:
        if batch.is_prefill:
            self._report_prefill(
                batch,
                running_reqs=running_reqs,
                queue_reqs=queue_reqs,
                kv_used_pages=kv_used_pages,
                kv_total_pages=kv_total_pages,
                mamba_slots=mamba_slots,
                swa_tokens=swa_tokens,
            )
        elif batch.is_decode:
            self._report_decode(
                batch,
                running_reqs=running_reqs,
                queue_reqs=queue_reqs,
                kv_used_pages=kv_used_pages,
                kv_total_pages=kv_total_pages,
                page_size=page_size,
                mamba_slots=mamba_slots,
                swa_tokens=swa_tokens,
            )

    def _report_prefill(
        self,
        batch: Batch,
        *,
        running_reqs: int,
        queue_reqs: int,
        kv_used_pages: int,
        kv_total_pages: int,
        mamba_slots: tuple[int, int] | None = None,
        swa_tokens: tuple[int, int] | None = None,
    ) -> None:
        now = self.clock()
        gap = now - self._last_prefill_time
        self._last_prefill_time = now
        # Read the schedule-time snapshot: by report time the forward's complete_one() has
        # advanced each req's cached_len to device_len, so reading the reqs here would log
        # decode-state values (#new-token == #reqs, #cached-token == full prompt).
        new_tokens = batch.log_new_tokens
        cached_tokens = batch.log_cached_tokens
        input_throughput = new_tokens / gap if gap > 0 else 0.0
        self.log(
            f"Prefill batch, "
            f"#new-seq: {len(batch.reqs)}, "
            f"#new-token: {new_tokens}, "
            f"#cached-token: {cached_tokens}, "
            f"token usage: {_usage_ratio(kv_used_pages, kv_total_pages):.2f}, "
            f"{_swa_msg(swa_tokens)}"
            f"{_mamba_msg(mamba_slots)}"
            f"#running-req: {running_reqs}, "
            f"#queue-req: {queue_reqs}, "
            f"input throughput (token/s): {input_throughput:.2f}"
        )

    def _report_decode(
        self,
        batch: Batch,
        *,
        running_reqs: int,
        queue_reqs: int,
        kv_used_pages: int,
        kv_total_pages: int,
        page_size: int,
        mamba_slots: tuple[int, int] | None = None,
        swa_tokens: tuple[int, int] | None = None,
    ) -> None:
        self._decode_forward_count += 1
        self._decode_generated_tokens += len(batch.reqs)
        if self._decode_forward_count % self.decode_log_interval != 0:
            return

        now = self.clock()
        gap = now - self._last_decode_time
        self._last_decode_time = now
        gen_throughput = self._decode_generated_tokens / gap if gap > 0 else 0.0
        self._decode_generated_tokens = 0
        self.log(
            f"Decode batch, "
            f"#running-req: {running_reqs}, "
            f"#token: {kv_used_pages * page_size}, "
            f"token usage: {_usage_ratio(kv_used_pages, kv_total_pages):.2f}, "
            f"{_swa_msg(swa_tokens)}"
            f"{_mamba_msg(mamba_slots)}"
            f"gen throughput (token/s): {gen_throughput:.2f}, "
            f"#queue-req: {queue_reqs}"
            f"{self._spec_msg()}"
        )

    def _spec_msg(self) -> str:
        """L'accettazione dall'ultima riga, azzerata dopo. Stessa forma di llama.cpp, cosi' i
        due motori si leggono con lo stesso metro invece che a occhio."""
        if not self._spec_blocks:
            return ""
        acc = self._spec_accepted / self._spec_drafted if self._spec_drafted else 0.0
        # +1: ogni blocco commette anche il token bonus, come nel conteggio di llama.cpp
        mean_len = (self._spec_accepted + self._spec_blocks) / self._spec_blocks
        msg = (f", draft acceptance: {acc:.3f} ({self._spec_accepted} accepted / "
               f"{self._spec_drafted} drafted), mean len: {mean_len:.2f}")
        self._spec_drafted = self._spec_accepted = self._spec_blocks = 0
        return msg


def _usage_ratio(used: int, total: int) -> float:
    return used / total if total > 0 else 0.0


def _mamba_msg(mamba_slots: tuple[int, int] | None) -> str:
    """GDN-state (mamba) pool occupancy for hybrid models; empty for the rest."""
    if mamba_slots is None:
        return ""
    used, total = mamba_slots
    return f"#mamba-slot: {used}/{total}, mamba usage: {_usage_ratio(used, total):.2f}, "


def _swa_msg(swa_tokens: tuple[int, int] | None) -> str:
    """Window (swa) pool occupancy for SWA models; empty for the rest."""
    if swa_tokens is None:
        return ""
    used, total = swa_tokens
    return f"#swa-token: {used}/{total}, swa usage: {_usage_ratio(used, total):.2f}, "
