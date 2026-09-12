from __future__ import annotations

from dataclasses import dataclass, field

from freetoken.engine import EngineConfig


def _get_pid_suffix() -> str:
    import os

    return f".pid={os.getpid()}"


def _canale(indice: int, suffisso: str) -> str:
    """One inter-process channel address, in a transport this OS actually has.

    ZeroMQ's ipc:// is a Unix domain socket, and libzmq does not implement it on
    Windows: bind raises `Protocol not supported (addr='ipc:///tmp/freetoken_3...')`
    and the engine dies before it ever serves a request. Loopback TCP is the
    substitute ZeroMQ documents for Windows.

    The port comes from the same pid the ipc path already encodes, so parent and
    workers agree without passing anything extra, and the five channels get five
    consecutive ports. Two runs collide only if their pids are congruent mod
    5000 -- the same class of collision the shared /tmp path already has -- and
    the range stays clear of Windows' dynamic ports, which start at 49152.
    """
    import os

    if os.name != "nt":
        return f"ipc:///tmp/freetoken_{indice}{suffisso}"
    try:
        pid = int(suffisso.rsplit("=", 1)[-1])
    except ValueError:
        pid = os.getpid()
    return f"tcp://127.0.0.1:{20000 + (pid % 5000) * 5 + indice}"


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    max_extend_tokens: int = 8192
    cache_type: str = "radix"
    offline_mode: bool = False
    decode_log_interval: int = 40
    special_token_ckpt: bool = False

    # networking config
    _unique_suffix: str = field(default_factory=_get_pid_suffix)

    @property
    def zmq_backend_addr(self) -> str:
        return _canale(0, self._unique_suffix)

    @property
    def zmq_detokenizer_addr(self) -> str:
        return _canale(1, self._unique_suffix)

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        return _canale(2, self._unique_suffix)

    @property
    def max_forward_len(self) -> int:
        return self.max_extend_tokens

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return True
