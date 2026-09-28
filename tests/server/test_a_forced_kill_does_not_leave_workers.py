"""A parent killed without running code still takes its children with it.

WHAT `test_a_failed_start_does_not_leave_workers` DOES NOT COVER, and says so in
its own function: `_teardown_workers_on_error`, `shutdown()` and the signal
handler all RUN CODE. They cover an exception escaping `run_api_server`, an
ordinary stop, `^C`, SIGTERM and SIGHUP. None of them runs when the parent is
killed outright -- `Stop-Process -Force` on Windows, which is `TerminateProcess`,
or an OOM kill. The workers then survive holding their share of VRAM.

MEASURED TWICE ON THE DELL before this existed. On 2026-09-25, seven orphaned
`multiprocessing.spawn` workers holding 10,9 GB of commit between them, one
alive for 14 h 22 m. On 2026-09-27, after a failed start, three more, one
holding 3982 MiB of a 6 GB card. The memory is not the expensive part: the NEXT
attempt fails with «Not enough memory for KV cache», a different failure from
the real one, and the diagnosis goes somewhere else entirely.

THE CONTROL ARM IS THE POINT OF THIS FILE. Asserting only that the child dies
would pass on a platform where children happen to die for some other reason --
so the same parent runs twice, once WITHOUT joining the job, and that arm must
leave the child alive. Without it, "the child is gone" says nothing about why.

Windows only, and skipped elsewhere rather than faked: the job object is a
Windows mechanism, and `PR_SET_PDEATHSIG` on Linux is a different fix that would
need its own test.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time

import pytest

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("win"),
    reason=(
        "the kill-on-close job object is a Windows mechanism; the equivalent "
        "elsewhere is PR_SET_PDEATHSIG and is not what this file tests"
    ),
)

#: The parent: optionally joins the job, spawns a sleeper, prints both pids and
#: then waits to be killed. Written to a file rather than passed with -c so the
#: import of `freetoken` -- which pulls torch and is not fast -- happens in a
#: module the child interpreter can read normally.
_PADRE = """
import importlib.util, os, subprocess, sys, time
if sys.argv[1] == "con-job":
    # LOADED BY PATH, not as `freetoken.server.windows_job`: importing the
    # package would pull torch, which lives in the engine's venv, and that venv
    # has no pytest -- so an import-based test would only run where neither
    # interpreter can satisfy both halves. The guard module imports nothing but
    # logging, sys and ctypes, and loading it this way PROVES that too.
    spec = importlib.util.spec_from_file_location("windows_job", {modulo!r})
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.install_kill_on_close_job()
figlio = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
print(f"{{os.getpid()}} {{figlio.pid}}", flush=True)
time.sleep(600)
"""


def _vivo(pid: int) -> bool:
    """Whether ``pid`` is still running, asked of the OS and not of a handle."""
    out = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
        capture_output=True, text=True, timeout=30,
    ).stdout
    return str(pid) in out


def _corsa(tmp_path, modo: str) -> bool:
    """Start the parent in ``modo``, kill it by force, return whether the child lived."""
    modulo = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "python", "freetoken", "server", "windows_job.py",
    )
    assert os.path.exists(modulo), f"the guard module is not where expected: {modulo}"
    sorgente = tmp_path / f"padre_{modo}.py"
    sorgente.write_text(textwrap.dedent(_PADRE).format(modulo=modulo), encoding="utf-8")
    uscita = tmp_path / f"out_{modo}.txt"

    errori = tmp_path / f"err_{modo}.txt"
    # STDERR KEPT, never DEVNULL: the first version of this discarded it, so
    # when the parent failed to start the assertion below could say THAT it had
    # not started and never why -- which cost a run to find out it was an
    # interpreter without the package on its path.
    with uscita.open("w", encoding="utf-8") as f, errori.open("w", encoding="utf-8") as e:
        padre = subprocess.Popen([sys.executable, str(sorgente), modo], stdout=f, stderr=e)
    try:
        pids = ""
        for _ in range(60):          # importing freetoken is not fast
            time.sleep(2)
            pids = uscita.read_text(encoding="utf-8").strip()
            if pids:
                break
        assert pids, (
            f"the {modo} parent never printed its pids: it did not start. "
            "Its stderr, last 1200 chars: "
            + errori.read_text(encoding="utf-8")[-1200:]
        )
        figlio = int(pids.split()[1])

        # TerminateProcess: no handler, no finally, no atexit. This is the case
        # the python-level teardown cannot reach, and the reason this file exists.
        subprocess.run(["taskkill", "/PID", str(padre.pid), "/F"],
                       capture_output=True, timeout=30)
        for _ in range(10):
            time.sleep(1)
            if not _vivo(figlio):
                return False
        return _vivo(figlio)
    finally:
        # `locals()` and not `dir()`: inside a function `dir()` happens to
        # return the local names, but it is documented as "the current local
        # scope" only by accident of implementation, and a teardown that raises
        # NameError would replace the real verdict with its own.
        padre.kill()
        _figlio = locals().get("figlio")
        if _figlio is not None:
            subprocess.run(["taskkill", "/PID", str(_figlio), "/F"],
                           capture_output=True, timeout=30)


def test_without_the_job_the_child_survives_the_parent(tmp_path):
    """The control arm: it must FAIL to clean up, or the other test proves nothing."""
    assert _corsa(tmp_path, "senza-job") is True, (
        "the child died even without the job object, so this platform cleans up "
        "on its own and the other test says nothing about the job"
    )


def test_with_the_job_the_child_dies_with_the_parent(tmp_path):
    """The measurement: a forced kill of the parent takes the worker with it."""
    assert _corsa(tmp_path, "con-job") is False, (
        "the child outlived a force-killed parent even inside the job: the "
        "kill-on-close guard is not doing what it exists for"
    )
