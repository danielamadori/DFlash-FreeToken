"""Tie the backend workers' lifetime to this process, at the level the OS enforces.

WHAT THE PYTHON-LEVEL TEARDOWN CANNOT COVER. `_teardown_workers_on_error`,
`shutdown()` and the signal handler all run CODE, so they cover an exception
escaping `run_api_server`, an ordinary stop and SIGTERM/SIGHUP. None of them
runs when the parent is killed without warning -- `Stop-Process -Force` on
Windows (which is `TerminateProcess`), an OOM kill, or a machine that loses
power mid-load. The workers then survive holding their share of VRAM.

MEASURED ON THE DELL, twice. On 2026-09-25 seven orphaned `multiprocessing.spawn`
workers held 10,9 GB of commit between them, one alive for 14 h 22 m. On
2026-09-27, after a failed start, three more, one holding 3982 MiB of a 6 GB
card. The VRAM is not the worst part: the NEXT attempt then fails with «Not
enough memory for KV cache», which is a different failure from the real one, so
the diagnosis goes somewhere else entirely. A false fault that replaces the true
one costs more than the memory.

THE JOB OBJECT IS THE ANSWER BECAUSE IT IS NOT CODE. A job with
`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` kills everything in it when its last handle
closes, and the kernel closes handles of a process however it dies -- including
`TerminateProcess`, where no user code runs at all.

THIS PROCESS JOINS THE JOB, NOT EACH CHILD, and that is the point rather than a
shortcut: on Windows a process created by a process already in a job belongs to
that job automatically. Adopting each worker after `start()` would leave a
window -- small, but exactly the window a crash during startup falls into, which
is when we saw the orphans both times. Joining once, before the first spawn,
leaves none.

WHAT IT STILL DOES NOT COVER, written here so the guard is not believed beyond
its limit: a worker that breaks away with `CREATE_BREAKAWAY_FROM_JOB`, and
anything on a platform that is not Windows, where the caller wants
`PR_SET_PDEATHSIG` instead. Nothing here pretends to be that.
"""
from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)

#: Kept alive for the life of the process ON PURPOSE. The job dies with its last
#: handle, so letting this be collected would kill the workers immediately --
#: the guard would become the fault it exists to prevent.
_JOB_HANDLE: object | None = None

_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9


def install_kill_on_close_job() -> bool:
    """Put this process in a kill-on-close job. True when the net is up.

    Idempotent: a second call with the job already installed returns True
    without creating a second one.

    NEVER RAISES, and says what is missing when it fails. The engine runs
    perfectly well without this -- it is a net under a crash, not a dependency
    -- so refusing to start would trade a rare orphan for a certain outage. But
    a net that quietly is not there is worse than no net, hence the warning
    names the call that failed and what is consequently not covered.
    """
    global _JOB_HANDLE

    if not sys.platform.startswith("win"):
        return False
    if _JOB_HANDLE is not None:
        return True

    import ctypes
    from ctypes import wintypes

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class _BASIC_LIMIT(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _EXTENDED_LIMIT(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _BASIC_LIMIT),
            ("IoInfo", _IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    # ARGTYPES ON EVERY CALL, not just restype. Without them ctypes marshals a
    # HANDLE as a signed int, and `GetCurrentProcess()` returns the pseudo-handle
    # (HANDLE)-1: the first version of this raised
    # `ctypes.ArgumentError: argument 2: OverflowError: int too long to convert`
    # before reaching the API at all -- a defect in the binding, wearing the
    # shape of a Windows refusal.
    k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.GetCurrentProcess.argtypes = []
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
    ]
    k32.SetInformationJobObject.restype = wintypes.BOOL
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    k32.AssignProcessToJobObject.restype = wintypes.BOOL
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    k32.CloseHandle.restype = wintypes.BOOL

    def _perche() -> str:
        return f"GetLastError={ctypes.get_last_error()}"

    job = k32.CreateJobObjectW(None, None)
    if not job:
        logger.warning(
            "windows job: CreateJobObject failed (%s); backend workers will NOT "
            "be killed if this process is terminated without running code",
            _perche(),
        )
        return False

    info = _EXTENDED_LIMIT()
    info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not k32.SetInformationJobObject(
        job, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
        ctypes.byref(info), ctypes.sizeof(info),
    ):
        logger.warning(
            "windows job: SetInformationJobObject(KILL_ON_JOB_CLOSE) failed (%s); "
            "the job exists but would NOT kill its workers, so it is dropped",
            _perche(),
        )
        k32.CloseHandle(job)
        return False

    if not k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()):
        # A parent already inside a job whose nesting is refused lands here.
        # Nested jobs work from Windows 8, so this is rare -- and it is a
        # warning and not a failure because the engine is fine without the net.
        logger.warning(
            "windows job: AssignProcessToJobObject failed (%s); backend workers "
            "will NOT be killed if this process is terminated without running "
            "code (this parent may already belong to a job that refuses nesting)",
            _perche(),
        )
        k32.CloseHandle(job)
        return False

    _JOB_HANDLE = job
    logger.info(
        "windows job: backend workers are tied to this process and die with it, "
        "including when it is terminated without running any code",
    )
    return True
