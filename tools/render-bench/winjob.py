"""Account for the entire Windows job, including venv launchers and workers.

Job peak commit is an OS high-water mark. Aggregate working set is sampled;
the observed sum of per-process peak working sets bounds the sampled aggregate.
Retrieval can miss a process's final high-water mark if it exits between polls.
"""
import ctypes as C
from ctypes import wintypes as W
import subprocess, time
import psutil

class Basic(C.Structure):
    _fields_ = [('PerProcessUserTimeLimit', C.c_int64), ('PerJobUserTimeLimit', C.c_int64),
                ('LimitFlags', W.DWORD), ('MinimumWorkingSetSize', C.c_size_t),
                ('MaximumWorkingSetSize', C.c_size_t), ('ActiveProcessLimit', W.DWORD),
                ('Affinity', C.c_size_t), ('PriorityClass', W.DWORD), ('SchedulingClass', W.DWORD)]

class IO(C.Structure):
    _fields_ = [(n, C.c_uint64) for n in ('ReadOperationCount', 'WriteOperationCount',
                'OtherOperationCount', 'ReadTransferCount', 'WriteTransferCount', 'OtherTransferCount')]

class Extended(C.Structure):
    _fields_ = [('BasicLimitInformation', Basic), ('IoInfo', IO)] + [
        (n, C.c_size_t) for n in ('ProcessMemoryLimit', 'JobMemoryLimit',
                                'PeakProcessMemoryUsed', 'PeakJobMemoryUsed')]

class Pids(C.Structure):
    _fields_ = [('assigned', W.DWORD), ('count', W.DWORD), ('ids', C.c_size_t * 128)]

K = C.WinDLL('kernel32', use_last_error=True)
for name, ret, args in [
    ('CreateJobObjectW', W.HANDLE, [C.c_void_p, W.LPCWSTR]),
    ('SetInformationJobObject', W.BOOL, [W.HANDLE, C.c_int, C.c_void_p, W.DWORD]),
    ('AssignProcessToJobObject', W.BOOL, [W.HANDLE, W.HANDLE]),
    ('QueryInformationJobObject', W.BOOL, [W.HANDLE, C.c_int, C.c_void_p, W.DWORD, C.c_void_p]),
    ('TerminateJobObject', W.BOOL, [W.HANDLE, W.UINT]),
    ('CloseHandle', W.BOOL, [W.HANDLE])]:
    fn = getattr(K, name); fn.restype = ret; fn.argtypes = args

def check(value):
    if not value: raise C.WinError(C.get_last_error())
    return value

def measured(args, env, stdout, stderr, cwd, timeout=3600, cores=range(12)):
    job = check(K.CreateJobObjectW(None, None))
    info = Extended()
    info.BasicLimitInformation.LimitFlags = 0x2000 | 0x10  # kill-on-close + affinity
    info.BasicLimitInformation.Affinity = sum(1 << c for c in cores)
    proc = None
    try:
        check(K.SetInformationJobObject(job, 9, C.byref(info), C.sizeof(info)))
        # Assignment before the first instruction prevents losing short-lived children.
        proc = subprocess.Popen(args, env=env, cwd=cwd, stdout=stdout, stderr=stderr,
                                creationflags=0x4 | 0x08000000)
        check(K.AssignProcessToJobObject(job, int(proc._handle)))
        psutil.Process(proc.pid).resume()
        started = time.perf_counter(); peaks = {}; names = {}; peak_rss = 0; peak_private = 0
        samples = 0; max_processes = 0; intervals = []; last = started; timed_out = False
        while True:
            ids = Pids()
            check(K.QueryInformationJobObject(job, 3, C.byref(ids), C.sizeof(ids), None))
            rss = private = 0
            for pid in list(ids.ids)[:ids.count]:
                try:
                    process = psutil.Process(pid)
                    m = process.memory_info()
                    if pid not in names: names[pid] = process.name()
                    rss += m.rss; private += m.private
                    peaks[pid] = max(peaks.get(pid, 0), m.peak_wset)
                except (psutil.NoSuchProcess, psutil.AccessDenied): pass
            now = time.perf_counter(); intervals.append(now-last); last=now
            samples += 1; peak_rss = max(peak_rss, rss); peak_private = max(peak_private, private)
            max_processes = max(max_processes, ids.count)
            if proc.poll() is not None and ids.count == 0: break
            if now-started > timeout:
                check(K.TerminateJobObject(job, 124)); timed_out=True
            time.sleep(.01)
        proc.wait()
        check(K.QueryInformationJobObject(job, 9, C.byref(info), C.sizeof(info), None))
        return dict(returncode=proc.returncode, timed_out=timed_out,
                    process_seconds=time.perf_counter()-started,
                    peak_job_commit_bytes=info.PeakJobMemoryUsed,
                    peak_process_commit_bytes=info.PeakProcessMemoryUsed,
                    peak_tree_rss_sampled_bytes=peak_rss,
                    peak_tree_private_sampled_bytes=peak_private,
                    sum_process_peak_wset_bytes=sum(peaks.values()),
                    process_peak_wset_bytes=peaks, process_names=names, max_processes=max_processes,
                    samples=samples, max_sample_gap_ms=max(intervals)*1000)
    finally:
        # Also cover assignment/resume failures before job ownership is established.
        if proc is not None and proc.poll() is None:
            proc.kill()
        K.CloseHandle(job)
