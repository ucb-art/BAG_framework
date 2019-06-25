# -*- coding: utf-8 -*-

"""This module define utility classes for performing concurrent operations.
"""

from typing import Optional, Sequence, Dict, Union, Tuple, Callable, Any

import os
import asyncio
# noinspection PyProtectedMember
from asyncio.subprocess import Process
import subprocess
import multiprocessing
from concurrent.futures import CancelledError


def batch_async_task(coro_list):
    """Execute a list of coroutines or futures concurrently.

    User may press Ctrl-C to cancel all given tasks.

    Parameters
    ----------
    coro_list :
        a list of coroutines or futures to run concurrently.

    Returns
    -------
    results :
        a list of return values or raised exceptions of given tasks.
    """
    top_future = asyncio.gather(*coro_list, return_exceptions=True)

    loop = asyncio.get_event_loop()
    try:
        print('Running tasks, Press Ctrl-C to cancel.')
        results = loop.run_until_complete(top_future)
    except KeyboardInterrupt:
        print('Ctrl-C detected, Cancelling tasks.')
        top_future.cancel()
        loop.run_forever()
        results = None

    return results


ProcInfo = Tuple[Union[str, Sequence[str]], str, Optional[Dict[str, str]], Optional[str]]
FlowInfo = Tuple[Union[str, Sequence[str]], str, Optional[Dict[str, str]], Optional[str],
                 Callable[[Optional[int], str], Any]]


class SubProcessManager(object):
    """A class that provides convenient methods to run multiple subprocesses in parallel using asyncio.

    Parameters
    ----------
    max_workers : Optional[int]
        number of maximum allowed subprocesses.  If None, defaults to system
        CPU count.
    cancel_timeout : Optional[float]
        Number of seconds to wait for a process to terminate once SIGTERM or
        SIGKILL is issued.  Defaults to 10 seconds.
    """

    def __init__(self, max_workers=None, cancel_timeout=10.0):
        # type: (Optional[int], Optional[float]) -> None
        if max_workers is None:
            max_workers = multiprocessing.cpu_count()
        if cancel_timeout is None:
            cancel_timeout = 10.0

        self._cancel_timeout = cancel_timeout
        self._semaphore = asyncio.Semaphore(max_workers)

    async def _kill_subprocess(self, proc: Optional[Process]) -> None:
        """Helper method; send SIGTERM/SIGKILL to a subprocess.

        This method first sends SIGTERM to the subprocess.  If the process hasn't terminated
        after a given timeout, it sends SIGKILL.

        Parameter
        ---------
        proc : Optional[Process]
            the process to attempt to terminate.  If None, this method does nothing.
        """
        if proc is not None:
            if proc.returncode is None:
                try:
                    proc.terminate()
                    try:
                        await asyncio.shield(asyncio.wait_for(proc.wait(), self._cancel_timeout))
                    except CancelledError:
                        pass

                    if proc.returncode is None:
                        proc.kill()
                        try:
                            await asyncio.shield(
                                asyncio.wait_for(proc.wait(), self._cancel_timeout))
                        except CancelledError:
                            pass
                except ProcessLookupError:
                    pass

    async def async_new_subprocess(self,
                                   args: Union[str, Sequence[str]],
                                   log: str,
                                   env: Optional[Dict[str, str]] = None,
                                   cwd: Optional[str] = None) -> Optional[int]:
        """A coroutine which starts a subprocess.

        If this coroutine is cancelled, it will shut down the subprocess gracefully using
        SIGTERM/SIGKILL, then raise CancelledError.

        Parameters
        ----------
        args : Union[str, Sequence[str]]
            command to run, as string or sequence of strings.
        log : str
            the log file name.
        env : Optional[Dict[str, str]]
            an optional dictionary of environment variables.  None to inherit from parent.
        cwd : Optional[str]
            the working directory.  None to inherit from parent.

        Returns
        -------
        retcode : Optional[int]
            the return code of the subprocess.
        """
        if isinstance(args, str):
            args = [args]

        # get log file name, make directory if necessary
        log = os.path.abspath(log)
        if os.path.isdir(log):
            raise ValueError('log file %s is a directory.' % log)
        os.makedirs(os.path.dirname(log), exist_ok=True)

        async with self._semaphore:
            proc = None
            with open(log, 'w') as logf:
                logf.write('command: %s\n' % (' '.join(args)))
                logf.flush()
                try:
                    proc = await asyncio.create_subprocess_exec(*args, stdout=logf,
                                                                stderr=subprocess.STDOUT,
                                                                env=env, cwd=cwd)
                    retcode = await proc.wait()
                    return retcode
                except CancelledError as err:
                    await self._kill_subprocess(proc)
                    raise err

    async def async_new_subprocess_flow(self,
                                        proc_info_list: Sequence[FlowInfo]) -> Any:
        """A coroutine which runs a series of subprocesses.

        If this coroutine is cancelled, it will shut down the current subprocess gracefully using
        SIGTERM/SIGKILL, then raise CancelledError.

        Parameters
        ----------
        proc_info_list : Sequence[FlowInfo]
            a list of processes to execute in series.  Each element is a tuple of:

            args : Union[str, Sequence[str]]
                command to run, as string or list of string arguments.
            log : str
                log file name.
            env : Optional[Dict[str, str]]
                environment variable dictionary.  None to inherit from parent.
            cwd : Optional[str]
                working directory path.  None to inherit from parent.
            vfun : Sequence[Callable[[Optional[int], str], Any]]
                a function to validate if it is ok to execute the next process.  The output of the
                last function is returned.  The first argument is the return code, the second
                argument is the log file name.

        Returns
        -------
        result : Any
            the return value of the last validate function.  None if validate function
            returns False.
        """
        num_proc = len(proc_info_list)
        if num_proc == 0:
            return None

        async with self._semaphore:
            for idx, (args, log, env, cwd, vfun) in enumerate(proc_info_list):
                if isinstance(args, str):
                    args = [args]

                # get log file name, make directory if necessary
                log = os.path.abspath(log)
                if os.path.isdir(log):
                    raise ValueError('log file %s is a directory.' % log)
                os.makedirs(os.path.dirname(log), exist_ok=True)

                proc, retcode = None, None
                with open(log, 'w') as logf:
                    logf.write('command: %s\n' % (' '.join(args)))
                    logf.flush()
                    try:
                        proc = await asyncio.create_subprocess_exec(*args, stdout=logf,
                                                                    stderr=subprocess.STDOUT,
                                                                    env=env, cwd=cwd)
                        retcode = await proc.wait()
                    except CancelledError as err:
                        await self._kill_subprocess(proc)
                        raise err

                fun_output = vfun(retcode, log)
                if idx == num_proc - 1:
                    return fun_output
                elif not fun_output:
                    return None

    def batch_subprocess(self, proc_info_list):
        # type: (Sequence[ProcInfo]) -> Optional[Sequence[Union[int, Exception]]]
        """Run all given subprocesses in parallel.

        Parameters
        ----------
        proc_info_list : Sequence[ProcInfo]
            a list of process information.  Each element is a tuple of:

            args : Union[str, Sequence[str]]
                command to run, as string or list of string arguments.
            log : str
                log file name.
            env : Optional[Dict[str, str]]
                environment variable dictionary.  None to inherit from parent.
            cwd : Optional[str]
                working directory path.  None to inherit from parent.

        Returns
        -------
        results : Optional[Sequence[Union[int, Exception]]]
            if user cancelled the subprocesses, None is returned.  Otherwise, a list of
            subprocess return codes or exceptions are returned.
        """
        num_proc = len(proc_info_list)
        if num_proc == 0:
            return []

        coro_list = [self.async_new_subprocess(args, log, env, cwd) for args, log, env, cwd in
                     proc_info_list]

        return batch_async_task(coro_list)

    def batch_subprocess_flow(self, proc_info_list):
        # type: (Sequence[Sequence[FlowInfo]]) -> Optional[Sequence[Union[int, Exception]]]
        """Run all given subprocesses flow in parallel.

        Parameters
        ----------
        proc_info_list : Sequence[Sequence[FlowInfo]
            a list of process flow information.  Each element is a sequence of tuples of:

            args : Union[str, Sequence[str]]
                command to run, as string or list of string arguments.
            log : str
                log file name.
            env : Optional[Dict[str, str]]
                environment variable dictionary.  None to inherit from parent.
            cwd : Optional[str]
                working directory path.  None to inherit from parent.
            vfun : Sequence[Callable[[Optional[int], str], Any]]
                a function to validate if it is ok to execute the next process.  The output of the
                last function is returned.  The first argument is the return code, the second
                argument is the log file name.

        Returns
        -------
        results : Optional[Sequence[Any]]
            if user cancelled the subprocess flows, None is returned.  Otherwise, a list of
            flow return values or exceptions are returned.
        """
        num_proc = len(proc_info_list)
        if num_proc == 0:
            return []

        coro_list = [self.async_new_subprocess_flow(flow_info) for flow_info in proc_info_list]

        return batch_async_task(coro_list)
