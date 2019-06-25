# -*- coding: utf-8 -*-

"""This module provides functions to help you run external processes.
"""

import os
import sys

from .common import bag_encoding, bag_codec_error
from .file import write_file

import multiprocessing
# noinspection PyCompatibility
import concurrent.futures

if sys.version_info[0] < 3:
    # use subprocess32 for timeout feature.
    if os.name != 'posix':
        raise Exception('bag.io.process module current only works for POSIX systems.')
    # noinspection PyUnresolvedReferences,PyPackageRequirements
    import subprocess32 as subprocess
else:
    import subprocess


def run_proc_with_quit(proc_id, quit_dict, args, logfile=None, append=False, env=None, cwd=None):
    if logfile is None:
        logfile = os.devnull

    mode = 'ab' if append else 'wb'
    with open(logfile, mode) as logf:
        if proc_id in quit_dict:
            return None
        proc = subprocess.Popen(args, stdout=logf, stderr=subprocess.STDOUT,
                                env=env, cwd=cwd)
        retcode = None
        num_kill = 0
        timeout = 0.05
        while retcode is None and num_kill <= 2:
            try:
                retcode = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                if proc_id in quit_dict:
                    if num_kill == 0:
                        proc.terminate()
                        timeout = quit_dict[proc_id]
                    elif num_kill == 1:
                        proc.kill()
                    num_kill += 1

        return proc.returncode


def run_and_wait(args, timeout=None, logfile=None, append=False,
                 env=None, cwd=None):
    """Run a command in a subprocess, then wait for it to finish.

    Parameters
    ----------
    args : string or list[string]
        the command to run.  Should be either a command string or a list
        of command string and its arguments as strings.  A list is preferred;
        see Python subprocess documentation.
    timeout : float or None
        the amount of time to wait for the command to finish, in seconds.
        If None, waits indefinitely.
    logfile : string or None
        If given, stdout and stderr will be written to this file.
    append : bool
        True to append to the logfile.  Defaults to False.
    env : dict[string, any]
        If not None, environment variables of the subprocess will be set
        according to this dictionary instead of inheriting from current
        process.
    cwd : string or None
        The current working directory of the subprocess.

    Returns
    -------
    output : string
        the standard output and standard error from the command.

    Raises
    ------
    subprocess.CalledProcessError
        if any error occurred in the subprocess.
    """
    output = subprocess.check_output(args, stderr=subprocess.STDOUT,
                                     timeout=timeout, env=env, cwd=cwd)
    output = output.decode(encoding=bag_encoding, errors=bag_codec_error)

    if logfile is not None:
        write_file(logfile, output, append=append)

    return output


class ProcessManager(object):
    """A class that manages subprocesses.

    This class is for starting processes that you do not need to wait on,
    and allows you to query for their status or terminate/kill them if needed.

    Parameters
    ----------
    max_workers : int or None
        number of maximum allowed subprocesses.  If None, defaults to system
        CPU count.
    cancel_timeout : float or None
        Number of seconds to wait for a process to terminate once SIGTERM or
        SIGKILL is issued.  Defaults to 10 seconds.
    """
    def __init__(self, max_workers=None, cancel_timeout=10.0):
        if max_workers is None:
            max_workers = multiprocessing.cpu_count()
        if cancel_timeout is None:
            cancel_timeout = 10.0
        self._exec = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._cancel_timeout = cancel_timeout
        self._future_dict = {}
        self._quit_dict = {}

    def close(self, timeout=10.0):
        """Cancel all processes.

        Parameters
        ----------
        timeout : float
            time to wait in seconds for each process to terminate.
        """
        for proc_id in self._future_dict.keys():
            self.cancel(proc_id, timeout=timeout)
        self._exec.shutdown()
        self._quit_dict.clear()
        self._future_dict.clear()

    def new_thread(self, fun, basename=None, callback=None):
        """Put a new custom task in queue.

        Execute the given function in a thread asynchronously.  The given function
        must take two arguments, The first argument is a unique string that represents
        this task, and the second argument is a dictionary.  The dictionary will
        map the unique string to a timeout (in second) if this task is being cancelled.
        The function should periodically check the dictionary and terminate gracefully.

        Before function returns, it should also delete the unique string from dictionary
        if it exists.

        Parameters
        ----------
        fun : callable
            the function to execute in a thread, as described above.
        basename : string or None
            If given, this will be used as the basis for generating the unique
            process ID.
        callback : callable
            If given, this function will automatically be executed when the
            process finished.  This function should take a single argument,
            which is a Future object that returns the return code of the
            process.

        Returns
        -------
        proc_id : string
            a unique string representing this process.  Can be used later
            to query process status or cancel process.
        """
        # find unique process ID
        proc_id = basename or 'proc'
        cur_idx = 1
        while proc_id in self._future_dict:
            proc_id = '%s_%d' % (proc_id, cur_idx)
            cur_idx += 1

        future = self._exec.submit(fun, proc_id, self._quit_dict)
        if callback is not None:
            future.add_done_callback(callback)

        self._future_dict[proc_id] = future
        return proc_id

    def new_process(self, args, basename=None, logfile=None, append=False,
                    env=None, cwd=None, callback=None):
        """Put a new process in queue.

        When the process is done, its return code will be returned.

        Parameters
        ----------
        args : string or list[string]
            the command to run as a string or list of string arguments.  See
            Python subprocess documentation.  list of string format is preferred.
        basename : string or None
            If given, this will be used as the basis for generating the unique
            process ID.
        logfile : string or None
            If given, stdout and stderr will be written to this file.  Otherwise,
            they will be redirected to `os.devnull`.
        append : bool
            True to append to ``logfile`` instead of overwritng it.
        env : dict[string, string] or None
            If given, environment variables of the process will be set according
            to this dictionary.
        cwd : string or None
            current working directory of the process.
        callback : callable
            If given, this function will automatically be executed when the
            process finished.  This function should take a single argument,
            which is a Future object that returns the return code of the
            process.

        Returns
        -------
        proc_id : string
            a unique string representing this process.  Can be used later
            to query process status or cancel process.
        """
        # find unique process ID
        proc_id = basename or 'proc'
        cur_idx = 1
        while proc_id in self._future_dict:
            proc_id = '%s_%d' % (proc_id, cur_idx)
            cur_idx += 1

        future = self._exec.submit(self._start_cmd, args, proc_id,
                                   logfile=logfile, append=append, env=env, cwd=cwd)
        if callback is not None:
            future.add_done_callback(callback)

        self._future_dict[proc_id] = future
        return proc_id

    @staticmethod
    def _get_output(future, timeout=None):
        """Get output from future.  Return None when exception."""
        try:
            if future.exception(timeout=timeout) is None:
                return future.result()
            else:
                return None
        except concurrent.futures.CancelledError:
            return None

    def cancel(self, proc_id, timeout=None):
        """Cancel the given process.

        If the process haven't started, this method prevents it from started.
        Otherwise, we first send a SIGTERM signal to kill the process.  If
        after ``timeout`` seconds the process is still alive, we will send a
        SIGKILL signal.  If after another ``timeout`` seconds the process is
        still alive, an Exception will be raised.

        Parameters
        ----------
        proc_id : string
            the process ID to cancel.
        timeout : float or None
            number of seconds to wait for cancellation.  If None, use default
            timeout.

        Returns
        -------
        output :
            output of the thread if it successfully terminates.
            Otherwise, return None.
        """
        if timeout is None:
            timeout = self._cancel_timeout

        future = self._future_dict.get(proc_id, None)
        if future is None:
            return None
        if future.done():
            # process already done, return status.
            del self._future_dict[proc_id]
            return self._get_output(future)
        if future.cancel():
            # we cancelled process before it made into the thread pool.
            del self._future_dict[proc_id]
            return None
        else:
            # inform thread it should try to quit.
            self._quit_dict[proc_id] = timeout
            try:
                output = self._get_output(future, timeout=4 * timeout)
                del self._future_dict[proc_id]
                return output
            except concurrent.futures.TimeoutError:
                # shouldn't get here, but we did
                print("*WARNING* worker thread refuse to die...")
                del self._future_dict[proc_id]
                return None

    def done(self, proc_id):
        """Returns True if the given process finished or is cancelled successfully.

        Parameters
        ----------
        proc_id : string
            the process ID.

        Returns
        -------
        done : bool
            True if the process is cancelled or completed.
        """
        return self._future_dict[proc_id].done()

    def wait(self, proc_id, timeout=None, cancel_timeout=None):
        """Wait for the given process to finish, then return its return code.

        If ``timeout`` is None, waits indefinitely.  Otherwise, if after
        ``timeout`` seconds the process is still running, a
        :class:`concurrent.futures.TimeoutError` will be raised.
        However, it is safe to catch this error and call wait again.

        If Ctrl-C is pressed before process finish or before timeout
        is reached, the process will be cancelled.

        Parameters
        ----------
        proc_id : string
            the process ID.
        timeout : float or None
            number of seconds to wait.  If None, waits indefinitely.
        cancel_timeout : float or None
            number of seconds to wait for process cancellation.  If None,
            use default timeout.

        Returns
        -------
        output :
            output of the thread if it successfully terminates.  Otherwise return None.
        """
        if cancel_timeout is None:
            cancel_timeout = self._cancel_timeout

        future = self._future_dict[proc_id]
        try:
            output = future.result(timeout=timeout)
            # remove future from dictionary.
            del self._future_dict[proc_id]
            return output
        except KeyboardInterrupt:
            # cancel the process
            print('KeyboardInterrupt received, cancelling %s...' % proc_id)
            return self.cancel(proc_id, timeout=cancel_timeout)

    def _start_cmd(self, args, proc_id, logfile=None, append=False, env=None, cwd=None):
        """The function that actually starts the subprocess.  Executed by thread."""

        retcode = run_proc_with_quit(proc_id, self._quit_dict, args, logfile=logfile,
                                     append=append, env=env, cwd=cwd)
        if proc_id in self._quit_dict:
            del self._quit_dict[proc_id]

        return retcode
