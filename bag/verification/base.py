# -*- coding: utf-8 -*-

"""This module defines Checker, an abstract base class that handles LVS/RCX."""

from typing import TYPE_CHECKING, List, Dict, Any, Tuple, Sequence, Optional

import abc

from ..io.template import new_template_env
from ..concurrent.core import SubProcessManager

if TYPE_CHECKING:
    from ..concurrent.core import FlowInfo, ProcInfo


class Checker(abc.ABC):
    """A class that handles LVS/RCX.

    Parameters
    ----------
    tmp_dir : str
        temporary directory to save files in.
    """
    def __init__(self, tmp_dir):
        # type: (str) -> None
        self.tmp_dir = tmp_dir
        self._tmp_env = new_template_env('bag.verification', 'templates')

    @abc.abstractmethod
    def get_rcx_netlists(self, lib_name, cell_name):
        # type: (str, str) -> List[str]
        """Returns a list of generated extraction netlist file names.

        Parameters
        ----------
        lib_name : str
            library name.
        cell_name : str
            cell_name

        Returns
        -------
        netlists : List[str]
            a list of generated extraction netlist file names.  The first index is the main netlist.
        """
        return []

    @abc.abstractmethod
    async def async_run_lvs(self, lib_name, cell_name, sch_view='schematic',
                            lay_view='layout', params=None, **kwargs):
        # type: (str, str, str, str, Optional[Dict[str, Any]], Any) -> Tuple[bool, str]
        """A coroutine for running LVS.

        Parameters
        ----------
        lib_name : str
            library name.
        cell_name : str
            cell name.
        sch_view : str
            schematic view name.  Optional.
        lay_view : str
            layout view name.  Optional.
        params : Optional[Dict[str, Any]]
            optional LVS parameter values.
        kwargs : Any
            optional keyword arguments.
            gds_layout_path : str
                Path to the gds of the layout. If passed, do not export layout, instead copy gds


        Returns
        -------
        success : bool
            True if LVS succeeds.
        log_fname : str
            LVS log file name.
        """
        return False, ''

    @abc.abstractmethod
    async def async_run_rcx(self, lib_name, cell_name, sch_view='schematic',
                            lay_view='layout', params=None):
        # type: (str, str, str, str, Optional[Dict[str, Any]]) -> Tuple[Optional[str], str]
        """A coroutine for running RCX.

        Parameters
        ----------
        lib_name : str
            library name.
        cell_name : str
            cell name.
        sch_view : str
            schematic view name.  Optional.
        lay_view : str
            layout view name.  Optional.
        params : Optional[Dict[str, Any]]
            optional RCX parameter values.

        Returns
        -------
        netlist : Optional[str]
            The RCX netlist file name.  None if RCX failed, empty if no extracted
            netlist is generated
        log_fname : str
            RCX log file name.
        """
        return '', ''

    @abc.abstractmethod
    async def async_export_layout(self, lib_name, cell_name, out_file,
                                  view_name='layout', params=None):
        # type: (str, str, str, str, Optional[Dict[str, Any]]) -> str
        """A coroutine for exporting layout.

        Parameters
        ----------
        lib_name : str
            library name.
        cell_name : str
            cell name.
        view_name : str
            layout view name.
        out_file : str
            output file name.
        params : Optional[Dict[str, Any]]
            optional export parameter values.

        Returns
        -------
        log_fname : str
            log file name.
        """
        return ''

    @abc.abstractmethod
    async def async_export_schematic(self, lib_name, cell_name, out_file,
                                     view_name='schematic', params=None):
        # type: (str, str, str, str, Optional[Dict[str, Any]]) -> str
        """A coroutine for exporting schematic.

        Parameters
        ----------
        lib_name : str
            library name.
        cell_name : str
            cell name.
        view_name : str
            schematic view name.
        out_file : str
            output file name.
        params : Optional[Dict[str, Any]]
            optional export parameter values.

        Returns
        -------
        log_fname : str
            log file name.
        """
        return ''

    def render_file_template(self, temp_name, params):
        # type: (str, Dict[str, Any]) -> str
        """Returns the rendered content from the given template file."""
        template = self._tmp_env.get_template(temp_name)
        return template.render(**params)

    def render_string_template(self, content, params):
        # type: (str, Dict[str, Any]) -> str
        """Returns the rendered content from the given template string."""
        template = self._tmp_env.from_string(content)
        return template.render(**params)


class SubProcessChecker(Checker, abc.ABC):
    """An implementation of :class:`Checker` using :class:`SubProcessManager`.

    Parameters
    ----------
    tmp_dir : str
        temporary file directory.
    max_workers : int
        maximum number of parallel processes.
    cancel_timeout : float
        timeout for cancelling a subprocess.
    """

    def __init__(self, tmp_dir, max_workers, cancel_timeout):
        # type: (str, int, float) -> None
        Checker.__init__(self, tmp_dir)
        self._manager = SubProcessManager(max_workers=max_workers, cancel_timeout=cancel_timeout)

    @abc.abstractmethod
    def setup_lvs_flow(self, lib_name, cell_name, sch_view='schematic',
                       lay_view='layout', params=None, **kwargs):
        # type: (str, str, str, str, Optional[Dict[str, Any]], Any) -> Sequence[FlowInfo]
        """This method performs any setup necessary to configure a LVS subprocess flow.

        Parameters
        ----------
        lib_name : str
            library name.
        cell_name : str
            cell name.
        sch_view : str
            schematic view name.
        lay_view : str
            layout view name.
        params : Optional[Dict[str, Any]]
            optional LVS parameter values.
        kwargs : Any
            optional keyword arguments.
            gds_layout_path : str
                Path to the gds of the layout. If passed, do not export layout, instead copy gds


        Returns
        -------
        flow_info : Sequence[FlowInfo]
            the LVS flow information list.  Each element is a tuple of:

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
                last function is returned.  The first argument is the return code, the
                second argument is the log file name.
        """
        return []

    @abc.abstractmethod
    def setup_rcx_flow(self, lib_name, cell_name, sch_view='schematic',
                       lay_view='layout', params=None):
        # type: (str, str, str, str, Optional[Dict[str, Any]]) -> Sequence[FlowInfo]
        """This method performs any setup necessary to configure a RCX subprocess flow.

        Parameters
        ----------
        lib_name : str
            library name.
        cell_name : str
            cell name.
        sch_view : str
            schematic view name.
        lay_view : str
            layout view name.
        params : Optional[Dict[str, Any]]
            optional RCX parameter values.

        Returns
        -------
        flow_info : Sequence[FlowInfo]
            the RCX flow information list.  Each element is a tuple of:

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
                last function is returned.  The first argument is the return code, the
                second argument is the log file name.
        """
        return []

    @abc.abstractmethod
    def setup_export_layout(self, lib_name, cell_name, out_file, view_name='layout', params=None):
        # type: (str, str, str, str, Optional[Dict[str, Any]]) -> ProcInfo
        """This method performs any setup necessary to export layout.

        Parameters
        ----------
        lib_name : str
            library name.
        cell_name : str
            cell name.
        out_file : str
            output file name.
        view_name : str
            layout view name.
        params : Optional[Dict[str, Any]]
            optional export parameter values.

        Returns
        -------
        args : Union[str, Sequence[str]]
            command to run, as string or list of string arguments.
        log : str
            log file name.
        env : Optional[Dict[str, str]]
            environment variable dictionary.  None to inherit from parent.
        cwd : Optional[str]
            working directory path.  None to inherit from parent.
        """
        return '', '', None, None

    @abc.abstractmethod
    def setup_export_schematic(self, lib_name, cell_name, out_file,
                               view_name='schematic', params=None):
        # type: (str, str, str, str, Optional[Dict[str, Any]]) -> ProcInfo
        """This method performs any setup necessary to export schematic.

        Parameters
        ----------
        lib_name : str
            library name.
        cell_name : str
            cell name.
        out_file : str
            output file name.
        view_name : str
            layout view name.
        params : Optional[Dict[str, Any]]
            optional export parameter values.

        Returns
        -------
        args : Union[str, Sequence[str]]
            command to run, as string or list of string arguments.
        log : str
            log file name.
        env : Optional[Dict[str, str]]
            environment variable dictionary.  None to inherit from parent.
        cwd : Optional[str]
            working directory path.  None to inherit from parent.
        """
        return '', '', None, None

    async def async_run_lvs(self, lib_name: str, cell_name: str,
                            sch_view: str = 'schematic',
                            lay_view: str = 'layout',
                            params: Optional[Dict[str, Any]] = None,
                            **kwargs: Any,
                            ) -> Tuple[bool, str]:

        flow_info = self.setup_lvs_flow(lib_name, cell_name, sch_view, lay_view, params, **kwargs)
        return await self._manager.async_new_subprocess_flow(flow_info)

    async def async_run_rcx(self, lib_name: str, cell_name: str,
                            sch_view: str = 'schematic',
                            lay_view: str = 'layout',
                            params: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
        flow_info = self.setup_rcx_flow(lib_name, cell_name, sch_view, lay_view, params)
        return await self._manager.async_new_subprocess_flow(flow_info)

    async def async_export_layout(self, lib_name: str, cell_name: str,
                                  out_file: str, view_name: str = 'layout',
                                  params: Optional[Dict[str, Any]] = None) -> str:
        proc_info = self.setup_export_layout(lib_name, cell_name, out_file, view_name, params)
        await self._manager.async_new_subprocess(*proc_info)
        return proc_info[1]

    async def async_export_schematic(self, lib_name: str, cell_name: str,
                                     out_file: str, view_name: str = 'layout',
                                     params: Optional[Dict[str, Any]] = None) -> str:
        proc_info = self.setup_export_schematic(lib_name, cell_name, out_file, view_name, params)
        await self._manager.async_new_subprocess(*proc_info)
        return proc_info[1]
