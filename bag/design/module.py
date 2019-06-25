# -*- coding: utf-8 -*-

"""This module defines base design module class and primitive design classes.
"""

import os
import abc
from typing import TYPE_CHECKING, List, Dict, Optional, Tuple, Any, Type, Set, Sequence, \
    Callable, Union

from bag import float_to_si_string
from bag.io import read_yaml
from bag.util.cache import DesignMaster, MasterDB

if TYPE_CHECKING:
    from bag.core import BagProject
    from bag.layout.core import TechInfo


class ModuleDB(MasterDB):
    """A database of all modules.

    This class is responsible for keeping track of module libraries and
    creating new modules.

    Parameters
    ----------
    lib_defs : str
        path to the design library definition file.
    tech_info : TechInfo
        the TechInfo instance.
    sch_exc_libs : List[str]
        list of libraries that are excluded from import.
    prj : Optional[BagProject]
        the BagProject instance.
    name_prefix : str
        generated layout name prefix.
    name_suffix : str
        generated layout name suffix.
    lib_path : str
        path to create generated library in.
    """

    def __init__(self, lib_defs, tech_info, sch_exc_libs, prj=None, name_prefix='',
                 name_suffix='', lib_path=''):
        # type: (str, TechInfo, List[str], Optional[BagProject], str, str, str) -> None
        MasterDB.__init__(self, '', lib_defs=lib_defs, name_prefix=name_prefix,
                          name_suffix=name_suffix)

        self._prj = prj
        self._tech_info = tech_info
        self._exc_libs = set(sch_exc_libs)
        self.lib_path = lib_path

    def create_master_instance(self, gen_cls, lib_name, params, used_cell_names, **kwargs):
        # type: (Type[Module], str, Dict[str, Any], Set[str], **Any) -> Module
        """Create a new non-finalized master instance.

        This instance is used to determine if we created this instance before.

        Parameters
        ----------
        gen_cls : Type[Module]
            the generator Python class.
        lib_name : str
            generated instance library name.
        params : Dict[str, Any]
            instance parameters dictionary.
        used_cell_names : Set[str]
            a set of all used cell names.
        **kwargs : Any
            optional arguments for the generator.

        Returns
        -------
        master : Module
            the non-finalized generated instance.
        """
        kwargs = kwargs.copy()
        kwargs['lib_name'] = lib_name
        kwargs['params'] = params
        kwargs['used_names'] = used_cell_names
        # noinspection PyTypeChecker
        return gen_cls(self, **kwargs)

    def create_masters_in_db(self, lib_name, content_list, debug=False):
        # type: (str, Sequence[Any], bool) -> None
        """Create the masters in the design database.

        Parameters
        ----------
        lib_name : str
            library to create the designs in.
        content_list : Sequence[Any]
            a list of the master contents.  Must be created in this order.
        debug : bool
            True to print debug messages
        """
        if self._prj is None:
            raise ValueError('BagProject is not defined.')

        self._prj.instantiate_schematic(lib_name, content_list, lib_path=self.lib_path)

    @property
    def tech_info(self):
        # type: () -> TechInfo
        """the :class:`~bag.layout.core.TechInfo` instance."""
        return self._tech_info

    def is_lib_excluded(self, lib_name):
        # type: (str) -> bool
        """Returns true if the given schematic library does not contain generators.

        Parameters
        ----------
        lib_name : str
            library name

        Returns
        -------
        is_excluded : bool
            True if given library is excluded.
        """
        return lib_name in self._exc_libs


class SchInstance(object):
    """A class representing a schematic instance.

    Parameters
    ----------
    database : ModuleDB
        the schematic generator database.
    gen_lib_name : str
        the schematic generator library name.
    gen_cell_name : str
        the schematic generator cell name.
    inst_name : str
        name of this instance.
    static : bool
        True if the schematic generator is static.
    connections : Optional[Dict[str, str]]
        If given, initialize instance terminal connections to this dictionary.
    master : Optional[Module]
        If given, set the master of this instance.
    parameters : Optional[Dict[str, Any]]
        If given, set the instance parameters to this dictionary.
    """

    def __init__(self,
                 database,  # type: MasterDB
                 gen_lib_name,  # type: str
                 gen_cell_name,  # type: str
                 inst_name,  # type: str
                 static=False,  # type: bool
                 connections=None,  # type: Optional[Dict[str, str]]
                 master=None,  # type: Optional[Module]
                 parameters=None,  # type: Optional[Dict[str, Any]]
                 ):
        # type: (...) -> None
        self._db = database
        self._master = master
        self._name = inst_name
        self._gen_lib_name = gen_lib_name
        self._gen_cell_name = gen_cell_name
        self._static = static
        self._term_mapping = {} if connections is None else connections
        self.parameters = {} if parameters is None else parameters

    def change_generator(self, gen_lib_name, gen_cell_name, static=False):
        # type: (str, str, bool) -> None
        """Change the master associated with this instance.

        All instance parameters and terminal mappings will be reset.

        Parameters
        ----------
        gen_lib_name : str
            the new schematic generator library name.
        gen_cell_name : str
            the new schematic generator cell name.
        static : bool
            True if the schematic generator is static.
        """
        self._master = None
        self._gen_lib_name = gen_lib_name
        self._gen_cell_name = gen_cell_name
        self._static = static
        self.parameters.clear()
        self._term_mapping.clear()

    @property
    def name(self):
        # type: () -> str
        """Returns the instance name."""
        return self._name

    @property
    def connections(self):
        # type: () -> Dict[str, str]
        """Returns the instance terminals connection dictionary."""
        return self._term_mapping

    @property
    def is_primitive(self):
        # type: () -> bool
        """Returns true if this is an instance of a primitive schematic generator."""
        if self._static:
            return True
        if self._master is None:
            raise ValueError('Instance %s has no master.  '
                             'Did you forget to call design()?' % self._name)
        return self._master.is_primitive()

    @property
    def should_delete(self):
        # type: () -> bool
        """Returns true if this instance should be deleted."""
        return self._master is not None and self._master.should_delete_instance()

    @property
    def master(self):
        # type: () -> Optional[Module]
        return self._master

    @property
    def master_cell_name(self):
        # type: () -> str
        """Returns the schematic master cell name."""
        return self._gen_cell_name if self._master is None else self._master.cell_name

    @property
    def master_key(self):
        # type: () -> Any
        return self._master.key

    def copy(self, inst_name, connections=None):
        # type: (str, Optional[Dict[str, str]]) -> SchInstance
        """Returns a copy of this SchInstance.

        Parameters
        ----------
        inst_name : str
            the new instance name.
        connections : Optional[Dict[str, str]]
            If given, will set the connections of this instance to this dictionary.

        Returns
        -------
        sch_inst : SchInstance
            a copy of this SchInstance, with connections potentially updated.
        """
        if connections is None:
            connections = self._term_mapping.copy()
        return SchInstance(self._db, self._gen_lib_name, self._gen_cell_name, inst_name,
                           static=self._static, connections=connections, master=self._master,
                           parameters=self.parameters.copy())

    def get_master_lib_name(self, impl_lib):
        # type: (str) -> str
        """Returns the schematic master library name.

        Parameters
        ----------
        impl_lib : str
            library where schematic masters will be created.

        Returns
        -------
        master_lib : str
            the schematic master library name.
        """
        return self._gen_lib_name if self.is_primitive else impl_lib

    def design_specs(self, *args, **kwargs):
        # type: (*Any, **Any) -> None
        """Update the instance master."""
        self._update_master('design_specs', args, kwargs)

    def design(self, *args, **kwargs):
        # type: (*Any, **Any) -> None
        """Update the instance master."""
        self._update_master('design', args, kwargs)

    def _update_master(self, design_fun, args, kwargs):
        # type: (str, Tuple[Any, ...], Dict[str, Any]) -> None
        """Create a new master."""
        if args:
            key = 'args'
            idx = 1
            while key in kwargs:
                key = 'args_%d' % idx
                idx += 1
            kwargs[key] = args
        else:
            key = None
        self._master = self._db.new_master(self._gen_lib_name, self._gen_cell_name,
                                           params=kwargs, design_args=key,
                                           design_fun=design_fun)  # type: Module
        if self._master.is_primitive():
            self.parameters.update(self._master.get_schematic_parameters())

    def implement_design(self, lib_name, top_cell_name='', prefix='', suffix='', **kwargs):
        # type: (str, str, str, str, **Any) -> None
        """Implement this design module in the given library.

        If the given library already exists, this method will not delete or override
        any pre-existing cells in that library.

        If you use this method, you do not need to call update_structure(),
        as this method calls it for you.

        This method only works if BagProject is given.

        Parameters
        ----------
        lib_name : str
            name of the new library to put the generated schematics.
        top_cell_name : str
            the cell name of the top level design.
        prefix : str
            prefix to add to cell names.
        suffix : str
            suffix to add to cell names.
        **kwargs : Any
            additional arguments.
        """
        if 'erase' in kwargs:
            print('DEPRECATED WARNING: erase is no longer supported '
                  'in implement_design() and has no effect')

        debug = kwargs.get('debug', False)
        rename_dict = kwargs.get('rename_dict', None)

        if not top_cell_name:
            top_cell_name = None

        if 'lib_path' in kwargs:
            self._db.lib_path = kwargs['lib_path']
        self._db.cell_prefix = prefix
        self._db.cell_suffix = suffix
        self._db.instantiate_masters([self._master], [top_cell_name], lib_name=lib_name,
                                     debug=debug, rename_dict=rename_dict)

    def get_layout_params(self, **kwargs):
        # type: (Any) -> Dict[str, Any]
        """Backwards compatibility function."""
        if hasattr(self._master, 'get_layout_params'):
            return getattr(self._master, 'get_layout_params')(**kwargs)
        else:
            return kwargs


class Module(DesignMaster, metaclass=abc.ABCMeta):
    """The base class of all schematic generators.  This represents a schematic master.

    This class defines all the methods needed to implement a design in the CAD database.

    Parameters
    ----------
    database : ModuleDB
        the design database object.
    yaml_fname : str
        the netlist information file name.
    **kwargs :
        additional arguments

    Attributes
    ----------
    parameters : dict[str, any]
        the design parameters dictionary.
    instances : dict[str, None or :class:`~bag.design.Module` or list[:class:`~bag.design.Module`]]
        the instance dictionary.
    """

    # noinspection PyUnusedLocal
    def __init__(self, database, yaml_fname, **kwargs):
        # type: (ModuleDB, str, **Any) -> None

        lib_name = kwargs['lib_name']
        params = kwargs['params']
        used_names = kwargs['used_names']
        design_fun = kwargs['design_fun']
        design_args = kwargs['design_args']

        self.tech_info = database.tech_info
        self.instances = {}  # type: Dict[str, Union[SchInstance, List[SchInstance]]]
        self.pin_map = {}
        self.new_pins = []
        self.parameters = {}
        self._pin_list = None

        self._yaml_fname = os.path.abspath(yaml_fname)
        self.sch_info = read_yaml(self._yaml_fname)

        self._orig_lib_name = self.sch_info['lib_name']
        self._orig_cell_name = self.sch_info['cell_name']
        self._design_fun = design_fun
        self._design_args = design_args

        # create initial instances and populate instance map
        for inst_name, inst_attr in self.sch_info['instances'].items():
            lib_name = inst_attr['lib_name']
            cell_name = inst_attr['cell_name']
            static = database.is_lib_excluded(lib_name)
            self.instances[inst_name] = SchInstance(database, lib_name, cell_name, inst_name,
                                                    static=static)

        # fill in pin map
        for pin in self.sch_info['pins']:
            self.pin_map[pin] = pin

        # initialize schematic master
        DesignMaster.__init__(self, database, lib_name, params, used_names)

    @property
    def pin_list(self):
        # type: () -> List[str]
        return self._pin_list

    @abc.abstractmethod
    def design(self, **kwargs):
        """To be overridden by subclasses to design this module.

        To design instances of this module, you can
        call their :meth:`.design` method or any other ways you coded.

        To modify schematic structure, call:

        :meth:`.rename_pin`

        :meth:`.delete_instance`

        :meth:`.replace_instance_master`

        :meth:`.reconnect_instance_terminal`

        :meth:`.array_instance`
        """
        pass

    def finalize(self):
        # type: () -> None
        """Finalize this master instance.
        """
        # invoke design function
        fun = getattr(self, self._design_fun)
        if self._design_args:
            args = self.params.pop(self._design_args)
            fun(*args, **self.params)
        else:
            fun(**self.params)

        # backwards compatibility
        if self.key is None:
            self.params.clear()
            self.params.update(self.parameters)
            self.update_master_info()

        self.children = set()
        for inst_list in self.instances.values():

            if isinstance(inst_list, SchInstance):
                if not inst_list.is_primitive:
                    self.children.add(inst_list.master_key)
            else:
                for inst in inst_list:
                    if not inst.is_primitive:
                        self.children.add(inst.master_key)

        # compute pins
        self._pin_list = [pin_name for pin_name, _ in self.new_pins]
        self._pin_list.extend((val for val in self.pin_map.values() if val))

        # call super finalize routine
        super(Module, self).finalize()

    @classmethod
    def get_params_info(cls):
        # type: () -> Optional[Dict[str, str]]
        """Returns a dictionary from parameter names to descriptions.

        Returns
        -------
        param_info : Optional[Dict[str, str]]
            dictionary from parameter names to descriptions.
        """
        return None

    def get_master_basename(self):
        # type: () -> str
        """Returns the base name to use for this instance.

        Returns
        -------
        basename : str
            the base name for this instance.
        """
        return self._orig_cell_name

    def get_content(self, lib_name, rename_fun):
        # type: (str, Callable[[str], str]) -> Optional[Tuple[Any,...]]
        """Returns the content of this master instance.

        Parameters
        ----------
        lib_name : str
            the library to create the design masters in.
        rename_fun : Callable[[str], str]
            a function that renames design masters.

        Returns
        -------
        content : Optional[Tuple[Any,...]]
            the master content data structure.
        """
        if self.is_primitive():
            return None

        # populate instance transform mapping dictionary
        inst_map = {}
        for inst_name, inst_list in self.instances.items():
            if isinstance(inst_list, SchInstance):
                inst_list = [inst_list]

            info_list = []
            for inst in inst_list:
                if not inst.should_delete:
                    cur_lib = inst.get_master_lib_name(lib_name)
                    info_list.append(dict(
                        name=inst.name,
                        lib_name=cur_lib,
                        cell_name=rename_fun(inst.master_cell_name),
                        params=inst.parameters,
                        term_mapping=inst.connections,
                    ))
            inst_map[inst_name] = info_list

        return (self._orig_lib_name, self._orig_cell_name, rename_fun(self.cell_name),
                self.pin_map, inst_map, self.new_pins)

    @property
    def cell_name(self):
        # type: () -> str
        """The master cell name."""
        if self.is_primitive():
            return self.get_cell_name_from_parameters()
        return super(Module, self).cell_name

    @property
    def orig_cell_name(self):
        # type: () -> str
        """The original schematic template cell name."""
        return self._orig_cell_name

    def is_primitive(self):
        # type: () -> bool
        """Returns True if this Module represents a BAG primitive.

        NOTE: This method is only used by BAG and schematic primitives.  This method prevents
        the module from being copied during design implementation.  Custom subclasses should
        not override this method.

        Returns
        -------
        is_primitive : bool
            True if this Module represents a BAG primitive.
        """
        return False

    def should_delete_instance(self):
        # type: () -> bool
        """Returns True if this instance should be deleted based on its parameters.

        This method is mainly used to delete 0 finger or 0 width transistors.  However,
        You can override this method if there exists parameter settings which corresponds
        to an empty schematic.

        Returns
        -------
        delete : bool
            True if parent should delete this instance.
        """
        return False

    def get_schematic_parameters(self):
        # type: () -> Dict[str, str]
        """Returns the schematic parameter dictionary of this instance.

        NOTE: This method is only used by BAG primitives, as they are
        implemented with parameterized cells in the CAD database.  Custom
        subclasses should not override this method.

        Returns
        -------
        params : Dict[str, str]
            the schematic parameter dictionary.
        """
        return {}

    def get_cell_name_from_parameters(self):
        """Returns new cell name based on parameters.

        NOTE: This method is only used by BAG primitives.  This method
        enables a BAG primitive to change the cell master based on
        design parameters (e.g. change transistor instance based on the
        intent parameter).  Custom subclasses should not override this
        method.

        Returns
        -------
        cell : str
            the cell name based on parameters.
        """
        return super(Module, self).cell_name

    def rename_pin(self, old_pin, new_pin):
        # type: (str, str) -> None
        """Renames an input/output pin of this schematic.

        NOTE: Make sure to call :meth:`.reconnect_instance_terminal` so that instances are
        connected to the new pin.

        Parameters
        ----------
        old_pin : str
            the old pin name.
        new_pin : str
            the new pin name.
        """
        self.pin_map[old_pin] = new_pin

    def add_pin(self, new_pin, pin_type):
        # type: (str, str) -> None
        """Adds a new pin to this schematic.

        NOTE: Make sure to call :meth:`.reconnect_instance_terminal` so that instances are
        connected to the new pin.

        Parameters
        ----------
        new_pin : str
            the new pin name.
        pin_type : str
            the new pin type.  We current support "input", "output", or "inputOutput"
        """
        self.new_pins.append([new_pin, pin_type])

    def remove_pin(self, remove_pin):
        # type: (str) -> None
        """Removes a pin from this schematic.

        Parameters
        ----------
        remove_pin : str
            the pin to remove.
        """
        self.rename_pin(remove_pin, '')

    def delete_instance(self, inst_name):
        # type: (str) -> None
        """Delete the instance with the given name.

        Parameters
        ----------
        inst_name : str
            the child instance to delete.
        """
        self.instances[inst_name] = []

    def replace_instance_master(self, inst_name, lib_name, cell_name, static=False, index=None):
        # type: (str, str, str, bool, Optional[int]) -> None
        """Replace the master of the given instance.

        NOTE: all terminal connections will be reset.  Call reconnect_instance_terminal() to modify
        terminal connections.

        Parameters
        ----------
        inst_name : str
            the child instance to replace.
        lib_name : str
            the new library name.
        cell_name : str
            the new cell name.
        static : bool
            True if we're replacing instance with a static schematic instead of a design module.
        index : Optional[int]
            If index is not None and the child instance has been arrayed, this is the instance
            array index that we are replacing.
            If index is None, the entire child instance (whether arrayed or not) will be replaced
            by a single new instance.
        """
        if inst_name not in self.instances:
            raise ValueError('Cannot find instance with name: %s' % inst_name)

        # check if this is arrayed
        if index is not None and isinstance(self.instances[inst_name], list):
            self.instances[inst_name][index].change_generator(lib_name, cell_name, static=static)
        else:
            self.instances[inst_name] = SchInstance(self.master_db, lib_name, cell_name, inst_name,
                                                    static=static)

    def reconnect_instance_terminal(self, inst_name, term_name, net_name, index=None):
        """Reconnect the instance terminal to a new net.

        Parameters
        ----------
        inst_name : str
            the child instance to modify.
        term_name : Union[str, List[str]]
            the instance terminal name to reconnect.
            If a list is given, it is applied to each arrayed instance.
        net_name : Union[str, List[str]]
            the net to connect the instance terminal to.
            If a list is given, it is applied to each arrayed instance.
        index : Optional[int]
            If not None and the given instance is arrayed, will only modify terminal
            connection for the instance at the given index.
            If None and the given instance is arrayed, all instances in the array
            will be reconnected.
        """
        if index is not None:
            # only modify terminal connection for one instance in the array
            if isinstance(term_name, str) and isinstance(net_name, str):
                self.instances[inst_name][index].connections[term_name] = net_name
            else:
                raise ValueError('If index is not None, '
                                 'both term_name and net_name must be string.')
        else:
            # modify terminal connection for all instances in the array
            cur_inst_list = self.instances[inst_name]
            if isinstance(cur_inst_list, SchInstance):
                cur_inst_list = [cur_inst_list]

            num_insts = len(cur_inst_list)
            if not isinstance(term_name, list) and not isinstance(term_name, tuple):
                if not isinstance(term_name, str):
                    raise ValueError('term_name = %s must be string.' % term_name)
                term_name = [term_name] * num_insts
            else:
                if len(term_name) != num_insts:
                    raise ValueError('term_name length = %d != %d' % (len(term_name), num_insts))

            if not isinstance(net_name, list) and not isinstance(net_name, tuple):
                if not isinstance(net_name, str):
                    raise ValueError('net_name = %s must be string.' % net_name)
                net_name = [net_name] * num_insts
            else:
                if len(net_name) != num_insts:
                    raise ValueError('net_name length = %d != %d' % (len(net_name), num_insts))

            for inst, tname, nname in zip(cur_inst_list, term_name, net_name):
                inst.connections[tname] = nname

    def array_instance(self, inst_name, inst_name_list, term_list=None):
        # type: (str, List[str], Optional[List[Dict[str, str]]]) -> None
        """Replace the given instance by an array of instances.

        This method will replace self.instances[inst_name] by a list of
        Modules.  The user can then design each of those modules.

        Parameters
        ----------
        inst_name : str
            the instance to array.
        inst_name_list : List[str]
            a list of the names for each array item.
        term_list : Optional[List[Dict[str, str]]]
            a list of modified terminal connections for each array item.  The keys are
            instance terminal names, and the values are the net names to connect
            them to.  Only terminal connections different than the parent instance
            should be listed here.
            If None, assume terminal connections are not changed.
        """
        num_inst = len(inst_name_list)
        if not term_list:
            term_list = [None] * num_inst
        if num_inst != len(term_list):
            msg = 'len(inst_name_list) = %d != len(term_list) = %d'
            raise ValueError(msg % (num_inst, len(term_list)))

        orig_inst = self.instances[inst_name]
        if not isinstance(orig_inst, SchInstance):
            raise ValueError('Instance %s is already arrayed.' % inst_name)

        self.instances[inst_name] = [orig_inst.copy(iname, connections=iterm)
                                     for iname, iterm in zip(inst_name_list, term_list)]

    def design_dc_bias_sources(self,  # type: Module
                               vbias_dict,  # type: Optional[Dict[str, List[str]]]
                               ibias_dict,  # type: Optional[Dict[str, List[str]]]
                               vinst_name,  # type: str
                               iinst_name,  # type: str
                               define_vdd=True,  # type: bool
                               ):
        # type: (...) -> None
        """Convenience function for generating DC bias sources.

        Given DC voltage/current bias sources information, array the given voltage/current bias
        sources and configure the voltage/current.

        Each bias dictionary is a dictionary from bias source name to a 3-element list.  The first
        two elements are the PLUS/MINUS net names, respectively, and the third element is the DC
        voltage/current value as a string or float. A variable name can be given to define a
        testbench parameter.

        Parameters
        ----------
        vbias_dict : Optional[Dict[str, List[str]]]
            the voltage bias dictionary.  None or empty to disable.
        ibias_dict : Optional[Dict[str, List[str]]]
            the current bias dictionary.  None or empty to disable.
        vinst_name : str
            the DC voltage source instance name.
        iinst_name : str
            the DC current source instance name.
        define_vdd : bool
            True to include a supply voltage source connected to VDD/VSS, with voltage value 'vdd'.
        """
        if define_vdd and 'SUP' not in vbias_dict:
            vbias_dict = vbias_dict.copy()
            vbias_dict['SUP'] = ['VDD', 'VSS', 'vdd']

        for bias_dict, name_template, param_name, inst_name in \
                ((vbias_dict, 'V%s', 'vdc', vinst_name), (ibias_dict, 'I%s', 'idc', iinst_name)):
            if bias_dict:
                name_list, term_list, val_list, param_dict_list = [], [], [], []
                for name in sorted(bias_dict.keys()):
                    value_tuple = bias_dict[name]
                    pname, nname, bias_val = value_tuple[:3]
                    param_dict = value_tuple[3] if len(value_tuple) > 3 \
                        else None  # type: Optional[Dict]
                    term_list.append(dict(PLUS=pname, MINUS=nname))
                    name_list.append(name_template % name)
                    param_dict_list.append(param_dict)
                    if isinstance(bias_val, str):
                        val_list.append(bias_val)
                    elif isinstance(bias_val, int) or isinstance(bias_val, float):
                        val_list.append(float_to_si_string(bias_val))
                    else:
                        raise ValueError('value %s of type %s '
                                         'not supported' % (bias_val, type(bias_val)))

                self.array_instance(inst_name, name_list, term_list=term_list)
                for inst, val, param_dict in zip(self.instances[inst_name], val_list,
                                                 param_dict_list):
                    inst.parameters[param_name] = val
                    if param_dict is not None:
                        for k, v in param_dict.items():
                            if isinstance(v, str):
                                pass
                            elif isinstance(v, int) or isinstance(v, float):
                                v = float_to_si_string(v)
                            else:
                                raise ValueError('value %s of type %s not supported' % (v, type(v)))

                            inst.parameters[k] = v
            else:
                self.delete_instance(inst_name)

    def design_dummy_transistors(self, dum_info, inst_name, vdd_name, vss_name, net_map=None):
        # type: (List[Tuple[Any]], str, str, str, Optional[Dict[str, str]]) -> None
        """Convenience function for generating dummy transistor schematic.

        Given dummy information (computed by AnalogBase) and a BAG transistor instance,
        this method generates dummy schematics by arraying and modifying the BAG
        transistor instance.

        Parameters
        ----------
        dum_info : List[Tuple[Any]]
            the dummy information data structure.
        inst_name : str
            the BAG transistor instance name.
        vdd_name : str
            VDD net name.  Used for PMOS dummies.
        vss_name : str
            VSS net name.  Used for NMOS dummies.
        net_map : Optional[Dict[str, str]]
            optional net name transformation mapping.
        """
        if not dum_info:
            self.delete_instance(inst_name)
        else:
            num_arr = len(dum_info)
            arr_name_list = ['XDUMMY%d' % idx for idx in range(num_arr)]
            self.array_instance(inst_name, arr_name_list)

            for idx, ((mos_type, w, lch, th, s_net, d_net), fg) in enumerate(dum_info):
                if mos_type == 'pch':
                    cell_name = 'pmos4_standard'
                    sup_name = vdd_name
                else:
                    cell_name = 'nmos4_standard'
                    sup_name = vss_name
                if net_map is not None:
                    s_net = net_map.get(s_net, s_net)
                    d_net = net_map.get(d_net, d_net)
                s_name = s_net if s_net else sup_name
                d_name = d_net if d_net else sup_name

                self.replace_instance_master(inst_name, 'BAG_prim', cell_name, index=idx)
                self.reconnect_instance_terminal(inst_name, 'G', sup_name, index=idx)
                self.reconnect_instance_terminal(inst_name, 'B', sup_name, index=idx)
                self.reconnect_instance_terminal(inst_name, 'D', d_name, index=idx)
                self.reconnect_instance_terminal(inst_name, 'S', s_name, index=idx)
                self.instances[inst_name][idx].design(w=w, l=lch, nf=fg, intent=th)


class MosModuleBase(Module):
    """The base design class for the bag primitive transistor.

    Parameters
    ----------
    database : ModuleDB
        the design database object.
    yaml_file : str
        the netlist information file name.
    **kwargs :
        additional arguments
    """

    def __init__(self, database, yaml_file, **kwargs):
        Module.__init__(self, database, yaml_file, **kwargs)

    @classmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        return dict(
            w='transistor width, in meters or number of fins.',
            l='transistor length, in meters.',
            nf='transistor number of fingers.',
            intent='transistor threshold flavor.',
        )

    def design(self, w=1e-6, l=60e-9, nf=1, intent='standard'):
        pass

    def get_schematic_parameters(self):
        # type: () -> Dict[str, str]
        w_res = self.tech_info.tech_params['mos']['width_resolution']
        l_res = self.tech_info.tech_params['mos']['length_resolution']
        w = self.params['w']
        l = self.params['l']
        nf = self.params['nf']
        wstr = w if isinstance(w, str) else float_to_si_string(int(round(w / w_res)) * w_res)
        lstr = l if isinstance(l, str) else float_to_si_string(int(round(l / l_res)) * l_res)
        nstr = nf if isinstance(nf, str) else '%d' % nf

        return dict(w=wstr, l=lstr, nf=nstr)

    def get_cell_name_from_parameters(self):
        # type: () -> str
        mos_type = self.orig_cell_name.split('_')[0]
        return '%s_%s' % (mos_type, self.params['intent'])

    def is_primitive(self):
        # type: () -> bool
        return True

    def should_delete_instance(self):
        # type: () -> bool
        return self.params['nf'] == 0 or self.params['w'] == 0 or self.params['l'] == 0


class ResPhysicalModuleBase(Module):
    """The base design class for a real resistor parametrized by width and length.

    Parameters
    ----------
    database : ModuleDB
        the design database object.
    yaml_file : str
        the netlist information file name.
    **kwargs :
        additional arguments
    """

    def __init__(self, database, yaml_file, **kwargs):
        Module.__init__(self, database, yaml_file, **kwargs)

    @classmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        return dict(
            w='resistor width, in meters.',
            l='resistor length, in meters.',
            intent='resistor flavor.',
        )

    def design(self, w=1e-6, l=1e-6, intent='standard'):
        pass

    def get_schematic_parameters(self):
        # type: () -> Dict[str, str]
        w = self.params['w']
        l = self.params['l']
        wstr = w if isinstance(w, str) else float_to_si_string(w)
        lstr = l if isinstance(l, str) else float_to_si_string(l)

        return dict(w=wstr, l=lstr)

    def get_cell_name_from_parameters(self):
        # type: () -> str
        return 'res_%s' % self.params['intent']

    def is_primitive(self):
        # type: () -> bool
        return True

    def should_delete_instance(self):
        # type: () -> bool
        return self.params['w'] == 0 or self.params['l'] == 0


class ResMetalModule(Module):
    """The base design class for a metal resistor.

    Parameters
    ----------
    database : ModuleDB
        the design database object.
    yaml_file : str
        the netlist information file name.
    **kwargs :
        additional arguments
    """

    def __init__(self, database, yaml_file, **kwargs):
        Module.__init__(self, database, yaml_file, **kwargs)

    @classmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        return dict(
            w='resistor width, in meters.',
            l='resistor length, in meters.',
            layer='the metal layer ID.',
        )

    def design(self, w, l, layer):
        # type: (float, float, int) -> None
        pass

    def get_schematic_parameters(self):
        # type: () -> Dict[str, str]
        w = self.params['w']
        l = self.params['l']
        layer = self.params['layer']
        wstr = float_to_si_string(w)
        lstr = float_to_si_string(l)
        lay_str = str(layer)
        return dict(w=wstr, l=lstr, layer=lay_str)

    def is_primitive(self):
        # type: () -> bool
        return True

    def should_delete_instance(self):
        # type: () -> bool
        return self.params['w'] == 0 or self.params['l'] == 0
