# -*- coding: utf-8 -*-

"""This module defines the base template class.
"""

from typing import Dict, List, Iterator, Tuple, Optional, Union, Callable, Any

import abc
import math
import numpy as np
from itertools import chain

import bag
import bag.io
from .util import BBox
from .objects import Rect, Via, ViaInfo, Instance, InstanceInfo, PinInfo
from .objects import Path, Polygon, Blockage, Boundary
from bag.util.search import BinaryIterator

# try to import cybagoa module
try:
    import cybagoa
except ImportError:
    cybagoa = None


class TechInfo(object, metaclass=abc.ABCMeta):
    """A base class that create vias.

    This class provides the API for making vias.  Each process should subclass this class and
    implement the make_via method.

    Parameters
    ----------
    res : float
        the grid resolution of this technology.
    layout_unit : float
        the layout unit, in meters.
    via_tech : string
        the via technology library name.  This is usually the PDK library name.
    process_params : dict[str, any]
        process specific parameters.

    Attributes
    ----------
    tech_params : dict[str, any]
        technology specific parameters.
    """

    def __init__(self, res, layout_unit, via_tech, process_params):
        self._resolution = res
        self._layout_unit = layout_unit
        self._via_tech = via_tech
        self.tech_params = process_params

    @abc.abstractmethod
    def get_well_layers(self, sub_type):
        # type: (str) -> List[Tuple[str, str]]
        """Returns a list of well layers associated with the given substrate type."""
        return []

    @abc.abstractmethod
    def get_implant_layers(self, mos_type, res_type=None):
        # type: (str, Optional[str]) -> List[Tuple[str, str]]
        """Returns a list of implant layers associated with the given device type.

        Parameters
        ----------
        mos_type : str
            one of 'nch', 'pch', 'ntap', or 'ptap'
        res_type : Optional[str]
            If given, the return layers will be for the substrate of the given resistor type.

        Returns
        -------
        imp_list : List[Tuple[str, str]]
            list of implant layers.
        """
        return []

    @abc.abstractmethod
    def get_threshold_layers(self, mos_type, threshold, res_type=None):
        # type: (str, str, Optional[str]) -> List[Tuple[str, str]]
        """Returns a list of threshold layers."""
        return []

    @abc.abstractmethod
    def get_exclude_layer(self, layer_id):
        # type: (int) -> Tuple[str, str]
        """Returns the metal exclude layer"""
        return '', ''

    @abc.abstractmethod
    def get_dnw_margin_unit(self, dnw_mode):
        # type: (str) -> int
        """Returns the required DNW margin given the DNW mode.

        Parameters
        ----------
        dnw_mode : str
            the DNW mode string.

        Returns
        -------
        dnw_margin : int
            the DNW margin in resolution units.
        """
        return 0

    @abc.abstractmethod
    def get_dnw_layers(self):
        # type: () -> List[Tuple[str, str]]
        """Returns a list of layers that defines DNW.

        Returns
        -------
        lay_list : List[Tuple[str, str]]
            list of DNW layers.
        """
        return []

    @abc.abstractmethod
    def get_res_metal_layers(self, layer_id):
        # type: (int) -> List[Tuple[str, str]]
        """Returns a list of layers associated with the given metal resistor.

        Parameters
        ----------
        layer_id : int
            the metal layer ID.

        Returns
        -------
        res_list : List[Tuple[str, str]]
            list of resistor layers.
        """
        return []

    @abc.abstractmethod
    def get_metal_dummy_layers(self, layer_id):
        # type: (int) -> List[Tuple[str, str]]
        """Returns a list of layers associated with the given metal dummy layers.

        Parameters
        ----------
        layer_id : int
            the metal layer ID.

        Returns
        -------
        res_list : List[Tuple[str, str]]
            list of metal dummy layers.
        """
        return []

    @abc.abstractmethod
    def add_cell_boundary(self, template, box):
        """Adds a cell boundary object to the given template.
        
        This is usually the PR boundary.
        
        Parameters
        ----------
        template : TemplateBase
            the template to draw the cell boundary in.
        box : BBox
            the cell boundary bounding box.
        """
        pass

    @abc.abstractmethod
    def draw_device_blockage(self, template):
        """Draw device blockage layers on the given template.

        Parameters
        ----------
        template : TemplateBase
            the template to draw the device block layers on
        """
        pass

    @abc.abstractmethod
    def get_via_drc_info(self, vname, vtype, mtype, mw_unit, is_bot):
        """Return data structures used to identify VIA DRC rules.

        Parameters
        ----------
        vname : string
            the via type name.
        vtype : string
            the via type, square/hrect/vrect/etc.
        mtype : string
            name of the metal layer via is connecting.  Can be either top or bottom.
        mw_unit : int
            width of the metal, in resolution units.
        is_bot : bool
            True if the given metal is the bottom metal.

        Returns
        -------
        sp : Tuple[int, int]
            horizontal/vertical space between adjacent vias, in resolution units.
        sp2_list : List[Tuple[int, int]] or None
            horizontal/vertical space between adjacent vias if the via has 2 or more neighbors.
            None if no constraint.
        sp3_list : List[Tuple[int, int]] or None
            horizontal/vertical space between adjacent vias if the via has 3 or more neighbors.
            None if no constraint.
        sp6_list : List[Tuple[int, int]] or None
            horizontal/vertical space between adjacent vias if the via has 6 or more neighbors.
            None if no constraint.
        dim : Tuple[int, int]
            the via width/height in resolution units.
        enc : List[Tuple[int, int]]
            a list of valid horizontal/vertical enclosure of the via on the given metal
            layer, in resolution units.
        arr_enc : List[Tuple[int, int]] or None
            a list of valid horizontal/vertical enclosure of the via on the given metal
            layer if this is a "via array", in layout units.
            None if no constraint.
        arr_test : callable or None
            a function that accepts two inputs, the number of via rows and number of via
            columns, and returns True if those numbers describe a "via array".
            None if no constraint.
        """
        return (0, 0), [(0, 0)], [(0, 0)], [(0, 0)], (0, 0), [(0, 0)], None, None

    @abc.abstractmethod
    def get_min_space(self, layer_type, width, unit_mode=False, same_color=False):
        """Returns the minimum spacing needed around a wire on the given layer with the given width.

        Parameters
        ----------
        layer_type : str
            the wiring layer type.
        width : Union[float, int]
            the width of the wire, in layout units.
        unit_mode : bool
            True if dimension are given/returned in resolution units.
        same_color : bool
            True to use same-color spacing.

        Returns
        -------
        sp : Union[float, int]
            the minimum spacing needed.
        """
        return 0.0

    @abc.abstractmethod
    def get_min_line_end_space(self, layer_type, width, unit_mode=False):
        """Returns the minimum line-end spacing of a wire with given width.

        Parameters
        ----------
        layer_type : str
            the wiring layer type.
        width : Union[float, int]
            the width of the wire, in layout units.
        unit_mode : bool
            True if dimension are given/returned in resolution units.

        Returns
        -------
        sp : Union[float, int]
            the minimum line-end space.
        """
        return 0.0

    @abc.abstractmethod
    def get_min_length(self, layer_type, width):
        # type: (str, float) -> float
        """Returns the minimum length of a wire on the given layer with the given width.

        Parameters
        ----------
        layer_type : str
            the wiring layer type.
        width : float
            the width of the wire, in layout units.

        Returns
        -------
        min_length : float
            the minimum length.
        """
        return 0.0

    @abc.abstractmethod
    def get_layer_id(self, layer_name):
        """Return the layer id for the given layer name.

        Parameters
        ----------
        layer_name : string
            the layer name.

        Returns
        -------
        layer_id : int
            the layer ID.
        """
        return 0

    @abc.abstractmethod
    def get_layer_name(self, layer_id):
        """Return the layer name(s) for the given routing grid layer ID.

        Parameters
        ----------
        layer_id : int
            the routing grid layer ID.

        Returns
        -------
        name : string or Tuple[string]
            name of the layer.  Returns a tuple of names if this is a double
            patterning layer.
        """
        return ''

    @abc.abstractmethod
    def get_layer_type(self, layer_name):
        """Returns the metal type of the given wiring layer.

        Parameters
        ----------
        layer_name : str
            the wiring layer name.

        Returns
        -------
        metal_type : string
            the metal layer type.
        """
        return ''

    @abc.abstractmethod
    def get_via_name(self, bot_layer_id):
        """Returns the via type name of the given via.

        Parameters
        ----------
        bot_layer_id : int
            the via bottom layer ID

        Returns
        -------
        name : string
            the via type name.
        """
        return ''

    @abc.abstractmethod
    def get_metal_em_specs(self, layer_name, w, l=-1, vertical=False, **kwargs):
        """Returns a tuple of EM current/resistance specs of the given wire.

        Parameters
        ----------
        layer_name : str
            the metal layer name.
        w : float
            the width of the metal in layout units (dimension perpendicular to current flow).
        l : float
            the length of the metal in layout units (dimension parallel to current flow).
            If negative, disable length enhancement.
        vertical : bool
            True to compute vertical current.
        **kwargs :
            optional EM specs parameters.

        Returns
        -------
        idc : float
            maximum DC current, in Amperes.
        iac_rms : float
            maximum AC RMS current, in Amperes.
        iac_peak : float
            maximum AC peak current, in Amperes.
        """
        return float('inf'), float('inf'), float('inf')

    @abc.abstractmethod
    def get_via_em_specs(self, via_name,  # type: str
                         bm_layer,  # type: str
                         tm_layer,  # type: str
                         via_type='square',  # type: str
                         bm_dim=(-1, -1),  # type: Tuple[float, float]
                         tm_dim=(-1, -1),  # type: Tuple[float, float]
                         array=False,  # type: bool
                         **kwargs):
        # type: (...) -> Tuple[float ,float, float]
        """Returns a tuple of EM current/resistance specs of the given via.

        Parameters
        ----------
        via_name : str
            the via type name.
        bm_layer : str
            the bottom layer name.
        tm_layer : str
            the top layer name.
        via_type : str
            the via type, square/vrect/hrect/etc.
        bm_dim : Tuple[float, float]
            bottom layer metal width/length in layout units.  If negative,
            disable length/width enhancement.
        tm_dim : Tuple[float, float]
            top layer metal width/length in layout units.  If negative,
            disable length/width enhancement.
        array : bool
            True if this via is in a via array.
        **kwargs :
            optional EM specs parameters.

        Returns
        -------
        idc : float
            maximum DC current per via, in Amperes.
        iac_rms : float
            maximum AC RMS current per via, in Amperes.
        iac_peak : float
            maximum AC peak current per via, in Amperes.
        """
        return float('inf'), float('inf'), float('inf')

    @abc.abstractmethod
    def get_res_rsquare(self, res_type):
        """Returns R-square for the given resistor type.

        This is used to do some approximate resistor dimension calculation.

        Parameters
        ----------
        res_type : string
            the resistor type.

        Returns
        -------
        rsquare : float
            resistance in Ohms per unit square of the given resistor type.
        """
        return 0.0

    @abc.abstractmethod
    def get_res_width_bounds(self, res_type):
        """Returns the maximum and minimum resistor width for the given resistor type.

        Parameters
        ----------
        res_type : string
            the resistor type.

        Returns
        -------
        wmin : float
            minimum resistor width, in layout units.
        wmax : float
            maximum resistor width, in layout units.
        """
        return 0.0, 0.0

    @abc.abstractmethod
    def get_res_length_bounds(self, res_type):
        """Returns the maximum and minimum resistor length for the given resistor type.

        Parameters
        ----------
        res_type : string
            the resistor type.

        Returns
        -------
        lmin : float
            minimum resistor length, in layout units.
        lmax : float
            maximum resistor length, in layout units.
        """
        return 0.0, 0.0

    @abc.abstractmethod
    def get_res_min_nsquare(self, res_type):
        """Returns the minimum allowable number of squares for the given resistor type.

        Parameters
        ----------
        res_type : string
            the resistor type.

        Returns
        -------
        nsq_min : flaot
            minimum number of squares needed.
        """
        return 1.0

    @abc.abstractmethod
    def get_res_em_specs(self, res_type, w, l=-1, **kwargs):
        # type: (str, float, float, **Any) -> Tuple[float, float, float]
        """Returns a tuple of EM current/resistance specs of the given resistor.

        Parameters
        ----------
        res_type : string
            the resistor type string.
        w : float
            the width of the metal in layout units (dimension perpendicular to current flow).
        l : float
            the length of the metal in layout units (dimension parallel to current flow).
            If negative, disable length enhancement.
        **kwargs : Any
            optional EM specs parameters.

        Returns
        -------
        idc : float
            maximum DC current, in Amperes.
        iac_rms : float
            maximum AC RMS current, in Amperes.
        iac_peak : float
            maximum AC peak current, in Amperes.
        """
        return float('inf'), float('inf'), float('inf')

    @property
    def via_tech_name(self):
        """Returns the via technology library name."""
        return self._via_tech

    @property
    def pin_purpose(self):
        """Returns the layout pin purpose name."""
        return 'pin'

    @property
    def resolution(self):
        """Returns the grid resolution."""
        return self._resolution

    @property
    def layout_unit(self):
        """Returns the layout unit length, in meters."""
        return self._layout_unit

    def merge_well(self, template, inst_list, sub_type, threshold=None, res_type=None,
                   merge_imp=False):
        # type: ('TemplateBase', List[Instance], str, Optional[str], Optional[str], bool) -> None
        """Merge the well of the given instances together."""

        if threshold is not None:
            lay_iter = chain(self.get_well_layers(sub_type),
                             self.get_threshold_layers(sub_type, threshold, res_type=res_type))
        else:
            lay_iter = self.get_well_layers(sub_type)
        if merge_imp:
            lay_iter = chain(lay_iter, self.get_implant_layers(sub_type, res_type=res_type))

        for lay in lay_iter:
            tot_box = BBox.get_invalid_bbox()
            for inst in inst_list:
                cur_box = inst.master.get_rect_bbox(lay)
                tot_box = tot_box.merge(inst.translate_master_box(cur_box))
            if tot_box.is_physical():
                template.add_rect(lay, tot_box)

    def use_flip_parity(self):
        # type: () -> bool
        """Returns True if flip_parity dictionary is needed in this technology."""
        return True

    def finalize_template(self, template):
        """Perform any operations necessary on the given layout template before finalizing it.

        By default, nothing is done.

        Parameters
        ----------
        template : TemplateBase
            the template object.
        """
        pass

    def get_res_info(self, res_type, w, l, **kwargs):
        """Returns a dictionary containing EM information of the given resistor.

        Parameters
        ----------
        res_type : string or (string, string)
            the resistor type.
        w : float
            the resistor width in layout units (dimension perpendicular to current flow).
        l : float
            the resistor length in layout units (dimension parallel to current flow).
        **kwargs :
            optional parameters for EM rule calculations, such as nominal temperature,
            AC rms delta-T, etc.

        Returns
        -------
        info : dict[string, any]
            A dictionary of wire information.  Should have the following:

            resistance : float
                The resistance, in Ohms.
            idc : float
                The maximum allowable DC current, in Amperes.
            iac_rms : float
                The maximum allowable AC RMS current, in Amperes.
            iac_peak : float
                The maximum allowable AC peak current, in Amperes.
        """
        rsq = self.get_res_rsquare(res_type)
        res = l / w * rsq
        idc, irms, ipeak = self.get_res_em_specs(res_type, w, l=l, **kwargs)

        return dict(
            resistance=res,
            idc=idc,
            iac_rms=irms,
            iac_peak=ipeak,
        )

    def get_via_types(self, bmtype, tmtype):
        return [('square', 1), ('vrect', 2), ('hrect', 2)]

    def get_best_via_array(self, vname, bmtype, tmtype, bot_dir, top_dir, w, h, extend):
        """Maximize the number of vias in the given bounding box.

        Parameters
        ----------
        vname : str
            the via type name.
        bmtype : str
            the bottom metal type name.
        tmtype : str
            the top metal type name.
        bot_dir : str
            the bottom wire direction.  Either 'x' or 'y'.
        top_dir : str
            the top wire direction.  Either 'x' or 'y'.
        w : float
            width of the via array bounding box, in layout units.
        h : float
            height of the via array bounding box, in layout units.
        extend : bool
            True if via can extend beyond bounding box.

        Returns
        -------
        best_nxy : Tuple[int, int]
            optimal number of vias per row/column.
        best_mdim_list : List[Tuple[int, int]]
            a list of bottom/top layer width/height, in resolution units.
        vtype : str
            the via type to draw, square/hrect/vrect/etc.
        vdim : Tuple[int, int]
            the via width/height, in resolution units.
        via_space : Tuple[int, int]
            the via horizontal/vertical spacing, in resolution units.
        via_arr_dim : Tuple[int, int]
            the via array width/height, in resolution units.
        """
        # This entire optimization routine relies on the bounding box being measured integer units
        res = self._resolution
        w = int(round(w / res))
        h = int(round(h / res))

        # Depending on the routing direction of the metal, the provided width/height of the
        # bounding box may correspond to either the x direction or y direction.
        if bot_dir == 'x':
            bb, be = h, w
        else:
            bb, be = w, h
        if top_dir == 'x':
            tb, te = h, w
        else:
            tb, te = w, h

        # Initialize variables that will hold optimal via size at the end of the algorithm
        best_num = None
        best_nxy = [-1, -1]
        best_mdim_list = None
        best_type = None
        best_vdim = None
        best_sp = None
        best_adim = None

        # Perform via optimization algorithm for all available via types. Some technologies have
        # both square and rectangular via types, which can be used in different situations. Each
        # via_type has a weight which signifies a preference for choosing one type over another
        via_type_list = self.get_via_types(bmtype, tmtype)
        for vtype, weight in via_type_list:
            # Extract via drc information from the loaded tech yaml file. Some drc info is optional
            # so catch ValueErrors from missing info and move on
            try:
                # get space and enclosure rules for top and bottom layer
                bot_drc_info = self.get_via_drc_info(vname, vtype, bmtype, bb, True)
                top_drc_info = self.get_via_drc_info(vname, vtype, tmtype, tb, False)
                sp, sp2_list, sp3_list, sp6_list, dim, encb, arr_encb, arr_testb = bot_drc_info
                _, _, _, _, _, enct, arr_enct, arr_testt = top_drc_info
            except ValueError:
                continue
            # optional sp2/sp3 rules enable different spacing rules for via arrays with 2 or 3 neighbors
            if sp2_list is None:
                sp2_list = [sp]
            if sp3_list is None:
                sp3_list = sp2_list
            if sp6_list is None:
                sp6_list = sp3_list

            # Get minimum possible spacing between vias
            spx_min, spy_min = sp
            for high_sp_list in (sp2_list, sp3_list, sp6_list):
                for high_spx, high_spy in high_sp_list:
                    spx_min = min(spx_min, high_spx)
                    spy_min = min(spy_min, high_spy)

            # Get minimum possible enclosure size for top or bottom layers
            extx = 0
            exty = 0
            for enc in chain(encb, enct):
                extx = min(extx, enc[0])
                exty = min(exty, enc[1])

            # Allocate area in the bounding box for minimum enclosure, then find
            # maximum number of vias that can fit in the remaining area with the minimum spacing
            if np.isinf(spx_min):
                nx_max = 1 if (w - 2 * extx) // dim[0] else 0
            else:
                nx_max = (w + spx_min - 2 * extx) // (dim[0] + spx_min)
            if np.isinf(spy_min):
                ny_max = 1 if (h - 2 * exty) // dim[1] else 0
            else:
                ny_max = (h + spy_min - 2 * exty) // (dim[1] + spy_min)

            # Theoretically any combination of via array size from (1, 1) to (nx_max, ny_max) may actually
            # work within the given bound box. Here we enumerate a list all of these possible via combinations
            # starting from the max via number
            nxy_list = [(a * b, a, b) for a in range(1, nx_max + 1) for b in range(1, ny_max + 1)]
            nxy_list = sorted(nxy_list, reverse=True)

            # Initialize variables that will hold the best working via array size for this via type
            opt_nxy = None
            opt_mdim_list = None
            opt_adim = None
            opt_sp = None

            # This looping procedure will iterate over all possible via array configurations and select
            # one that maximizes the number of vias while meeting all rules
            for num, nx, ny in nxy_list:
                # Determine whether we should be using sp/sp2/sp3 rules for the current via configuration
                if (nx == 1 and ny >= 1) or (nx >= 1 and ny == 1):
                    sp_combo = [sp]
                elif nx == 2 and ny == 2:
                    sp_combo = sp2_list
                elif nx >= 6 and ny >= 6:
                    sp_combo = sp6_list
                else:
                    sp_combo = sp3_list

                # DRC rules can typically be satisfied with a number of different spacing rules, so here we
                # iterate over each to find the best one. Note that since we break out of the loop immediately upon
                # finding a valid via configuration, this code prioritizes spacing rules that are early on in the list
                for spx, spy in sp_combo:
                    # Compute a bounding box for the via array without the enclosure
                    w_arr = dim[0] if nx == 1 else nx * (spx + dim[0]) - spx
                    h_arr = dim[1] if ny == 1 else ny * (spy + dim[1]) - spy
                    mdim_list = [None, None]

                    # Loop over all possible enclosure types and check whether this via configuration satisfies
                    # one of them for both the bottom metal and top metal
                    for idx, (mdir, tot_enc_list, arr_enc, arr_test) in \
                            enumerate([(bot_dir, encb, arr_encb, arr_testb),
                                       (top_dir, enct, arr_enct, arr_testt)]):
                        # arr_test is a function that takes an array size as input and returns a boolean. If its
                        # is true the array size is valid and is added to the list of valid enclosures
                        if arr_test is not None and arr_test(ny, nx):
                            tot_enc_list = tot_enc_list + arr_enc

                        # If the routing direction is y, start by computing x-direction enclosure. ext_dim
                        # corresponds to x-direction. Vice-versa if the routing direction is x
                        if mdir == 'y':
                            enc_idx = 0
                            enc_dim = w_arr
                            ext_dim = h_arr
                            dim_lim = w
                            max_ext_dim = h
                        else:
                            enc_idx = 1
                            enc_dim = h_arr
                            ext_dim = w_arr
                            dim_lim = h
                            max_ext_dim = w

                        # Initialize variable to hold opposite direction enclosure size
                        min_ext_dim = None

                        # This loop selects the minimum opposite direction size that satisfies the enclosure
                        # rules
                        for enc in tot_enc_list:
                            cur_ext_dim = ext_dim + 2 * enc[1 - enc_idx]
                            # Check that the enclosure rule is satisfied. If extend is true, this passing enclosure
                            # size can exceed the maximum size set by the user provided bounding box
                            if (enc[enc_idx] * 2 + enc_dim <= dim_lim) and (extend or cur_ext_dim <= max_ext_dim):
                                # Select the minimum of all enclosures in the non-routing direction that satisfies
                                # the enclosure rules
                                if min_ext_dim is None or min_ext_dim > cur_ext_dim:
                                    min_ext_dim = cur_ext_dim

                        # If none of the enclosures in the list meet the rules, the current spacing rules cannot
                        # be used to create a valid via, so we continue on to the next set of spacing rules
                        if min_ext_dim is None:
                            break
                        # Otherwise record the computed via dimensions that pass all checks
                        else:
                            min_ext_dim = max(min_ext_dim, max_ext_dim)
                            mdim_list[idx] = [min_ext_dim, min_ext_dim]
                            mdim_list[idx][enc_idx] = dim_lim

                    # If we've found a valid via configuration immediately break out of the loop
                    if mdim_list[0] is not None and mdim_list[1] is not None:
                        # passed
                        opt_mdim_list = mdim_list
                        opt_nxy = (nx, ny)
                        opt_adim = (w_arr, h_arr)
                        opt_sp = (spx, spy)
                        break

                # If we've found a valid via array size immediately break out of the loop
                if opt_nxy is not None:
                    break

            # Select the best via out of all the passing via types. Vias are selected by choosing the
            # highest 'best_num'. This is calculated by multiplying the via array size by the via weight
            # Ties between vias are broken by minimizing drawn via area
            if opt_nxy is not None:
                opt_num = weight * opt_nxy[0] * opt_nxy[1]
                if (best_num is None or opt_num > best_num or
                        (opt_num == best_num and self._via_better(best_mdim_list, opt_mdim_list))):
                    best_num = opt_num
                    best_nxy = opt_nxy
                    best_mdim_list = opt_mdim_list
                    best_type = vtype
                    best_vdim = dim
                    best_sp = opt_sp
                    best_adim = opt_adim

        if best_num is None:
            return None
        return best_nxy, best_mdim_list, best_type, best_vdim, best_sp, best_adim

    def _via_better(self, mdim_list1, mdim_list2):
        """Returns true if the via in mdim_list1 has smaller area compared with via in mdim_list2"""
        res = self._resolution
        better = False
        for mdim1, mdim2 in zip(mdim_list1, mdim_list2):
            area1 = int(round(mdim1[0] / res)) * int(round(mdim1[1] / res))
            area2 = int(round(mdim2[0] / res)) * int(round(mdim2[1] / res))
            if area1 < area2:
                better = True
            elif area1 > area2:
                return False
        return better

    # noinspection PyMethodMayBeStatic
    def get_via_id(self, bot_layer, top_layer):
        """Returns the via ID string given bottom and top layer name.

        Defaults to "<bot_layer>_<top_layer>"

        Parameters
        ----------
        bot_layer : string
            the bottom layer name.
        top_layer : string
            the top layer name.

        Returns
        -------
        via_id : string
            the via ID string.
        """
        return '%s_%s' % (top_layer, bot_layer)

    def get_via_info(self, bbox, bot_layer, top_layer, bot_dir, bot_len=-1, top_len=-1,
                     extend=True, top_dir=None, **kwargs):
        """Create a via on the routing grid given the bounding box.

        Parameters
        ----------
        bbox : bag.layout.util.BBox
            the bounding box of the via.
        bot_layer : Union[str, Tuple[str, str]]
            the bottom layer name, or a tuple of layer name and purpose name.
            If purpose name not given, defaults to 'drawing'.
        top_layer : Union[str, Tuple[str, str]]
            the top layer name, or a tuple of layer name and purpose name.
            If purpose name not given, defaults to 'drawing'.
        bot_dir : str
            the bottom layer extension direction.  Either 'x' or 'y'
        bot_len : float
            length of bottom wire connected to this Via, in layout units.
            Used for length enhancement EM calculation.
        top_len : float
            length of top wire connected to this Via, in layout units.
            Used for length enhancement EM calculation.
        extend : bool
            True if via extension can be drawn outside of bounding box.
        top_dir : Optional[str]
            top layer extension direction.  Can force to extend in same direction as bottom.
        **kwargs :
            optional parameters for EM rule calculations, such as nominal temperature,
            AC rms delta-T, etc.

        Returns
        -------
        info : dict[string, any]
            A dictionary of via information, or None if no solution.  Should have the following:

            resistance : float
                The total via array resistance, in Ohms.
            idc : float
                The total via array maximum allowable DC current, in Amperes.
            iac_rms : float
                The total via array maximum allowable AC RMS current, in Amperes.
            iac_peak : float
                The total via array maximum allowable AC peak current, in Amperes.
            params : dict[str, any]
                A dictionary of via parameters.
            top_box : bag.layout.util.BBox
                the top via layer bounding box, including extensions.
            bot_box : bag.layout.util.BBox
                the bottom via layer bounding box, including extensions.

        """
        # remove purpose
        if isinstance(bot_layer, tuple):
            bot_layer = bot_layer[0]
        if isinstance(top_layer, tuple):
            top_layer = top_layer[0]
        bot_layer = bag.io.fix_string(bot_layer)
        top_layer = bag.io.fix_string(top_layer)

        bot_id = self.get_layer_id(bot_layer)
        bmtype = self.get_layer_type(bot_layer)
        tmtype = self.get_layer_type(top_layer)
        vname = self.get_via_name(bot_id)

        if not top_dir:
            top_dir = 'x' if bot_dir == 'y' else 'y'

        via_result = self.get_best_via_array(vname, bmtype, tmtype, bot_dir, top_dir,
                                             bbox.width, bbox.height, extend)
        if via_result is None:
            # no solution found
            return None

        (nx, ny), mdim_list, vtype, vdim, (spx, spy), (warr_norm, harr_norm) = via_result

        res = self.resolution
        xc_norm = bbox.xc_unit
        yc_norm = bbox.yc_unit

        wbot_norm = mdim_list[0][0]
        hbot_norm = mdim_list[0][1]
        wtop_norm = mdim_list[1][0]
        htop_norm = mdim_list[1][1]

        # OpenAccess Via can't handle even + odd enclosure, so we truncate.
        enc1_x = (wbot_norm - warr_norm) // 2 * res
        enc1_y = (hbot_norm - harr_norm) // 2 * res
        enc2_x = (wtop_norm - warr_norm) // 2 * res
        enc2_y = (htop_norm - harr_norm) // 2 * res

        # compute EM rule dimensions
        if bot_dir == 'x':
            bw, tw = hbot_norm * res, wtop_norm * res
        else:
            bw, tw = wbot_norm * res, htop_norm * res

        bot_xl_norm = xc_norm - wbot_norm // 2
        bot_yb_norm = yc_norm - hbot_norm // 2
        top_xl_norm = xc_norm - wtop_norm // 2
        top_yb_norm = yc_norm - htop_norm // 2

        bot_box = BBox(bot_xl_norm, bot_yb_norm, bot_xl_norm + wbot_norm,
                       bot_yb_norm + hbot_norm, res, unit_mode=True)
        top_box = BBox(top_xl_norm, top_yb_norm, top_xl_norm + wtop_norm,
                       top_yb_norm + htop_norm, res, unit_mode=True)

        idc, irms, ipeak = self.get_via_em_specs(vname, bot_layer, top_layer, via_type=vtype,
                                                 bm_dim=(bw, bot_len), tm_dim=(tw, top_len),
                                                 array=nx > 1 or ny > 1, **kwargs)

        params = {'id': self.get_via_id(bot_layer, top_layer),
                  'loc': (xc_norm * res, yc_norm * res),
                  'orient': 'R0',
                  'num_rows': ny,
                  'num_cols': nx,
                  'sp_rows': spy * res,
                  'sp_cols': spx * res,
                  # increase left/bottom enclusion if off-center.
                  'enc1': [enc1_x, enc1_x, enc1_y, enc1_y],
                  'enc2': [enc2_x, enc2_x, enc2_y, enc2_y],
                  'cut_width': vdim[0] * res,
                  'cut_height': vdim[1] * res,
                  }

        ntot = nx * ny
        return dict(
            resistance=0.0,
            idc=idc * ntot,
            iac_rms=irms * ntot,
            iac_peak=ipeak * ntot,
            params=params,
            top_box=top_box,
            bot_box=bot_box,
        )

    def design_resistor(self, res_type, res_targ, idc=0.0, iac_rms=0.0,
                        iac_peak=0.0, num_even=True, **kwargs):
        """Finds the optimal resistor dimension that meets the given specs.

        Assumes resistor length does not effect EM specs.

        Parameters
        ----------
        res_type : string
            the resistor type.
        res_targ : float
            target resistor, in Ohms.
        idc : float
            maximum DC current spec, in Amperes.
        iac_rms : float
            maximum AC RMS current spec, in Amperes.
        iac_peak : float
            maximum AC peak current spec, in Amperes.
        num_even : int
            True to return even number of resistors.
        **kwargs :
            optional EM spec calculation parameters.

        Returns
        -------
        num_par : int
            number of resistors needed in parallel.
        num_ser : int
            number of resistors needed in series.
        w : float
            width of a unit resistor, in meters.
        l : float
            length of a unit resistor, in meters.
        """
        resolution = self.resolution
        rsq = self.get_res_rsquare(res_type)
        wmin, wmax = self.get_res_width_bounds(res_type)
        lmin, lmax = self.get_res_length_bounds(res_type)
        min_nsq = self.get_res_min_nsquare(res_type)

        wmin_unit = int(round(wmin / resolution))
        wmax_unit = int(round(wmax / resolution))
        lmin_unit = int(round(lmin / resolution))
        lmax_unit = int(round(lmax / resolution))
        # make sure width is always even
        wmin_unit = -2 * (-wmin_unit // 2)
        wmax_unit = 2 * (wmax_unit // 2)

        # step 1: find number of parallel resistors and minimum resistor width.
        if num_even:
            npar_iter = BinaryIterator(2, None, step=2)
        else:
            npar_iter = BinaryIterator(1, None, step=1)
        while npar_iter.has_next():
            npar = npar_iter.get_next()
            res_targ_par = res_targ * npar
            idc_par = idc / npar
            iac_rms_par = iac_rms / npar
            iac_peak_par = iac_peak / npar
            res_idc, res_irms, res_ipeak = self.get_res_em_specs(res_type, wmax, **kwargs)
            if (0.0 < res_idc < idc_par or 0.0 < res_irms < iac_rms_par or
                    0.0 < res_ipeak < iac_peak_par):
                npar_iter.up()
            else:
                # This could potentially work, find width solution
                w_iter = BinaryIterator(wmin_unit, wmax_unit + 1, step=2)
                while w_iter.has_next():
                    wcur_unit = w_iter.get_next()
                    lcur_unit = int(math.ceil(res_targ_par / rsq * wcur_unit))
                    if lcur_unit < max(lmin_unit, int(math.ceil(min_nsq * wcur_unit))):
                        w_iter.down()
                    else:
                        tmp = self.get_res_em_specs(res_type, wcur_unit * resolution,
                                                    l=lcur_unit * resolution, **kwargs)
                        res_idc, res_irms, res_ipeak = tmp
                        if (0.0 < res_idc < idc_par or 0.0 < res_irms < iac_rms_par or
                                0.0 < res_ipeak < iac_peak_par):
                            w_iter.up()
                        else:
                            w_iter.save_info((wcur_unit, lcur_unit))
                            w_iter.down()

                w_info = w_iter.get_last_save_info()
                if w_info is None:
                    # no solution; we need more parallel resistors
                    npar_iter.up()
                else:
                    # solution!
                    npar_iter.save_info((npar, w_info[0], w_info[1]))
                    npar_iter.down()

        # step 3: fix maximum length violation by having resistor in series.
        num_par, wopt_unit, lopt_unit = npar_iter.get_last_save_info()
        wopt = wopt_unit * resolution
        if lopt_unit > lmax_unit:
            num_ser = -(-lopt_unit // lmax_unit)
            lopt = round(lopt_unit / num_ser / resolution) * resolution
        else:
            num_ser = 1
            lopt = lopt_unit * resolution

        # step 4: return answer
        return num_par, num_ser, wopt * self.layout_unit, lopt * self.layout_unit


class DummyTechInfo(TechInfo):
    """A dummy TechInfo class.

    Parameters
    ----------
    tech_params : dict[str, any]
        technology parameters dictionary.
    """

    def __init__(self, tech_params):
        TechInfo.__init__(self, 0.001, 1e-6, '', tech_params)

    def get_well_layers(self, sub_type):
        return []

    def get_implant_layers(self, mos_type, res_type=None):
        return []

    def get_threshold_layers(self, mos_type, threshold, res_type=None):
        return []

    def get_dnw_layers(self):
        # type: () -> List[Tuple[str, str]]
        return []

    def get_exclude_layer(self, layer_id):
        # type: (int) -> Tuple[str, str]
        """Returns the metal exclude layer"""
        return '', ''

    def get_dnw_margin_unit(self, dnw_mode):
        # type: (str) -> int
        return 0

    def get_res_metal_layers(self, layer_id):
        # type: (int) -> List[Tuple[str, str]]
        return []

    def get_metal_dummy_layers(self, layer_id):
        # type: (int) -> List[Tuple[str, str]]
        return []

    def add_cell_boundary(self, template, box):
        pass

    def draw_device_blockage(self, template):
        pass

    def get_via_drc_info(self, vname, vtype, mtype, mw_unit, is_bot):
        return (0, 0), [(0, 0)], [(0, 0)], [(0, 0)], (0, 0), [(0, 0)], None, None

    def get_min_space(self, layer_type, width, unit_mode=False, same_color=False):
        return 0

    def get_min_line_end_space(self, layer_type, width, unit_mode=False):
        return 0

    def get_min_length(self, layer_type, width):
        return 0.0

    def get_layer_id(self, layer_name):
        return -1

    def get_layer_name(self, layer_id):
        return ''

    def get_layer_type(self, layer_name):
        return ''

    def get_via_name(self, bot_layer_id):
        return ''

    def get_metal_em_specs(self, layer_name, w, l=-1, vertical=False, **kwargs):
        return float('inf'), float('inf'), float('inf')

    def get_via_em_specs(self, via_name, bm_layer, tm_layer, via_type='square',
                         bm_dim=(-1, -1), tm_dim=(-1, -1), array=False, **kwargs):
        return float('inf'), float('inf'), float('inf')

    def get_res_rsquare(self, res_type):
        return 0.0

    def get_res_width_bounds(self, res_type):
        return 0.0, 0.0

    def get_res_length_bounds(self, res_type):
        return 0.0, 0.0

    def get_res_min_nsquare(self, res_type):
        return 1.0

    def get_res_em_specs(self, res_type, w, l=-1, **kwargs):
        return float('inf'), float('inf'), float('inf')


class BagLayout(object):
    """This class contains layout information of a cell.

    Parameters
    ----------
    grid : :class:`bag.layout.routing.RoutingGrid`
        the routing grid instance.
    use_cybagoa : bool
        True to use cybagoa package to accelerate layout.
    """

    def __init__(self, grid, use_cybagoa=False):
        self._res = grid.resolution
        self._via_tech = grid.tech_info.via_tech_name
        self._pin_purpose = grid.tech_info.pin_purpose
        self._make_pin_rect = True
        self._inst_list = []  # type: List[Instance]
        self._inst_primitives = []  # type: List[InstanceInfo]
        self._rect_list = []  # type: List[Rect]
        self._via_list = []  # type: List[Via]
        self._via_primitives = []  # type: List[ViaInfo]
        self._pin_list = []  # type: List[PinInfo]
        self._path_list = []  # type: List[Path]
        self._polygon_list = []  # type: List[Polygon]
        self._blockage_list = []  # type: List[Blockage]
        self._boundary_list = []  # type: List[Boundary]
        self._used_inst_names = set()
        self._used_pin_names = set()
        self._raw_content = None
        self._is_empty = True
        self._finalized = False
        self._use_cybagoa = use_cybagoa

    @property
    def pin_purpose(self):
        """Returns the default pin layer purpose name."""
        return self._pin_purpose

    @property
    def is_empty(self):
        """Returns True if this layout is empty."""
        return self._is_empty

    def inst_iter(self):
        # type: () -> Iterator[Instance]
        return iter(self._inst_list)

    def finalize(self):
        # type: () -> None
        """Prevents any further changes to this layout.
        """
        self._finalized = True

        # get rectangles
        rect_list = []
        for obj in self._rect_list:
            if obj.valid:
                if not obj.bbox.is_physical():
                    print('WARNING: rectangle with non-physical bounding box found.', obj.layer)
                else:
                    obj_content = obj.content
                    rect_list.append(obj_content)

        # filter out invalid geometries
        path_list, polygon_list, blockage_list, boundary_list, via_list = [], [], [], [], []
        for targ_list, obj_list in ((path_list, self._path_list),
                                    (polygon_list, self._polygon_list),
                                    (blockage_list, self._blockage_list),
                                    (boundary_list, self._boundary_list),
                                    (via_list, self._via_list)):
            for obj in obj_list:
                if obj.valid:
                    targ_list.append(obj.content)

        # get via primitives
        via_list.extend(self._via_primitives)

        # get instances
        inst_list = []  # type: List[InstanceInfo]
        for obj in self._inst_list:
            if obj.valid:
                obj_content = self._format_inst(obj)
                inst_list.append(obj_content)

        self._raw_content = [inst_list,
                             self._inst_primitives,
                             rect_list,
                             via_list,
                             self._pin_list,
                             path_list,
                             blockage_list,
                             boundary_list,
                             polygon_list,
                             ]

        if (not inst_list and not self._inst_primitives and not rect_list and not blockage_list and
                not boundary_list and not via_list and not self._pin_list and not path_list and
                not polygon_list):
            self._is_empty = True
        else:
            self._is_empty = False

    def get_rect_bbox(self, layer):
        # type: (Union[str, Tuple[str, str]]) -> BBox
        """Returns the overall bounding box of all rectangles on the given layer.

        Note: currently this does not check primitive instances or vias.
        """
        if isinstance(layer, str):
            layer = (layer, 'drawing')

        box = BBox.get_invalid_bbox()
        for rect in self._rect_list:
            if layer == rect.layer:
                box = box.merge(rect.bbox_array.get_overall_bbox())

        for inst in self._inst_list:
            box = box.merge(inst.get_rect_bbox(layer))

        return box

    def get_masters_set(self):
        """Returns a set of all template master keys used in this layout."""
        return set((inst.master.key for inst in self._inst_list))

    def _get_unused_inst_name(self, inst_name):
        """Returns a new inst name."""
        if inst_name is None or inst_name in self._used_inst_names:
            cnt = 0
            inst_name = 'X%d' % cnt
            while inst_name in self._used_inst_names:
                cnt += 1
                inst_name = 'X%d' % cnt

        return inst_name

    def _format_inst(self, inst):
        # type: (Instance) -> InstanceInfo
        """Convert the given instance into dictionary representation."""
        content = inst.content
        inst_name = self._get_unused_inst_name(content.name)
        content.name = inst_name
        self._used_inst_names.add(inst_name)
        return content

    def get_content(self,  # type: BagLayout
                    lib_name,  # type: str
                    cell_name,  # type: str
                    rename_fun,  # type: Callable[[str], str]
                    ):
        # type: (...) -> Union[List[Any], Tuple[str, 'cybagoa.PyOALayout']]
        """returns a list describing geometries in this layout.

        Parameters
        ----------
        lib_name : str
            the layout library name.
        cell_name : str
            the layout top level cell name.
        rename_fun : Callable[[str], str]
            the layout cell renaming function.

        Returns
        -------
        content : Union[List[Any], Tuple[str, 'cybagoa.PyOALayout']]
            a list describing this layout, or PyOALayout if cybagoa package is enabled.
        """
        if not self._finalized:
            raise Exception('Layout is not finalized.')

        cell_name = rename_fun(cell_name)
        (inst_list, inst_prim_list, rect_list, via_list, pin_list,
         path_list, blockage_list, boundary_list, polygon_list) = self._raw_content

        # update library name and apply layout cell renaming on instances
        inst_tot_list = []
        for inst in inst_list:
            inst_temp = inst.copy()
            inst_temp['lib'] = lib_name
            inst_temp['cell'] = rename_fun(inst_temp['cell'])
            inst_tot_list.append(inst_temp)
        inst_tot_list.extend(inst_prim_list)

        if self._use_cybagoa and cybagoa is not None:
            encoding = bag.io.get_encoding()
            oa_layout = cybagoa.PyLayout(encoding)

            for obj in inst_tot_list:
                obj.pop('master_key', None)
                oa_layout.add_inst(**obj)
            for obj in rect_list:
                oa_layout.add_rect(**obj)
            for obj in via_list:
                oa_layout.add_via(**obj)
            for obj in pin_list:
                oa_layout.add_pin(**obj)
            for obj in path_list:
                oa_layout.add_path(**obj)
            for obj in blockage_list:
                oa_layout.add_blockage(**obj)
            for obj in boundary_list:
                oa_layout.add_boundary(**obj)
            for obj in polygon_list:
                oa_layout.add_polygon(**obj)

            return cell_name, oa_layout
        else:
            ans = [cell_name, inst_tot_list, rect_list, via_list, pin_list, path_list,
                   blockage_list, boundary_list, polygon_list]
            return ans

    def add_instance(self, instance):
        """Adds the given instance to this layout.

        Parameters
        ----------
        instance : bag.layout.objects.Instance
            the instance to add.
        """
        if self._finalized:
            raise Exception('Layout is already finalized.')

        # if isinstance(instance.nx, float) or isinstance(instance.ny, float):
        #     raise Exception('float nx/ny')

        self._inst_list.append(instance)

    def move_all_by(self, dx=0.0, dy=0.0, unit_mode=False):
        # type: (Union[float, int], Union[float, int], bool) -> None
        """Move all layout objects in this layout by the given amount.

        Parameters
        ----------
        dx : Union[float, int]
            the X shift.
        dy : Union[float, int]
            the Y shift.
        unit_mode : bool
            True if shift values are given in resolution units.
        """
        if self._finalized:
            raise Exception('Layout is already finalized.')

        for obj in chain(self._inst_list, self._inst_primitives, self._rect_list,
                         self._via_primitives, self._via_list, self._pin_list,
                         self._path_list, self._blockage_list, self._boundary_list,
                         self._polygon_list):
            obj.move_by(dx=dx, dy=dy, unit_mode=unit_mode)

    def add_instance_primitive(self,  # type: BagLayout
                               lib_name,  # type: str
                               cell_name,  # type: str
                               loc,  # type: Tuple[Union[float, int], Union[float, int]]
                               view_name='layout',  # type: str
                               inst_name=None,  # type: Optional[str]
                               orient="R0",  # type: str
                               num_rows=1,  # type: int
                               num_cols=1,  # type: int
                               sp_rows=0,  # type: Union[float, int]
                               sp_cols=0,  # type: Union[float, int]
                               params=None,  # type: Optional[Dict[str, Any]]
                               unit_mode=False,  # type: bool
                               **kwargs
                               ):
        """Adds a new (arrayed) primitive instance to this layout.

        Parameters
        ----------
        lib_name : str
            instance library name.
        cell_name : str
            instance cell name.
        loc : Tuple[Union[float, int], Union[float, int]]
            instance location.
        view_name : str
            instance view name.  Defaults to 'layout'.
        inst_name : Optional[str]
            instance name.  If None or an instance with this name already exists,
            a generated unique name is used.
        orient : str
            instance orientation.  Defaults to "R0"
        num_rows : int
            number of rows.  Must be positive integer.
        num_cols : int
            number of columns.  Must be positive integer.
        sp_rows : Union[float, int]
            row spacing.  Used for arraying given instance.
        sp_cols : Union[float, int]
            column spacing.  Used for arraying given instance.
        params : Optional[Dict[str, Any]]
            the parameter dictionary.  Used for adding pcell instance.
        unit_mode : bool
            True if distances are specified in resolution units.
        **kwargs :
            additional arguments.  Usually implementation specific.
        """
        if self._finalized:
            raise Exception('Layout is already finalized.')

        res = self._res
        if not unit_mode:
            loc = [round(loc[0] / res) * res,
                   round(loc[1] / res) * res]
            sp_rows = round(sp_rows / res) * res
            sp_cols = round(sp_cols / res) * res
        else:
            loc = [loc[0] * res, loc[1] * res]
            sp_rows *= res
            sp_cols *= res

        # get unique instance name
        inst_name = self._get_unused_inst_name(inst_name)
        self._used_inst_names.add(inst_name)

        inst_info = InstanceInfo(self._res, lib=lib_name,
                                 cell=cell_name,
                                 view=view_name,
                                 name=inst_name,
                                 loc=loc,
                                 orient=orient,
                                 num_rows=num_rows,
                                 num_cols=num_cols,
                                 sp_rows=sp_rows,
                                 sp_cols=sp_cols)

        # if isinstance(num_rows, float) or isinstance(num_cols, float):
        #     raise Exception('float nx/ny')

        if params is not None:
            inst_info.params = params
        inst_info.update(kwargs)

        self._inst_primitives.append(inst_info)

    def add_rect(self, rect):
        """Add a new (arrayed) rectangle.

        Parameters
        ----------
        rect : bag.layout.objects.Rect
            the rectangle object to add.
        """
        if self._finalized:
            raise Exception('Layout is already finalized.')

        self._rect_list.append(rect)

    def add_path(self, path):
        # type: (Path) -> None
        """Add a new path.

        Parameters
        ----------
        path : Path
            the path object to add.
        """
        if self._finalized:
            raise Exception('Layout is already finalized.')

        self._path_list.append(path)

    def add_polygon(self, polygon):
        # type: (Polygon) -> None
        """Add a new polygon.

        Parameters
        ----------
        polygon : Polygon
            the polygon object to add.
        """
        if self._finalized:
            raise Exception('Layout is already finalized.')

        self._polygon_list.append(polygon)

    def add_blockage(self, blockage):
        # type: (Blockage) -> None
        """Add a new blockage.

        Parameters
        ----------
        blockage : Blockage
            the blockage object to add.
        """
        if self._finalized:
            raise Exception('Layout is already finalized.')

        self._blockage_list.append(blockage)

    def add_boundary(self, boundary):
        # type: (Boundary) -> None
        """Add a new boundary.

        Parameters
        ----------
        boundary : Boundary
            the boundary object to add.
        """
        if self._finalized:
            raise Exception('Layout is already finalized.')

        self._boundary_list.append(boundary)

    def add_via(self, via):
        """Add a new (arrayed) via.

        Parameters
        ----------
        via : bag.layout.objects.Via
            the via object to add.
        """
        if self._finalized:
            raise Exception('Layout is already finalized.')

        # if isinstance(via.nx, float) or isinstance(via.ny, float):
        #     raise Exception('float nx/ny')

        self._via_list.append(via)

    def add_via_primitive(self, via_type, loc, num_rows=1, num_cols=1, sp_rows=0.0, sp_cols=0.0,
                          enc1=None, enc2=None, orient='R0', cut_width=None, cut_height=None,
                          arr_nx=1, arr_ny=1, arr_spx=0.0, arr_spy=0.0):
        """Adds a primitive via by specifying all parameters.

        Parameters
        ----------
        via_type : str
            the via type name.
        loc : Tuple[float, float]
            the via location as a two-element tuple.
        num_rows : int
            number of via cut rows.
        num_cols : int
            number of via cut columns.
        sp_rows : float
            spacing between via cut rows.
        sp_cols : float
            spacing between via cut columns.
        enc1 : list[float]
            a list of left, right, top, and bottom enclosure values on bottom layer.
            Defaults to all 0.
        enc2 : list[float]
            a list of left, right, top, and bottom enclosure values on top layer.
            Defaults. to all 0.
        orient : str
            orientation of the via.
        cut_width : float or None
            via cut width.  This is used to create rectangle via.
        cut_height : float or None
            via cut height.  This is used to create rectangle via.
        arr_nx : int
            number of columns.
        arr_ny : int
            number of rows.
        arr_spx : float
            column pitch.
        arr_spy : float
            row pitch.
        """
        if self._finalized:
            raise Exception('Layout is already finalized.')

        if arr_nx > 0 and arr_ny > 0:
            if enc1 is None:
                enc1 = [0.0, 0.0, 0.0, 0.0]
            if enc2 is None:
                enc2 = [0.0, 0.0, 0.0, 0.0]

            # if isinstance(arr_nx, float) or isinstance(arr_ny, float):
            #     raise Exception('float nx/ny')

            par = ViaInfo(self._res, id=via_type, loc=loc, orient=orient, num_rows=num_rows,
                          num_cols=num_cols,
                          sp_rows=sp_rows, sp_cols=sp_cols, enc1=enc1, enc2=enc2, )
            if cut_width is not None:
                par['cut_width'] = cut_width
            if cut_height is not None:
                par['cut_height'] = cut_height
            if arr_nx > 1 or arr_ny > 1:
                par['arr_nx'] = arr_nx
                par['arr_ny'] = arr_ny
                par['arr_spx'] = arr_spx
                par['arr_spy'] = arr_spy

            self._via_primitives.append(par)

    def add_pin(self, net_name, layer, bbox, pin_name=None, label=None):
        """Add a new pin.

        Parameters
        ----------
        net_name : str
            the net name associated with this pin.
        layer : string or (string, string)
            the layer name, or (layer, purpose) pair.
            if purpose is not specified, defaults to 'pin'.
        bbox : bag.layout.util.BBox
            the rectangle bounding box
        pin_name : str or None
            the pin name.  If None or empty, auto-generate from net name.
        label : str or None
            the pin label text.  If None or empty, will use net name as the text.
        """
        if self._finalized:
            raise Exception('Layout is already finalized.')

        if isinstance(layer, bytes):
            # interpret as unicode
            layer = layer.decode('utf-8')
        if isinstance(layer, str):
            layer = (layer, self._pin_purpose)
        else:
            layer = layer[0], layer[1]

        if not label:
            label = net_name

        pin_name = pin_name or net_name
        idx = 1
        while pin_name in self._used_pin_names:
            pin_name = '%s_%d' % (net_name, idx)
            idx += 1

        par = PinInfo(self._res, net_name=net_name,
                      pin_name=pin_name,
                      label=label,
                      layer=list(layer),
                      bbox=[[bbox.left, bbox.bottom], [bbox.right, bbox.top]],
                      make_rect=self._make_pin_rect)

        self._used_pin_names.add(pin_name)
        self._pin_list.append(par)

    def add_label(self, label, layer, bbox):
        """Add a new label.

        This is mainly used to add voltage text labels.

        Parameters
        ----------
        label : str
            the label text.
        layer : Union[str, Tuple[str, str]]
            the layer name, or (layer, purpose) pair.
            if purpose is not specified, defaults to 'pin'.
        bbox : bag.layout.util.BBox
            the rectangle bounding box
        """
        if self._finalized:
            raise Exception('Layout is already finalized.')

        if isinstance(layer, bytes):
            # interpret as unicode
            layer = layer.decode('utf-8')
        if isinstance(layer, str):
            layer = (layer, self._pin_purpose)
        else:
            layer = layer[0], layer[1]

        par = PinInfo(self._res, net_name='',
                      pin_name='',
                      label=label,
                      layer=list(layer),
                      bbox=[[bbox.left, bbox.bottom], [bbox.right, bbox.top]],
                      make_rect=False)

        self._pin_list.append(par)
