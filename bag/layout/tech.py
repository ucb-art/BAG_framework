# -*- coding: utf-8 -*-

from typing import List, Tuple, Union, Optional, Callable, TYPE_CHECKING

import abc

from .core import TechInfo

if TYPE_CHECKING:
    from bag.layout.util import BBox
    from bag.layout.template import TemplateBase


class TechInfoConfig(TechInfo, metaclass=abc.ABCMeta):
    """An implementation of TechInfo that implements most methods with a technology file."""
    def __init__(self, config, tech_params, mos_entry_name='mos'):
        TechInfo.__init__(self, config['resolution'], config['layout_unit'],
                          config['tech_lib'], tech_params)

        self.config = config
        self._mos_entry_name = mos_entry_name
        self.idc_temp = tech_params['layout']['em']['dc_temp']
        self.irms_dt = tech_params['layout']['em']['rms_dt']

    @abc.abstractmethod
    def get_metal_em_specs(self, layer_name, w, l=-1, vertical=False, **kwargs):
        return float('inf'), float('inf'), float('inf')

    @abc.abstractmethod
    def get_via_em_specs(self, via_name, bm_layer, tm_layer, via_type='square',
                         bm_dim=(-1, -1), tm_dim=(-1, -1), array=False, **kwargs):
        return float('inf'), float('inf'), float('inf')

    @abc.abstractmethod
    def get_res_em_specs(self, res_type, w, l=-1, **kwargs):
        return float('inf'), float('inf'), float('inf')

    @abc.abstractmethod
    def add_cell_boundary(self, template, box):
        # type: (TemplateBase, BBox) -> None
        pass

    @abc.abstractmethod
    def draw_device_blockage(self, template):
        # type: (TemplateBase) -> None
        pass

    @abc.abstractmethod
    def get_via_arr_enc(self, vname, vtype, mtype, mw_unit, is_bot):
        # type: (...) -> Tuple[Optional[List[Tuple[int, int]]], Optional[Callable[[int, int], bool]]]
        return None, None

    @property
    def pin_purpose(self):
        return self.config.get('pin_purpose', 'pin')

    def get_via_types(self, bmtype, tmtype):
        default = [('square', 1), ('vrect', 2), ('hrect', 2)]
        if 'via_type_order' in self.config:
            table = self.config['via_type_order']
            return table.get((bmtype, tmtype), default)
        return default

    def get_well_layers(self, sub_type):
        # type: (str) -> List[Tuple[str, str]]
        return self.config['well_layers'][sub_type]

    def get_implant_layers(self, mos_type, res_type=None):
        # type: (str, Optional[str]) -> List[Tuple[str, str]]
        if res_type is None:
            table = self.config[self._mos_entry_name]
        else:
            table = self.config['resistor']

        return list(table['imp_layers'][mos_type].keys())

    def get_threshold_layers(self, mos_type, threshold, res_type=None):
        # type: (str, str, Optional[str]) -> List[Tuple[str, str]]
        if res_type is None:
            table = self.config[self._mos_entry_name]
        else:
            table = self.config['resistor']

        return list(table['thres_layers'][mos_type][threshold].keys())

    def get_exclude_layer(self, layer_id):
        # type: (int) -> Tuple[str, str]
        """Returns the metal exclude layer"""
        return self.config['metal_exclude_table'][layer_id]

    def get_dnw_margin_unit(self, dnw_mode):
        # type: (str) -> int
        return self.config['dnw_margins'][dnw_mode]

    def get_dnw_layers(self):
        # type: () -> List[Tuple[str, str]]
        return self.config[self._mos_entry_name]['dnw_layers']

    def get_res_metal_layers(self, layer_id):
        # type: (int) -> List[Tuple[str, str]]
        return self.config['res_metal_layer_table'][layer_id]

    def get_metal_dummy_layers(self, layer_id):
        # type: (int) -> List[Tuple[str, str]]
        return self.config['metal_dummy_table'][layer_id]

    def use_flip_parity(self):
        # type: () -> bool
        return self.config['use_flip_parity']

    def get_layer_name(self, layer_id):
        # type: (int) -> str
        name_dict = self.config['layer_name']
        return name_dict[layer_id]

    def get_layer_id(self, layer_name):
        # type: (str) -> int
        for key, val in self.config['layer_name'].items():
            if val == layer_name:
                return key
        raise ValueError('Unknown layer: %s' % layer_name)

    def get_layer_type(self, layer_name):
        # type: (str) -> str
        type_dict = self.config['layer_type']
        return type_dict[layer_name]

    def get_idc_scale_factor(self, temp, mtype, is_res=False):
        # type: (float, str, bool) -> float
        if is_res:
            mtype = 'res'
        idc_em_scale = self.config['idc_em_scale']
        if mtype in idc_em_scale:
            idc_params = idc_em_scale[mtype]
        else:
            idc_params = idc_em_scale['default']

        temp_list = idc_params['temp']
        scale_list = idc_params['scale']

        for temp_test, scale in zip(temp_list, scale_list):
            if temp <= temp_test:
                return scale
        return scale_list[-1]

    def get_via_name(self, bot_layer_id):
        # type: (int) -> str
        return self.config['via_name'][bot_layer_id]

    def get_via_id(self, bot_layer, top_layer):
        # type: (str, str) -> str
        return self.config['via_id'][(bot_layer, top_layer)]

    def get_via_drc_info(self, vname, vtype, mtype, mw_unit, is_bot):
        via_config = self.config['via']
        if vname not in via_config:
            raise ValueError('Unsupported vname %s' % vname)

        via_config = via_config[vname]
        if vtype.startswith('vrect') and vtype not in via_config:
            # trying vertical rectangle via, but it does not exist,
            # so try rotating horizontal rectangle instead
            rotate = True
            vtype2 = 'hrect' + vtype[5:]
        else:
            rotate = False
            vtype2 = vtype
        if vtype2 not in via_config:
            raise ValueError('Unsupported vtype %s' % vtype2)

        via_config = via_config[vtype2]

        dim = via_config['dim']
        sp = via_config['sp']
        sp2_list = via_config.get('sp2', None)
        sp3_list = via_config.get('sp3', None)
        sp6_list = via_config.get('sp6', None)

        if not is_bot or via_config['bot_enc'] is None:
            enc_data = via_config['top_enc']
        else:
            enc_data = via_config['bot_enc']

        enc_w_list = enc_data['w_list']
        enc_list = enc_data['enc_list']

        enc_cur = []
        for mw_max, enc in zip(enc_w_list, enc_list):
            if mw_unit <= mw_max:
                enc_cur = enc
                break

        arr_enc, arr_test_tmp = self.get_via_arr_enc(vname, vtype, mtype, mw_unit, is_bot)
        arr_test = arr_test_tmp

        if rotate:
            sp = sp[1], sp[0]
            dim = dim[1], dim[0]
            enc_cur = [(yv, xv) for xv, yv in enc_cur]
            if sp2_list is not None:
                sp2_list = [(spy, spx) for spx, spy in sp2_list]
            if sp3_list is not None:
                sp3_list = [(spy, spx) for spx, spy in sp3_list]
            if sp6_list is not None:
                sp6_list = [(spy, spx) for spx, spy in sp6_list]
            if arr_enc is not None:
                arr_enc = [(yv, xv) for xv, yv in arr_enc]
            if arr_test_tmp is not None:
                def arr_test(nrow, ncol):
                    return arr_test_tmp(ncol, nrow)

        return sp, sp2_list, sp3_list, sp6_list, dim, enc_cur, arr_enc, arr_test

    def _space_helper(self, config_name, layer_type, width):
        sp_min_config = self.config[config_name]
        if layer_type not in sp_min_config:
            raise ValueError('Unsupported layer type: %s' % layer_type)

        sp_min_config = sp_min_config[layer_type]
        w_list = sp_min_config['w_list']
        sp_list = sp_min_config['sp_list']

        for w, sp in zip(w_list, sp_list):
            if width <= w:
                return sp
        return None

    def get_min_space_unit(self, layer_type, w_unit, same_color=False):
        # type: (str, int, bool) -> int
        if not same_color or 'sp_sc_min' not in self.config:
            config_name = 'sp_min'
        else:
            config_name = 'sp_sc_min'

        return self._space_helper(config_name, layer_type, w_unit)

    def get_min_line_end_space_unit(self, layer_type, w_unit):
        return self._space_helper('sp_le_min', layer_type, w_unit)

    def get_min_space(self, layer_type, width, unit_mode=False, same_color=False):
        # type: (str, float, bool, bool) -> Union[float, int]
        res = self.config['resolution']
        if not unit_mode:
            width = int(round(width / res))

        ans = self.get_min_space_unit(layer_type, width, same_color=same_color)

        if unit_mode:
            return ans
        return ans * res

    def get_min_line_end_space(self, layer_type, width, unit_mode=False):
        # type: (str, float, bool) -> Union[float, int]
        res = self.config['resolution']
        if not unit_mode:
            width = int(round(width / res))

        ans = self.get_min_line_end_space_unit(layer_type, width)

        if unit_mode:
            return ans
        return ans * res

    def layer_id_to_type(self, layer_id):
        name_dict = self.config['layer_name']
        type_dict = self.config['layer_type']
        return type_dict[name_dict[layer_id]]

    def get_min_length_unit(self, layer_type, w_unit):
        len_min_config = self.config['len_min']
        if layer_type not in len_min_config:
            raise ValueError('Unsupported layer type: %s' % layer_type)

        w_list = len_min_config[layer_type]['w_list']
        w_al_list = len_min_config[layer_type]['w_al_list']
        md_list = len_min_config[layer_type]['md_list']
        md_al_list = len_min_config[layer_type]['md_al_list']

        # get minimum length from width spec
        l_unit = 0
        for w, (area, len_min) in zip(w_list, w_al_list):
            if w_unit <= w:
                l_unit = max(len_min, -(-area // w_unit))
                break

        # check maximum dimension spec
        for max_dim, (area, len_min) in zip(reversed(md_list), reversed(md_al_list)):
            if max(w_unit, l_unit) > max_dim:
                return l_unit
            l_unit = max(l_unit, len_min, -(-area // w_unit))

        return -(-l_unit // 2) * 2

    def get_min_length(self, layer_type, width):
        res = self.resolution
        w_unit = int(round(width / res))
        return res * self.get_min_length_unit(layer_type, w_unit)

    def get_res_rsquare(self, res_type):
        return self.config['resistor']['info'][res_type]['rsq']

    def get_res_width_bounds(self, res_type):
        return self.config['resistor']['info'][res_type]['w_bounds']

    def get_res_length_bounds(self, res_type):
        return self.config['resistor']['info'][res_type]['l_bounds']

    def get_res_min_nsquare(self, res_type):
        return self.config['resistor']['info'][res_type]['min_nsq']
