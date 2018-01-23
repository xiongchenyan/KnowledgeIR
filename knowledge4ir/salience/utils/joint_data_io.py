"""
data io
"""
import json

import torch
from torch.autograd import Variable
from knowledge4ir.utils import (
    term2lm,
    SPOT_FIELD,
    EVENT_SPOT_FIELD,
    body_field,
    title_field,
    abstract_field,
    salience_gold,
    adjacent_field
)
from knowledge4ir.salience.utils.data_io import DataIO
import numpy as np
import logging
from traitlets import (
    Int,
    Unicode,
    List,
)
import math

use_cuda = torch.cuda.is_available()


class EventDataIO(DataIO):
    event_spot_field = Unicode(EVENT_SPOT_FIELD, help='event spot field').tag(
        config=True)
    adjacent_field = Unicode(adjacent_field, help='adjacent field').tag(
        config=True)
    max_e_per_d = Int(200, help='max entity per doc').tag(config=True)
    entity_vocab_size = Int(help='vocabulary size of entity').tag(config=True)
    e_feature_dim = Int(help='entity feature dimension').tag(config=True)
    evm_feature_dim = Int(help='event feature dimension').tag(config=True)

    def __init__(self, **kwargs):
        super(EventDataIO, self).__init__(**kwargs)

    def _data_config(self):
        h_joint_target_group = {
            'event_raw': ['mtx_e', 'mtx_score', 'label'],
            'event_feature': ['mtx_e', 'mtx_score', 'ts_feature', 'label'],
            'joint_raw': ['mtx_e', 'mtx_score', 'label'],
            'joint_feature': ['mtx_e', 'mtx_score', 'ts_feature', 'label'],
            'joint_graph': ['mtx_e', 'mtx_evm', 'ts_args', 'mtx_arg_length',
                            'mtx_score', 'ts_feature', 'label', 'ts_laplacian']
        }
        self.h_target_group.update(h_joint_target_group)

        h_joint_data_meta = {
            'mtx_evm': {'dim': 2, 'd_type': 'Int'},
            'ts_args': {'dim': 3, 'd_type': 'Int'},
            'ts_arg_mask': {'dim': 3, 'd_type': 'Float'},
            'mtx_arg_length': {'dim': 2, 'd_type': 'Int'},
            'ts_laplacian': {'dim': 3, 'd_type': 'Float'},
        }
        self.h_data_meta.update(h_joint_data_meta)

        self.h_data_mask = {
            'joint_raw': ['mtx_e'],
            'joint_feature': ['mtx_e'],
            'joint_graph': ['ts_args', 'mtx_e', 'mtx_evm'],
        }

        # Natural NP data. Different padding; Different conversion.
        self.h_np_data = {
            'ts_laplacian': {'dim': 2}
        }

    def is_empty_line(self, line):
        if self.group_name.startswith('event'):
            h = json.loads(line)
            l_s = h[self.event_spot_field].get(self.content_field, {}).get(
                'salience')
            return not l_s
        if self.group_name.startswith('joint'):
            h = json.loads(line)
            l_s = h[self.event_spot_field].get(self.content_field, {}).get(
                'salience')
            empty_entity = super(EventDataIO, self).is_empty_line(line)
            return (not l_s) and empty_entity
        else:
            return super(EventDataIO, self).is_empty_line(line)

    def parse_data(self, l_line):
        l_data = []
        while len(l_data) < len(self.l_target_data):
            l_data.append([])

        h_parsed_data = dict(zip(self.l_target_data, l_data))
        for line in l_line:
            h_info = json.loads(line)
            if self.group_name.startswith('event'):
                h_this_data, _ = self._parse_event(h_info)
            elif self.group_name.startswith('joint'):
                if self.group_name == 'joint_graph':
                    h_this_data = self._parse_graph(h_info)
                else:
                    h_this_data = self._parse_joint(h_info)
            else:
                logging.error("Input group is not event related.")
                raise NotImplementedError

            for key in h_parsed_data.keys():
                assert key in h_this_data
                h_parsed_data[key].append(h_this_data[key])

        h_parsed_data = self._canonicalize_data(h_parsed_data)

        return h_parsed_data, h_parsed_data['label']

    def _canonicalize_data(self, h_parsed_data):
        for key in h_parsed_data:
            if key in self.h_np_data:
                dim = self.h_np_data[key]['dim']
                padded = self._pad_np(h_parsed_data[key], dim)
            else:
                dim = self.h_data_meta[key]['dim']
                padded = self._padding(h_parsed_data[key], dim)

            h_parsed_data[key] = padded

        # Compute masks from the padded value.
        mask_data = {}
        for key in self.h_data_mask[self.group_name]:
            data = h_parsed_data[key]
            if not self._is_empty(data, self.h_data_meta[key]['dim']):
                mask = self._pad_mask(data, key)
                mask_data[key] = self._data_to_variable(mask, data_type='Float')
            else:
                # Empty data will have empty mask.
                mask_data[key] = None

        for key in h_parsed_data:
            if key in self.h_np_data:
                h_parsed_data[key] = self._np_data_to_variable(
                    h_parsed_data[key]
                )
            else:
                dim = self.h_data_meta[key]['dim']
                data = h_parsed_data[key]

                if not self._is_empty(data, dim):
                    h_parsed_data[key] = self._data_to_variable(
                        h_parsed_data[key],
                        data_type=self.h_data_meta[key]['d_type']
                    )
                else:
                    h_parsed_data[key] = None

        h_parsed_data['masks'] = mask_data

        return h_parsed_data

    def _np_data_to_variable(self, list_data):
        v = Variable(torch.from_numpy(np.stack(list_data)).float())
        if use_cuda:
            v = v.cuda()
        return v

    def _pad_np(self, data, dim=2, default_value=0):
        if dim == 2:
            return self.two_d_np_padding(data, default_value)
        else:
            raise NotImplementedError

    @classmethod
    def two_d_np_padding(cls, data, default_value):
        max_row = 0
        max_col = 0
        for mtx in data:
            row, col = mtx.shape
            if row > max_row:
                max_row = row
            if col > max_col:
                max_col = col

        padded_data = []
        for mtx in data:
            row, col = mtx.shape
            row_pad = max_row - row
            col_pad = max_col - col
            padded = np.lib.pad(mtx, ((0, row_pad), (0, col_pad)), 'constant',
                                constant_values=default_value)
            padded_data.append(padded)

        return padded_data

    def _pad_mask(self, padded_data, key):
        if self.h_data_meta[key]['dim'] == 2:
            return self._2d_pad_mask(padded_data)
        elif self.h_data_meta[key]['dim'] == 3:
            return self._3d_pad_mask(padded_data)
        else:
            logging.error("No implementation for %d "
                          "masking." % self.h_data_meta[key]['dim'])
            raise NotImplementedError

    @classmethod
    def _2d_pad_mask(cls, matrix, pad_value=0):
        mask = []
        for row in matrix:
            mask.append([0 if e == pad_value else 1 for e in row])
        return mask

    @classmethod
    def _3d_pad_mask(cls, tensor, pad_value=0):
        mask = [[] for _ in range(len(tensor))]
        for index, matrix in enumerate(tensor):
            for row in matrix:
                mask[index].append([0 if e == pad_value else 1 for e in row])
        return mask

    def _parse_graph(self, h_info):
        """
        io with events and entities with their corresponding feature matrices.
        This will combine the event and entity embedding
        """
        # Note that we didn't pad entity and event separately.
        # This is currently fine using the kernel models.
        h_entity_res = self._parse_entity(h_info)
        l_e = h_entity_res['mtx_e']
        l_e_tf = h_entity_res['mtx_score']
        l_e_label = h_entity_res['label']
        ll_e_feat = h_entity_res['ts_feature']

        h_event_res, freq_mask = self._parse_event(h_info)
        l_evm = h_event_res['mtx_e']
        l_evm_tf = h_event_res['mtx_score']
        l_evm_label = h_event_res['label']
        ll_evm_feat = h_event_res['ts_feature']
        ll_args = h_event_res['mtx_args']

        # Shift event index after entities.
        l_evm = [e + self.entity_vocab_size for e in l_evm]

        # # Add offset for the empty entity.
        # l_e = [e + 1 for e in l_e]
        # l_evm = [e + 1 for e in l_evm]

        l_arg_length = [len(l) for l in ll_args]

        l_label_all = l_e_label + l_evm_label
        l_tf_all = l_e_tf + l_evm_tf

        if self.e_feature_dim and self.evm_feature_dim:
            # print 'Combining %d e feat and %d evm feat for %d events' % (
            #     len(ll_e_feat), len(ll_evm_feat), len(l_evm))
            ll_feat_all = self._combine_features(ll_e_feat, ll_evm_feat,
                                                 self.e_feature_dim,
                                                 self.evm_feature_dim)
        else:
            ll_feat_all = []

        mtx_adjacent = self._compute_laplacian(ll_args, l_e)

        h_res = {
            'mtx_e': l_e,
            'mtx_evm': l_evm,
            'ts_args': ll_args,
            'ts_laplacian': mtx_adjacent,
            'mtx_arg_length': l_arg_length,
            'mtx_score': l_tf_all,
            'ts_feature': ll_feat_all,
            'label': l_label_all,
        }
        return h_res

    def _compute_laplacian(self, ll_args, l_e):
        h_e = dict([(e, i) for i, e in enumerate(l_e)])
        dim = len(l_e) + len(ll_args)

        # Initialize with self loop graph.
        adjacent = np.eye(dim)

        # Only add self link to entities.
        ds = [1.0] * len(l_e)
        for index, l_args in enumerate(ll_args):
            row = index + len(l_e)
            for arg in l_args:
                # Some error in data processing cause this.
                if arg in h_e:
                    adjacent[row, h_e[arg]] = 1
            ds.append(1.0 / math.sqrt(len(l_args) + 1))

        rcpr_sqrt_degree = np.diag(ds)
        return rcpr_sqrt_degree * adjacent * rcpr_sqrt_degree

    def _parse_joint(self, h_info):
        """
        io with events and entities with their corresponding feature matrices.
        When e_feature_dim + evm_feature_dim = 0, it will fall back to raw io,
        a tf matrix will be computed instead.
        """
        # Note that we didn't pad entity and event separately.
        # This is currently fine using the kernel models.
        h_entity_res = self._parse_entity(h_info)
        l_e = h_entity_res['mtx_e']
        l_e_tf = h_entity_res['mtx_score']
        l_e_label = h_entity_res['label']
        ll_e_feat = h_entity_res['ts_feature']

        h_event_res, freq_mask = self._parse_event(h_info)
        l_evm = h_event_res['mtx_e']
        l_evm_tf = h_event_res['mtx_score']
        l_evm_label = h_event_res['label']
        ll_evm_feat = h_event_res['ts_feature']

        # shift the event id by an offset so entity and event use different ids.
        l_e_all = l_e + [e + self.entity_vocab_size for e in l_evm]
        l_tf_all = l_e_tf + l_evm_tf
        l_label_all = l_e_label + l_evm_label

        if self.e_feature_dim and self.evm_feature_dim:
            ll_feat_all = self._combine_features(ll_e_feat, ll_evm_feat,
                                                 self.e_feature_dim,
                                                 self.evm_feature_dim)
        else:
            ll_feat_all = []

        h_res = {
            'mtx_e': l_e_all,
            'mtx_score': l_tf_all,
            'ts_feature': ll_feat_all,
            'label': l_label_all,
        }
        return h_res

    def _parse_event(self, h_info):
        event_spots = h_info.get(self.event_spot_field, {}).get(
            self.content_field, {})

        l_h = event_spots.get('sparse_features', {}).get('LexicalHead', [])
        ll_feature = event_spots.get('features', [])
        # Take label from salience field.
        test_label = event_spots.get(self.salience_label_field, [0] * len(l_h))
        l_label = [1 if label == 1 else -1 for label in test_label]

        # Take a subset of event features only (others doesn't work).
        # We put -2 to the first position because it is frequency.
        # The reorganized features are respectively:
        # headcount, sentence loc, event voting, entity voting,
        # ss entity vote aver, ss entity vote max, ss entity vote min
        ll_feature = [l[-2:] + l[-3:-2] + l[9:13] for l in ll_feature]

        z = float(sum([item[0] for item in ll_feature]))
        l_tf = [item[0] / z for item in ll_feature]

        # Now take the most frequent events based on the feature. Here we
        # assume the first element in the feature is the frequency count.
        most_freq_indices = self._get_frequency_mask(ll_feature,
                                                     self.max_e_per_d)
        l_h = self._apply_mask(l_h, most_freq_indices)
        ll_feature = self._apply_mask(ll_feature, most_freq_indices)
        l_label = self._apply_mask(l_label, most_freq_indices)
        l_tf = self._apply_mask(l_tf, most_freq_indices)

        m_args = h_info.get(self.adjacent_field, [])
        m_args_masked = self._apply_mask(m_args, most_freq_indices)

        assert len(ll_feature) == len(l_h)
        assert len(l_h) == len(l_tf)
        assert len(l_h) == len(l_label)

        h_res = {
            'mtx_e': l_h,
            'mtx_score': l_tf,
            'ts_feature': ll_feature,
            'label': l_label,
            'mtx_args': m_args_masked
        }
        return h_res, most_freq_indices

    @classmethod
    def _combine_features(cls, ll_feature_e, ll_feature_evm, e_dim, evm_dim,
                          filler=0):
        e_pads = [filler] * e_dim
        evm_pads = [filler] * evm_dim

        for i in xrange(len(ll_feature_e)):
            if ll_feature_e[i]:
                ll_feature_e[i] = ll_feature_e[i] + evm_pads
            else:
                ll_feature_e[i] = e_pads + evm_pads
        for i in xrange(len(ll_feature_evm)):
            if ll_feature_evm[i]:
                ll_feature_evm[i] = e_pads + ll_feature_evm[i]
            else:
                ll_feature_evm[i] = e_pads + evm_pads

        return ll_feature_e + ll_feature_evm

    @classmethod
    def _get_frequency_mask(cls, ll_feature, max_e_per_d):
        if max_e_per_d is None:
            return range(len(ll_feature))
        if not ll_feature:
            return set()

        sorted_features = sorted(enumerate(ll_feature), key=lambda x: x[1][0],
                                 reverse=True)
        return set(zip(*sorted_features[:max_e_per_d])[0])

    @classmethod
    def _apply_mask(cls, l, mask):
        masked = []
        for i, e in enumerate(l):
            if i in mask:
                masked.append(e)
        return masked


if __name__ == '__main__':
    """
    unit test tbd
    """
    import sys
    from knowledge4ir.utils import (
        set_basic_log,
        load_py_config,
    )
    from traitlets.config import Configurable
    from traitlets import (
        Unicode,
        List,
        Int
    )

    set_basic_log()


    class IOTester(Configurable):
        in_name = Unicode(help='in data test').tag(config=True)
        io_func = Unicode('uw', help='io function to test').tag(config=True)
