import torch
import torch.nn as nn
from torch.autograd import Variable
from knowledge4ir.salience.base import SalienceBaseModel, KernelPooling
from knowledge4ir.salience.knrm_vote import KNRM
from knowledge4ir.salience.crf_model import (
    LinearKernelCRF,
)
from knowledge4ir.salience.masked_knrm_vote import (
    MaskKNRM,
)
import logging
import json
import torch.nn.functional as F
import numpy as np

from knowledge4ir.salience.utils.debugger import Debugger

use_cuda = torch.cuda.is_available()

import pickle


class MaskKernelCrf(LinearKernelCRF):
    def __init__(self, para, ext_data=None):
        super(MaskKernelCrf, self).__init__(para, ext_data)
        self.use_mask = para.use_mask
        self.event_labels_only = para.event_labels_only

        if self.use_mask:
            logging.info('Running model with masking on empty slots.')
        else:
            logging.info('Running model without masking.')

    def forward(self, h_packed_data):
        mtx_e = h_packed_data['mtx_e']
        ts_feature = h_packed_data['ts_feature']

        if ts_feature.size()[-1] != self.node_feature_dim:
            logging.error('feature shape: %s != feature dim [%d]',
                          json.dumps(ts_feature.size()), self.node_feature_dim)
        assert ts_feature.size()[-1] == self.node_feature_dim
        if mtx_e.size()[:2] != ts_feature.size()[:2]:
            logging.error(
                'e mtx and feature tensor shape do not match: %s != %s',
                json.dumps(mtx_e.size()), json.dumps(ts_feature.size()))
        assert mtx_e.size()[:2] == ts_feature.size()[:2]

        node_score = F.tanh(self.node_lr(ts_feature))

        # frequency is the first dim of feature, always
        # mtx_score = ts_feature.narrow(-1, 0, 1).squeeze(-1)
        mtx_score = h_packed_data['mtx_score']

        h_mid_data = {
            "mtx_e": mtx_e,
            "mtx_score": mtx_score
        }

        if self.use_mask:
            mtx_e_mask = h_packed_data['masks']['mtx_e']
            mtx_embedding = self.embedding(mtx_e)
            mtx_masked_embedding = mtx_embedding * mtx_e_mask.unsqueeze(-1)
            kp_mtx = self._kernel_scores(mtx_masked_embedding, mtx_score)
            knrm_res = self.linear(kp_mtx).squeeze(-1)
        else:
            knrm_res = super(LinearKernelCRF, self).forward(h_mid_data)

        mixed_knrm = torch.cat((knrm_res.unsqueeze(-1), node_score), -1)
        output = self.linear_combine(mixed_knrm).squeeze(-1)

        if self.event_labels_only:
            # mask to keep only event outputs.
            evm_mask = h_packed_data['mtx_evm_mask']
            output = output * evm_mask

        return output


class StructEventKernelCRF(MaskKNRM):
    def __init__(self, para, ext_data=None):
        super(StructEventKernelCRF, self).__init__(para, ext_data)
        self.embedding_dim = para.embedding_dim
        self.node_feature_dim = para.node_feature_dim
        self.node_lr = nn.Linear(self.node_feature_dim, 1, bias=False)

        logging.info('node feature dim %d', self.node_feature_dim)

        self.use_mask = para.use_mask
        self.event_labels_only = para.event_labels_only

        if self.use_mask:
            logging.info('Running model with masking on empty slots.')
        else:
            logging.info('Running model without masking.')

        if use_cuda:
            self.node_lr.cuda()

    def forward(self, h_packed_data):
        ts_feature = h_packed_data['ts_feature']

        if ts_feature.size()[-1] != self.node_feature_dim:
            logging.error('feature shape: %s != feature dim [%d]',
                          json.dumps(ts_feature.size()), self.node_feature_dim)
        assert ts_feature.size()[-1] == self.node_feature_dim

        output = self.compute_score(h_packed_data)

        if self.event_labels_only:
            # mask to keep only event outputs.
            evm_mask = h_packed_data['mtx_evm_mask']
            output = output * evm_mask

        return output

    def event_embedding(self, mtx_evm, ts_args, mtx_arg_length, ts_arg_mask):
        return self.embedding(mtx_evm)

    def compute_score(self, h_packed_data):
        mtx_e, mtx_e_mask, mtx_score, node_score = self.get_raw_features(
            h_packed_data)

        if self.use_mask:
            knrm_res = self._forward_kernel_with_mask_and_features(
                mtx_e_mask, mtx_e, mtx_score, node_score)
        else:
            knrm_res = self._forward_kernel_with_features(mtx_e, mtx_score,
                                                          node_score)
        return knrm_res

    def get_raw_features(self, h_packed_data):
        ts_feature = h_packed_data['ts_feature']
        if ts_feature.size()[-1] != self.node_feature_dim:
            logging.error('feature shape: %s != feature dim [%d]',
                          json.dumps(ts_feature.size()), self.node_feature_dim)
        assert ts_feature.size()[-1] == self.node_feature_dim

        mtx_e = h_packed_data['mtx_e']
        mtx_evm = h_packed_data['mtx_evm']

        masks = h_packed_data['masks']
        mtx_e_mask = masks['mtx_e']
        mtx_evm_mask = masks['mtx_evm']

        mtx_e_embedding = self.embedding(mtx_e)
        if mtx_evm is None:
            # For documents without events.
            combined_mtx_e = mtx_e_embedding
            combined_mtx_e_mask = mtx_e_mask
        else:
            ts_args = h_packed_data['ts_args']
            mtx_arg_length = h_packed_data['mtx_arg_length']
            ts_arg_mask = masks['ts_args']
            mtx_evm_embedding = self.event_embedding(mtx_evm, ts_args,
                                                     mtx_arg_length,
                                                     ts_arg_mask)

            combined_mtx_e = torch.cat((mtx_e_embedding, mtx_evm_embedding), 1)
            combined_mtx_e_mask = torch.cat((mtx_e_mask, mtx_evm_mask), 1)

        node_score = F.tanh(self.node_lr(ts_feature))
        mtx_score = h_packed_data['mtx_score']

        return combined_mtx_e, combined_mtx_e_mask, mtx_score, node_score

    def _argument_sum(self, ts_args, ts_arg_mask):
        l_arg_embedding = []

        for mtx_args, mask in zip(ts_args, ts_arg_mask):
            mtx_args_embedding = self.embedding(mtx_args)
            masked_embedding = mtx_args_embedding * mask.unsqueeze(-1)
            arg_embedding_sum = masked_embedding.sum(1)
            l_arg_embedding.append(arg_embedding_sum)
        return torch.stack(l_arg_embedding)

    def save_model(self, output_name):
        logging.info('saving knrm embedding and linear weights to [%s]',
                     output_name)
        super(StructEventKernelCRF, self).save_model(output_name)


class MultiEventKernelCRF(StructEventKernelCRF):
    def __init__(self, para, ext_data=None):
        # Basic kernel parameters are shared.
        l_mu, l_sigma = para.form_kernels()
        super(MultiEventKernelCRF, self).__init__(para, ext_data)
        self.kernel_type = para.entity_event_kernel_type
        logging.info("Kernel type is %d", self.kernel_type)

        # Implementation note:
        # 1. The 4 sections of similarities may have their own kernels or shared.
        # 2. We add additional argument kernels (the 5th kernel)
        # 3. Node LR is always shared since the features are on different dimensions.

        if self.kernel_type == 0:
            # Type 0, falling back to the basic one kernel mode.
            # We only use the default kernel pooling layer and linear.
            return

        if self.kernel_type == 1:
            # Type 1, the kernels are shared, but the sections
            # are pooled individually.
            self.kp_evm = self.kp
            self.kp_ent_evm = self.kp
            self.kp_evm_ent = self.kp

            # The output layers are shared too, note that we have two
            # sets of pooled features.
            self.e_linear = nn.Linear(self.K * 2 + 1, 1, bias=True)
            self.evm_linear = self.e_linear
        elif self.kernel_type == 2:
            # Type 2, events have their own votes, yet there are
            # no direction in entity event similarities.
            self.kp_evm = KernelPooling(l_mu, l_sigma)
            self.kp_ent_evm = KernelPooling(l_mu, l_sigma)
            self.kp_evm_ent = self.kp_ent_evm

            # The output layers are not shared.
            self.e_linear = nn.Linear(self.K * 2 + 1, 1, bias=True)
            if para.arg_voting:
                self.evm_linear = nn.Linear(self.K * 3 + 1, 1, bias=True)
            else:
                self.evm_linear = nn.Linear(self.K * 2 + 1, 1, bias=True)
        elif self.kernel_type == 3:
            # Type 3, there are even direction between event entity
            # similarities
            self.kp_evm = KernelPooling(l_mu, l_sigma)
            self.kp_ent_evm = KernelPooling(l_mu, l_sigma)
            self.kp_evm_ent = KernelPooling(l_mu, l_sigma)

            # The output layers are not shared.
            self.e_linear = nn.Linear(self.K * 2 + 1, 1, bias=True)
            if para.arg_voting:
                self.evm_linear = nn.Linear(self.K * 3 + 1, 1, bias=True)
            else:
                self.evm_linear = nn.Linear(self.K * 2 + 1, 1, bias=True)

        if use_cuda:
            self.kp_evm.cuda()
            self.kp_ent_evm.cuda()
            self.kp_evm_ent.cuda()
            self.e_linear.cuda()
            self.evm_linear.cuda()

        if para.arg_voting:
            # Argument voting always have its own kernel, to simplify experiments.
            self.kp_args = KernelPooling(l_mu, l_sigma)
            if use_cuda:
                self.kp_args.cuda()

        self.arg_voting = para.arg_voting

    def combined_embedding(self, mtx_e, mtx_evm, mask_e, mask_evm):
        mtx_e_embedding = self.embedding(mtx_e)

        # Fall back to single kernel model.
        if mtx_evm is None:
            combined_mtx_emb = mtx_e_embedding
            combined_mtx_emb_mask = mask_e
        else:
            mtx_evm_embedding = self.embedding(mtx_evm)
            combined_mtx_emb = torch.cat((mtx_e_embedding, mtx_evm_embedding), 1)
            combined_mtx_emb_mask = torch.cat((mask_e, mask_evm), 1)

        return combined_mtx_emb, combined_mtx_emb_mask

    def compute_kernel_features_scores(self, kp, mtx_emb, voter_score, node_features):
        norm_mtx_emb = nn.functional.normalize(mtx_emb, p=2, dim=-1)
        trans_mtx = torch.matmul(norm_mtx_emb, norm_mtx_emb.transpose(-2, -1))
        kp_mtx = kp(trans_mtx, voter_score)
        features = torch.cat((kp_mtx, node_features), -1)
        return self.linear(features).squeeze(-1)

    def single_kernel_forward(self, h_packed_data):
        mtx_e = h_packed_data['mtx_e']
        mtx_evm = h_packed_data['mtx_evm']

        masks = h_packed_data['masks']
        mask_e = masks['mtx_e']
        mask_evm = masks['mtx_evm']

        mtx_e_score = h_packed_data['mtx_e_score']
        mtx_evm_score = h_packed_data['mtx_evm_score']

        ts_e_feature = h_packed_data['ts_e_feature']
        ts_evm_feature = h_packed_data['ts_evm_feature']

        # Combine with node features.
        e_node_score = F.tanh(self.node_lr(ts_e_feature))
        evm_node_score = F.tanh(self.node_lr(ts_evm_feature))
        node_features = torch.cat((e_node_score, evm_node_score), -2)

        combined_mtx_emb, combined_mtx_emb_mask = self.\
            combined_embedding(mtx_e, mtx_evm, mask_e, mask_evm)
        if self.use_mask:
            masked_mtx_emb = combined_mtx_emb * combined_mtx_emb_mask.unsqueeze(-1)
        else:
            masked_mtx_emb = combined_mtx_emb

        mtx_score = torch.cat((mtx_e_score, mtx_evm_score), -1)

        output = self.compute_kernel_features_scores(self.kp, masked_mtx_emb,
                                                     mtx_score, node_features)

        entity_length = e_node_score.size()[1]
        e_output = output[:, :entity_length]
        evm_output = output[:, entity_length:]

        return e_output, evm_output

    def multi_kernel_forward(self, h_packed_data):
        ts_e_feature = h_packed_data['ts_e_feature']
        ts_evm_feature = h_packed_data['ts_evm_feature']

        mtx_e = h_packed_data['mtx_e']
        mtx_evm = h_packed_data['mtx_evm']

        masks = h_packed_data['masks']
        mask_e = masks['mtx_e']
        mask_evm = masks['mtx_evm']

        mtx_e_score = h_packed_data['mtx_e_score']
        mtx_evm_score = h_packed_data['mtx_evm_score']

        ts_args = h_packed_data['ts_args']
        mtx_arg_length = h_packed_data['mtx_arg_length']
        ts_arg_mask = masks['ts_args']

        e_node_score = F.tanh(self.node_lr(ts_e_feature))
        evm_node_score = F.tanh(self.node_lr(ts_evm_feature))

        mtx_e_embedding = self.embedding(mtx_e)

        if self.use_mask:
            mtx_e_embedding = mtx_e_embedding * mask_e
        norm_entity_emb = nn.functional.normalize(mtx_e_embedding, p=2, dim=-1)

        kp_e_mtx = self.__entity_kernel_vote(norm_entity_emb, mtx_e_score)

        # Event Section:
        if mtx_evm is not None:
            mtx_evm_embedding = self.embedding(mtx_evm)
            norm_event_emb = nn.functional.normalize(mtx_evm_embedding, p=2, dim=-1)
            if self.use_mask:
                norm_event_emb = norm_event_emb * mask_evm

            # Event vote from event: batch x len(evm) x K
            kp_evm_mtx = self.__evm_kernel_vote(norm_event_emb, mtx_evm_score)

            entity_event_trans = self.__entity_event_trans(norm_entity_emb,
                                                           norm_event_emb)

            event_entity_trans = entity_event_trans.transpose(-2, -1)

            # Event vote from entity: batch x len(evm) x K
            kp_event_ent_mtx = self.__entity_event_kernel_vote(entity_event_trans,
                                                               mtx_e_score)
            # Entity vote from event: batch x len(e) x K
            kp_ent_event_mtx = self.__entity_event_kernel_vote(event_entity_trans,
                                                               mtx_evm_score)

            # Compute event features, size based on whether using arguments.
            if self.arg_voting:
                # Now work on arguments:
                mtx_arg_emb = self.argument_vector(ts_args, ts_arg_mask,
                                                   mtx_arg_length)
                norm_args_emb = nn.functional.normalize(mtx_arg_emb, p=2, dim=-1)
                norm_args_emb = norm_args_emb * mask_evm
                # batch x len(evm) x K
                kp_arg_mtx = self.__arg_kernel_vote(norm_args_emb, mtx_evm_score)
                event_features = torch.cat((kp_event_ent_mtx, kp_evm_mtx, kp_arg_mtx,
                                            evm_node_score), -1)
            else:
                event_features = torch.cat((kp_event_ent_mtx, kp_evm_mtx,
                                            evm_node_score), -1)

            # Compute entity features.
            # batch x len(e) x K
            entity_features = torch.cat((kp_e_mtx, kp_ent_event_mtx, e_node_score), -1)
            event_output = self.evm_linear(event_features).squeeze(-1)

        else:
            entity_features = torch.cat((kp_e_mtx, e_node_score), -1)
            # how to output an empty result?
            event_output = 0

        entity_output = self.e_linear(entity_features).squeeze(-1)

        return entity_output, event_output

    def forward(self, h_packed_data):
        if self.kernel_type == 0:
            return self.single_kernel_forward(h_packed_data)
        else:
            return self.multi_kernel_forward(h_packed_data)

    def argument_vector(self, ts_args, ts_arg_mask, mtx_arg_length):
        mtx_arg_embedding_sum = self._argument_sum(ts_args, ts_arg_mask)

        # Remove zero lengths before division.
        mtx_arg_length[mtx_arg_length == 0] = 1

        broadcast_length = mtx_arg_length.unsqueeze(2).type_as(
            mtx_arg_embedding_sum)
        # Average argument embedding.
        mtx_arg_emb = mtx_arg_embedding_sum / broadcast_length
        return mtx_arg_emb

    def __entity_event_trans(self, entity_emb, event_emb):
        # Shared trans matrix between block (1,2) and block (2,1)
        return torch.matmul(entity_emb, event_emb.transpose(-2, -1))

    def __entity_event_kernel_vote(self, trans_mtx, voter_score):
        # Block (1,2)
        return self.kp_ent_evm(trans_mtx, voter_score)

    def __event_entity_kernel_vote(self, trans_mtx, voter_score):
        # Block (2,1)
        return self.kp_evm_ent(trans_mtx, voter_score)

    def __entity_kernel_vote(self, entity_emb, voter_score):
        # Block (1,1)
        trans_mtx = torch.matmul(entity_emb, entity_emb.transpose(-2, -1))
        return self.kp(trans_mtx, voter_score)

    def __evm_kernel_vote(self, evm_emb, voter_score):
        # Block (2,2)
        trans_mtx = torch.matmul(evm_emb, evm_emb.transpose(-2, -1))
        return self.kp_args(trans_mtx, voter_score)

    def __arg_kernel_vote(self, arg_emb, voter_score):
        trans_mtx = torch.matmul(arg_emb, arg_emb.transpose(-2, -1))
        return self.kp_args(trans_mtx, voter_score)

    def _softmax_feature_size(self):
        return self.K + 1


class FeatureConcatKernelCRF(StructEventKernelCRF):
    io_group = 'joint_graph_simple'

    def __init__(self, para, ext_data=None):
        super(FeatureConcatKernelCRF, self).__init__(para, ext_data)

    def compute_score(self, h_packed_data):
        adjacent = h_packed_data['ts_adjacent']
        mtx_e, mtx_e_mask, mtx_score, node_score = self.get_raw_features(
            h_packed_data)

        if self.use_mask:
            kp_mtx = self._masked_kernel_scores(mtx_e_mask,
                                                mtx_e, mtx_score)
        else:
            kp_mtx = self._kernel_scores(mtx_e, mtx_score)

        features = torch.cat((kp_mtx, node_score), -1)

        # These features are the kernelized voting to the related entities.
        # I think putting them together with the events are making the kernels
        # confusing.
        edge_features = torch.bmm(adjacent, features)
        full_features = torch.cat((features, edge_features), -1)

        output = self.linear(full_features).squeeze(-1)

        return output

    def _softmax_feature_size(self):
        return (self.K + 1) * 2


# Average embedding models don't quite work.

class AverageEventKernelCRF(StructEventKernelCRF):
    def __init__(self, para, ext_data=None):
        super(AverageEventKernelCRF, self).__init__(para, ext_data)

    def event_embedding(self, mtx_evm, ts_args, mtx_arg_length, ts_arg_mask):
        mtx_p_embedding = self.embedding(mtx_evm)

        if ts_args is None:
            # When there are no arguments, the embedding is just the predicate.
            mtx_evm_embedding_aver = mtx_p_embedding
        else:
            mtx_arg_embedding_sum = self._argument_sum(ts_args, ts_arg_mask)
            mtx_evm_embedding_sum = mtx_p_embedding + mtx_arg_embedding_sum

            # aver = (embedding sum) / (1 + arg length)
            mtx_full_length = (mtx_arg_length + 1).type_as(
                mtx_evm_embedding_sum).unsqueeze(2)
            mtx_evm_embedding_aver = mtx_evm_embedding_sum / mtx_full_length
        return mtx_evm_embedding_aver

    def _softmax_feature_size(self):
        return self.K + 1


class AverageArgumentKernelCRF(StructEventKernelCRF):
    def __init__(self, para, ext_data=None):
        super(AverageArgumentKernelCRF, self).__init__(para, ext_data)
        self.args_linear = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.evm_arg_linear = nn.Linear(self.embedding_dim * 2,
                                        self.embedding_dim)

        if use_cuda:
            self.args_linear.cuda()
            self.evm_arg_linear.cuda()

    def event_embedding(self, mtx_evm, ts_args, mtx_arg_length, ts_arg_mask):
        mtx_p_embedding = self.embedding(mtx_evm)

        if ts_args is None:
            # This is actually wrong, you cannot simulate the embedding with zeros.
            mtx_arg = torch.zeros(mtx_p_embedding.size())
            if use_cuda:
                mtx_arg = mtx_arg.cuda()
        else:
            mtx_arg_embedding_sum = self._argument_sum(ts_args, ts_arg_mask)
            # Remove zero lengths.
            mtx_arg_length[mtx_arg_length == 0] = 1
            broadcast_length = mtx_arg_length.unsqueeze(2).type_as(
                mtx_arg_embedding_sum)
            # Average argument embedding.
            mtx_arg_embedding_aver = mtx_arg_embedding_sum / broadcast_length
            # Non linearly map the argument embeddings.
            mtx_arg = F.tanh(self.args_linear(mtx_arg_embedding_aver))

        mtx_evm_args_cat = torch.cat((mtx_p_embedding, mtx_arg), 2)

        # Non linearly combine event and argument embeddings.
        return F.tanh(self.evm_arg_linear(mtx_evm_args_cat))

    def _softmax_feature_size(self):
        return self.K + 1


# Graph models don't quite work.
class GraphCNNKernelCRF(StructEventKernelCRF):
    def __init__(self, para, ext_data=None):
        super(GraphCNNKernelCRF, self).__init__(para, ext_data)
        self.w_cnn = nn.Linear(self.K + 1, self.K + 1, bias=True)
        if use_cuda:
            self.w_cnn.cuda()

    def compute_score(self, h_packed_data):
        adjacent = h_packed_data['ts_adjacent']
        gcnn_input = self.combined_features(h_packed_data)
        gcnn_out = self.gcnn_layer(adjacent, gcnn_input)
        output = self.linear(gcnn_out).squeeze(-1)
        return output

    def combined_features(self, h_packed_data):
        mtx_e, mtx_e_mask, mtx_score, node_score = self.get_raw_features(
            h_packed_data)
        if self.use_mask:
            kp_mtx = self._masked_kernel_scores(mtx_e_mask,
                                                mtx_e, mtx_score)
        else:
            kp_mtx = self._kernel_scores(mtx_e, mtx_score)
        return torch.cat((kp_mtx, node_score), -1)

    def gcnn_layer(self, adjacent, gcnn_input):
        gcnn_features = torch.bmm(adjacent, gcnn_input)
        return F.dropout(F.relu(self.w_cnn(gcnn_features)))

    def _softmax_feature_size(self):
        return self.K + 1


class ResidualGraphCNNKernelCRF(GraphCNNKernelCRF):
    def __init__(self, para, ext_data=None):
        super(ResidualGraphCNNKernelCRF, self).__init__(para, ext_data)

    def compute_score(self, h_packed_data):
        adjacent = h_packed_data['ts_adjacent']
        gcnn_input = self.combined_features(h_packed_data)
        gcnn_out = self.gcnn_layer(adjacent, gcnn_input)
        full_features = gcnn_input + gcnn_out
        output = self.linear(full_features).squeeze(-1)
        return output

    def _softmax_feature_size(self):
        return self.K + 1


class ConcatGraphCNNKernelCRF(GraphCNNKernelCRF):
    def __init__(self, para, ext_data=None):
        super(ConcatGraphCNNKernelCRF, self).__init__(para, ext_data)

    def compute_score(self, h_packed_data):
        adjacent = h_packed_data['ts_adjacent']
        gcnn_input = self.combined_features(h_packed_data)
        gcnn_out = self.gcnn_layer(adjacent, gcnn_input)
        full_features = torch.cat((gcnn_input, gcnn_out), -1)
        output = self.linear(full_features).squeeze(-1)

        return output

    def _softmax_feature_size(self):
        return (self.K + 1) * 2
