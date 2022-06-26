# -*- coding: utf-8 -*
from copy import deepcopy

from loguru import logger

import torch
from torch import nn

from siamfcpp.utils import md5sum

from .utils.load_state import (filter_reused_missing_keys,
                               get_missing_parameters_message,
                               get_unexpected_parameters_message)

import numpy as np

class ModuleBase(nn.Module):
    r"""
    Module/component base class
    """
    # Define your default hyper-parameters here in your sub-class.
    default_hyper_params = dict(pretrain_model_path="")

    def __init__(self):
        super(ModuleBase, self).__init__()
        self._hyper_params = deepcopy(self.default_hyper_params)

    def get_hps(self) -> dict():
        r"""
        Getter function for hyper-parameters

        Returns
        -------
        dict
            hyper-parameters
        """
        return self._hyper_params

    def set_hps(self, hps: dict()) -> None:
        r"""
        Set hyper-parameters

        Arguments
        ---------
        hps: dict
            dict of hyper-parameters, the keys must in self.__hyper_params__
        """
        for key in hps:
            if key not in self._hyper_params:
                raise KeyError
            self._hyper_params[key] = hps[key]

    def update_params(self):
        model_file = self._hyper_params.get("pretrain_model_path", "")
        if model_file != "":
            state_dict = torch.load(model_file,
                                    map_location=torch.device("cpu"))
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            self.load_model_param(state_dict)
            # params=self.load_model_param_and_compute(state_dict)
            logger.info(
                "Load pretrained {} parameters from: {} whose md5sum is {}".
                format(self.__class__.__name__, model_file, md5sum(model_file)))

    def load_model_param(self, checkpoint_state_dict):
        model_state_dict = self.state_dict()
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    logger.warning(
                        "'{}' has shape {} in the checkpoint but {} in the "
                        "model! Skipped.".format(k, shape_checkpoint,
                                                 shape_model))
                    checkpoint_state_dict.pop(k)
        # pyre-ignore
        # self.load_backbone(checkpoint_state_dict)
        incompatible = self.load_state_dict(checkpoint_state_dict, strict=False)
        if incompatible.missing_keys:
            missing_keys = filter_reused_missing_keys(self,
                                                      incompatible.missing_keys)
            if missing_keys:
                logger.warning(get_missing_parameters_message(missing_keys))
        if incompatible.unexpected_keys:
            logger.warning(
                get_unexpected_parameters_message(incompatible.unexpected_keys))
    def load_model_param_and_compute(self, checkpoint_state_dict):
        model_state_dict = self.state_dict()
        # Parameters are loaded, no need to initialized
        params=0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                print(m)
                params+=(np.prod(m.kernel_size)*m.in_channels+1)*m.out_channels

            elif isinstance(m, nn.BatchNorm2d):
                print(m)
                params += 4*m.weight.shape[0]

            elif isinstance(m, nn.Linear):
                print(m)
        return params

    def load_backbone(self, oristate_dict):
        state_dict = self.state_dict()
        last_select_index = None  # Conv index selected in the previous layer

        cnt = 0
        prefix =  '/home/lsw/model-compression/test1/SiamTrackers-compression1/SiamFCpp/SiamFCpp-video_analyst_cmp1/rank_conv'
        subfix = ".npy"
        for name, module in self.named_modules():
            name = name.replace('module.', '')

            if isinstance(module, nn.Conv2d):

                cnt += 1
                oriweight = oristate_dict[name + '.weight']
                curweight = state_dict[name + '.weight']
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                # dist_ranks_avg = np.load(prefix, allow_pickle=True).item()
                select_index = np.argsort(dist_ranks_avg)[orifilter_num - currentfilter_num:]
                select_index.sort()
                for index_i, i in enumerate(select_index):
                    state_dict[name_base + name + '.weight'][index_i] = \
                        oristate_dict[name + '.weight'][i]
                self.load_state_dict(state_dict)

        #     if orifilter_num != currentfilter_num:
        #
        #             cov_id = cnt
        #             logger.info('loading rank from: ' + prefix + str(cov_id) + subfix)
        #             rank = np.load(prefix + str(cov_id) + subfix)
        #             select_index = np.argsort(rank)[orifilter_num - currentfilter_num:]  # preserved filter id
        #             select_index.sort()
        #
        #             if last_select_index is not None:
        #                 for index_i, i in enumerate(select_index):
        #                     for index_j, j in enumerate(last_select_index):
        #                         state_dict[name_base + name + '.weight'][index_i][index_j] = \
        #                             oristate_dict[name + '.weight'][i][j]
        #             else:
        #                 for index_i, i in enumerate(select_index):
        #                     state_dict[name_base + name + '.weight'][index_i] = \
        #                         oristate_dict[name + '.weight'][i]
        #
        #             last_select_index = select_index
        #
        #         elif last_select_index is not None:
        #             for i in range(orifilter_num):
        #                 for index_j, j in enumerate(last_select_index):
        #                     state_dict[name_base + name + '.weight'][i][index_j] = \
        #                         oristate_dict[name + '.weight'][i][j]
        #         else:
        #             state_dict[name_base + name + '.weight'] = oriweight
        #             last_select_index = None
        #
        # self.load_state_dict(state_dict)
    def load_state_dict_Alexnet(self, state_dict: 'OrderedDict[str, Tensor]',
                        strict: bool = True):
        r"""Copies parameters and buffers from :attr:`state_dict` into
        this module and its descendants. If :attr:`strict` is ``True``, then
        the keys of :attr:`state_dict` must exactly match the keys returned
        by this module's :meth:`~torch.nn.Module.state_dict` function.

        Args:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            strict (bool, optional): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~torch.nn.Module.state_dict` function. Default: ``True``

        Returns:
            ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
                * **missing_keys** is a list of str containing the missing keys
                * **unexpected_keys** is a list of str containing the unexpected keys

        Note:
            If a parameter or buffer is registered as ``None`` and its corresponding key
            exists in :attr:`state_dict`, :meth:`load_state_dict` will raise a
            ``RuntimeError``.
        """
        missing_keys: List[str] = []
        unexpected_keys: List[str] = []
        error_msgs: List[str] = []

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            # mypy isn't aware that "_metadata" exists in state_dict
            state_dict._metadata = metadata  # type: ignore[attr-defined]

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(self)
        del load

        if strict:
            if len(unexpected_keys) > 0:
                error_msgs.insert(
                    0, 'Unexpected key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in unexpected_keys)))
            if len(missing_keys) > 0:
                error_msgs.insert(
                    0, 'Missing key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in missing_keys)))

        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               self.__class__.__name__, "\n\t".join(error_msgs)))
        return _IncompatibleKeys(missing_keys, unexpected_keys)