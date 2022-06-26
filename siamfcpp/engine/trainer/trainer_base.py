# -*- coding: utf-8 -*
import os
import os.path as osp
from copy import deepcopy
from typing import Dict, List, Tuple

from loguru import logger

import torch
from torch import nn
from torch.utils.data import DataLoader

from siamfcpp.model.module_base import ModuleBase
from siamfcpp.optim.optimizer.optimizer_base import OptimizerBase
from siamfcpp.utils import Registry, ensure_dir, unwrap_model

import numpy as np

TRACK_TRAINERS = Registry('TRACK_TRAINERS')
VOS_TRAINERS = Registry('VOS_TRAINERS')

TASK_TRAINERS = dict(
    track=TRACK_TRAINERS,
    vos=VOS_TRAINERS,
)


class TrainerBase:
    r"""
    Trainer base class (e.g. procedure defined for tracker / segmenter / etc.)
    Interface descriptions:
    """
    # Define your default hyper-parameters here in your sub-class.
    default_hyper_params = dict(
        exp_name="default_training",
        exp_save="snapshots",
        max_epoch=20,
    )

    def __init__(self, optimizer, dataloader, monitors=[]):
        self._hyper_params = deepcopy(
            self.default_hyper_params)  # mapping-like object
        self._state = dict()  # pipeline state
        self._model = optimizer._model
        self._losses = optimizer._model.loss
        self._optimizer = optimizer
        self._monitors = monitors
        self._dataloader = iter(dataloader)  # get the iterabel dataloader

    def get_hps(self) -> Dict:
        r"""
        Getter function for hyper-parameters

        Returns
        -------
        dict
            hyper-parameters
        """
        return self._hyper_params

    def set_hps(self, hps: Dict) -> None:
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

    def update_params(self, ):
        self._hyper_params["num_iterations"] = self._hyper_params[
            "nr_image_per_epoch"] // self._hyper_params["minibatch"]
        self._state["snapshot_dir"] = osp.join(self._hyper_params["exp_save"],
                                               self._hyper_params["exp_name"])

        self._state["snapshot_file"] = self._hyper_params["snapshot"]

    def init_train(self):
        r"""
        an interface to process pre-train overhead before training
        trainer's implementation is responsible for checking and 
            calling it automatically before training starts.
        """
        for monitor in self._monitors:
            monitor.init(self._state)

    def train(self):
        r"""
        an interface to train for one epoch
        """
    def set_dataloader(self, dataloader: DataLoader):
        r""""""
        self._dataloader = dataloader

    def set_optimizer(self, optimizer: OptimizerBase):
        r""""""
        self._optimizer = optimizer

    def is_completed(self):
        r"""Return completion status"""
        is_completed = (self._state["epoch"] + 1 >=
                        self._hyper_params["max_epoch"])
        return is_completed

    def load_snapshot(self):
        r""" 
        load snapshot based on self._hyper_params["snapshot"] or self._state["epoch"]
        """
        snapshot_file = self._state["snapshot_file"]
        dev = self._state["devices"][0]
        # snapshot1 = torch.load('/home/lsw/lsw/siamfc++/models/snapshots/siamfcpp_alexnet-got/epoch-17.pkl', map_location=dev)
        # # self.load_backbone_head(snapshot["model_state_dict"])
        snapshot = torch.load('/home/lsw/PycharmProjects/siamfc++dynprune/models/pretrained_models/alexnet-nopad-bn-md5.pkl',
                              map_location=dev)
        self.load_backbone_head_pretrained(snapshot)

        if osp.exists(snapshot_file):
            dev = self._state["devices"][0]
            snapshot = torch.load(snapshot_file, map_location=dev)
            self.load_backbone_head(snapshot["model_state_dict"])
            # self._model.load_state_dict(snapshot["model_state_dict"])    #============================
            self._optimizer.load_state_dict(snapshot["optimizer_state_dict"])
            self._state["epoch"] = snapshot["epoch"]
            logger.info("Load snapshot from: %s" % osp.realpath(snapshot_file))
        else:
            logger.info("%s does not exist, no snapshot loaded." %
                        snapshot_file)

        logger.info("Train from epoch %d" % (self._state["epoch"] + 1))

    def load_backbone_head_pretrained(self, oristate_dict):
        state_dict = self._model.state_dict()
        last_select_index = None  # Conv index selected in the previous layer

        cnt = 0
        ranks_path = '/home/lsw/PycharmProjects/siamfc++dynprune/rank_conv/ranks_avg.npy'
        dist_ranks_avg = np.load(ranks_path, allow_pickle=True).item()
        # for conv in dist_ranks_avg:
        #     if conv =='feature_result_cls_p5_conv3':
        #         print(conv)
        #     print(conv)
        #     print(dist_ranks_avg[conv].shape)
        #     print(dist_ranks_avg[conv])
        #     print((dist_ranks_avg[conv] < 10).sum())
        backbone = self._model.basemodel
        #===============backbone=================
        # for module in backbone:
        #     name = name.replace('module.', '')
        #     if isinstance(module, nn.Conv2d):
        #         oriweight = oristate_dict[name + '.weight']
        #=============================================
        dist_in_channels = {
            'basemodel.conv1.conv': [0, 1, 2]
        }

        # for name in oristate_dict.keys():
        #     # name = name.replace('module.', '')
        #     # name = name.replace('basemodel.', '')
        #     print(name)

        for name, module in self._model.named_modules():
            name = name.replace('module.', '')
            if not 'basemodel' in name:
                continue

            if isinstance(module, nn.Conv2d):
                print(name)
                cnt += 1
                oriweight = oristate_dict[name.replace('basemodel.', '') + '.weight']
                curweight = state_dict[name + '.weight']
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                # dist_ranks_avg = np.load(prefix, allow_pickle=True).item()
                select_index = np.argsort(dist_ranks_avg['feature_result_' + name.split('.')[1]])[orifilter_num - currentfilter_num:]
                select_index = np.array(select_index)
                select_index.sort()
                dist_in_channels[name.split('.')[0] + '.conv'+str(int(name.split('.')[1][-1])+1) + '.' + name.split('.')[2]] = select_index

                for index_i, i in enumerate(select_index):
                    state_dict[name + '.weight'][index_i] = oristate_dict[name.replace('basemodel.', '') + '.weight'][i][dist_in_channels[name],:,:]
        # ===============transf=================
        # self._model.r_z_k.conv.weight
        # state_dict['r_z_k.conv.weight'] = oristate_dict[name + '.weight'][:,dist_in_channels['basemodel.conv5.conv'],:,:]

        # for name, module in self._model.named_modules():
        #     name = name.replace('module.', '')
        #     name = name.replace('basemodel.', '')
        #     print(name)
        #     if 'basemodel' in name or 'head' in name:
        #         continue
        #
        #     if isinstance(module, nn.Conv2d):
        #         print(name)
        #         cnt += 1
        #         oriweight = oristate_dict[name + '.weight']
        #         curweight = state_dict[name + '.weight']
        #         orifilter_num = oriweight.size(0)
        #         currentfilter_num = curweight.size(0)
        #
        #         # dist_ranks_avg = np.load(prefix, allow_pickle=True).item()
        #         select_index = np.argsort(dist_ranks_avg['feature_result_' + name.split('.')[0]])[orifilter_num - currentfilter_num:]
        #         select_index = np.array(select_index)
        #         select_index.sort()
        #         if 'c_x' in name:
        #             dist_in_channels['head.cls_p5_conv1.conv'] = select_index
        #         else:
        #             dist_in_channels['head.bbox_p5_conv1.conv'] = select_index
        #
        #         for index_i, i in enumerate(select_index):
        #             state_dict[name + '.weight'][index_i] = oristate_dict[name + '.weight'][i][dist_in_channels['basemodel.conv6.conv'],:,:]
        # # ===============head=================
        # for name, module in self._model.named_modules():
        #     name = name.replace('module.', '')
        #     name = name.replace('basemodel.', '')
        #     print(name)
        #     if not 'head' in name:
        #         continue
        #     if 'score' in name or 'offset' in name:
        #         continue
        #
        #     if isinstance(module, nn.Conv2d):
        #         print(name)
        #         cnt += 1
        #         oriweight = oristate_dict[name + '.weight']
        #         curweight = state_dict[name + '.weight']
        #         orifilter_num = oriweight.size(0)
        #         currentfilter_num = curweight.size(0)
        #
        #         # dist_ranks_avg = np.load(prefix, allow_pickle=True).item()
        #         select_index = np.argsort(dist_ranks_avg['feature_result_' + name.split('.')[1]])[orifilter_num - currentfilter_num:]
        #         select_index = np.array(select_index)
        #         select_index.sort()
        #         dist_in_channels[name.split('.')[0] + '.' + name.split('.')[1][:-1] +str(int(name.split('.')[1][-1])+1) + '.' + name.split('.')[2]] = select_index
        #
        #         for index_i, i in enumerate(select_index):
        #             state_dict[name + '.weight'][index_i] = oristate_dict[name + '.weight'][i][dist_in_channels[name],:,:]
        #
        # # ===============output=================
        # for name, module in self._model.named_modules():
        #     name = name.replace('module.', '')
        #     name = name.replace('basemodel.', '')
        #     print(name)
        #     if not ('score' in name or 'offset' in name):
        #         continue
        #
        #     if isinstance(module, nn.Conv2d):
        #         print(name)
        #         cnt += 1
        #         oriweight = oristate_dict[name + '.weight']
        #         curweight = state_dict[name + '.weight']
        #         # orifilter_num = oriweight.size(0)
        #         # currentfilter_num = curweight.size(0)
        #
        #         if 'cls' in name:
        #             state_dict[name + '.weight'] = oristate_dict[name + '.weight'][:,dist_in_channels['head.cls_p5_conv4.conv'], :, :]
        #         else:
        #             state_dict[name + '.weight'] = oristate_dict[name + '.weight'][:,dist_in_channels['head.bbox_p5_conv4.conv'], :, :]
        self._model.load_state_dict(state_dict)

    def load_backbone_head(self, oristate_dict):
        state_dict = self._model.state_dict()
        last_select_index = None  # Conv index selected in the previous layer

        cnt = 0
        ranks_path = '/home/lsw/model-compression/FisherInfo/SiamFCpp-video_analyst_cmp_Fisher/rank_conv/ranks_avg_fisher.npy'
        dist_ranks_avg = np.load(ranks_path, allow_pickle=True).item()
        # for conv in dist_ranks_avg:
        #     if conv =='feature_result_cls_p5_conv3':
        #         print(conv)
        #     print(conv)
        #     print(dist_ranks_avg[conv].shape)
        #     print(dist_ranks_avg[conv])
        #     print((dist_ranks_avg[conv] < 10).sum())
        backbone = self._model.basemodel
        #===============backbone=================
        # for module in backbone:
        #     name = name.replace('module.', '')
        #     if isinstance(module, nn.Conv2d):
        #         oriweight = oristate_dict[name + '.weight']
        #=============================================
        dist_in_channels = {
            'basemodel.conv1.conv': [0, 1, 2]
        }
        for name, module in self._model.named_modules():
            name = name.replace('module.', '')
            if not 'basemodel' in name:
                continue

            if isinstance(module, nn.Conv2d):
                print(name)
                cnt += 1
                oriweight = oristate_dict[name + '.weight']
                curweight = state_dict[name + '.weight']
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                # dist_ranks_avg = np.load(prefix, allow_pickle=True).item()
                select_index = np.argsort(dist_ranks_avg['feature_result_' + name.split('.')[1]])[orifilter_num - currentfilter_num:]
                select_index = np.array(select_index)
                select_index.sort()
                dist_in_channels[name.split('.')[0] + '.conv'+str(int(name.split('.')[1][-1])+1) + '.' + name.split('.')[2]] = select_index

                for index_i, i in enumerate(select_index):
                    state_dict[name + '.weight'][index_i] = oristate_dict[name + '.weight'][i][dist_in_channels[name],:,:]
        # ===============transf=================
        # self._model.r_z_k.conv.weight
        # state_dict['r_z_k.conv.weight'] = oristate_dict[name + '.weight'][:,dist_in_channels['basemodel.conv5.conv'],:,:]

        for name, module in self._model.named_modules():
            name = name.replace('module.', '')
            print(name)
            if 'basemodel' in name or 'head' in name:
                continue

            if isinstance(module, nn.Conv2d):
                print(name)
                cnt += 1
                oriweight = oristate_dict[name + '.weight']
                curweight = state_dict[name + '.weight']
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                # dist_ranks_avg = np.load(prefix, allow_pickle=True).item()
                select_index = np.argsort(dist_ranks_avg['feature_result_' + name.split('.')[0]])[orifilter_num - currentfilter_num:]
                select_index = np.array(select_index)
                select_index.sort()
                if 'c_x' in name:
                    dist_in_channels['head.cls_p5_conv1.conv'] = select_index
                else:
                    dist_in_channels['head.bbox_p5_conv1.conv'] = select_index

                for index_i, i in enumerate(select_index):
                    state_dict[name + '.weight'][index_i] = oristate_dict[name + '.weight'][i][dist_in_channels['basemodel.conv6.conv'],:,:]
        # ===============head=================
        for name, module in self._model.named_modules():
            name = name.replace('module.', '')
            print(name)
            if not 'head' in name:
                continue
            if 'score' in name or 'offset' in name:
                continue

            if isinstance(module, nn.Conv2d):
                print(name)
                cnt += 1
                oriweight = oristate_dict[name + '.weight']
                curweight = state_dict[name + '.weight']
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                # dist_ranks_avg = np.load(prefix, allow_pickle=True).item()
                select_index = np.argsort(dist_ranks_avg['feature_result_' + name.split('.')[1]])[orifilter_num - currentfilter_num:]
                select_index = np.array(select_index)
                select_index.sort()
                dist_in_channels[name.split('.')[0] + '.' + name.split('.')[1][:-1] +str(int(name.split('.')[1][-1])+1) + '.' + name.split('.')[2]] = select_index

                for index_i, i in enumerate(select_index):
                    state_dict[name + '.weight'][index_i] = oristate_dict[name + '.weight'][i][dist_in_channels[name],:,:]

        # ===============output=================
        for name, module in self._model.named_modules():
            name = name.replace('module.', '')
            print(name)
            if not ('score' in name or 'offset' in name):
                continue

            if isinstance(module, nn.Conv2d):
                print(name)
                cnt += 1
                oriweight = oristate_dict[name + '.weight']
                curweight = state_dict[name + '.weight']
                # orifilter_num = oriweight.size(0)
                # currentfilter_num = curweight.size(0)

                if 'cls' in name:
                    state_dict[name + '.weight'] = oristate_dict[name + '.weight'][:,dist_in_channels['head.cls_p5_conv4.conv'], :, :]
                else:
                    state_dict[name + '.weight'] = oristate_dict[name + '.weight'][:,dist_in_channels['head.bbox_p5_conv4.conv'], :, :]
        self._model.load_state_dict(state_dict)

    def save_snapshot(self, ):
        r""" 
        save snapshot for current epoch
        """
        epoch = self._state["epoch"]
        snapshot_dir, snapshot_file = self._infer_snapshot_dir_file_from_epoch(
            epoch)
        snapshot_dict = {
            'epoch': epoch,
            'model_state_dict': unwrap_model(self._model).state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict()
        }
        ensure_dir(snapshot_dir)
        torch.save(snapshot_dict, snapshot_file)
        while not osp.exists(snapshot_file):
            logger.info("retrying")
            torch.save(snapshot_dict, snapshot_file)
        logger.info("Snapshot saved at: %s" % snapshot_file)

    def _infer_snapshot_dir_file_from_epoch(self,
                                            epoch: int) -> Tuple[str, str]:
        r"""Infer snapshot's directory & file path based on self._state & epoch number pased in

        Parameters
        ----------
        epoch : int
            epoch number
        
        Returns
        -------
        Tuple[str, str]
            directory and snapshot file
            dir, path
        """
        snapshot_dir = self._state["snapshot_dir"]
        snapshot_file = osp.join(snapshot_dir, "epoch-{}.pkl".format(epoch))
        return snapshot_dir, snapshot_file

    def _get_latest_model_path(self):
        file_dir = self._state["snapshot_dir"]
        file_list = os.listdir(file_dir)
        file_list = [
            file_name for file_name in file_list if file_name.endswith("pkl")
        ]
        if not file_list:
            return "none"
        file_list.sort(key=lambda fn: os.path.getmtime(osp.join(file_dir, fn))
                       if not os.path.isdir(osp.join(file_dir, fn)) else 0)
        return osp.join(file_dir, file_list[-1])

    def resume(self, resume):
        r"""Apply resuming by setting self._state["snapshot_file"]
        Priviledge snapshot_file to epoch number

        Parameters
        ----------
        resume :str
            latest epoch number, by default -1, "latest" or model path
        """
        if resume.isdigit():
            _, snapshot_file = self._infer_snapshot_dir_file_from_epoch(resume)
            self._state["snapshot_file"] = snapshot_file
        elif resume == "latest":
            self._state["snapshot_file"] = self._get_latest_model_path()
        else:
            self._state["snapshot_file"] = resume

    def set_device(self, devs: List[str]):
        self._state["devices"] = [torch.device(dev) for dev in devs]
