# -*- coding: utf-8 -*-

import os
import argparse
from loguru import logger
import torch
root_path=os.path.abspath(os.path.dirname(os.getcwd()))
import sys
import yaml
sys.path.append(os.getcwd())
print(torch.cuda.is_available())
print(os.path.join(sys.prefix,'Library','bin','geos_c.dll'))
from siamfcpp.config.config import cfg as root_cfg
from siamfcpp.config.config import specify_task         # sot or vos
from siamfcpp.engine.builder import build as tester_builder
from siamfcpp.model import builder as model_builder
from siamfcpp.pipeline import builder as pipeline_builder
from siamfcpp.utils import complete_path_wt_root_in_cfg
'''

18 prec_score:0.63732030993138,--succ_score: 0.8433910774871092, --succ_rate:0.7988308783361867, --speed_fps: 83.8371658513596
    
17 prec_score:0.8425296573792467 --succ_score:0.6311629903175178 --succ_rate:0.7922191904567035  --speed_fps:


# 效果最好 
15 prec_score:0.853383016847436 --succ_score:0.6447268507216857 --succ_rate:0.8144395854907881   --speed_fps: 106.08829672531054
------------------------------------------------------------------
|   Tracker Name   | Accuracy | Robustness | Lost Number |  EAO  |
------------------------------------------------------------------
| siamfcpp_alexnet |  0.584   |   0.342    |    73.0     | 0.308 |

13 
------------------------------------------------------------------
|   Tracker Name   | Accuracy | Robustness | Lost Number |  EAO  |
------------------------------------------------------------------
| siamfcpp_alexnet |  0.568   |   0.435    |    93.0     | 0.260 |
------------------------------------------------------------------

'''

model=3#3，4, 5.6，8，9

if model==1:
    config=root_path+'/models/siamfcpp/test/vot/siamfcpp_alexnet.yaml'   #VOT
if model==2:
    config=root_path+'/models/siamfcpp/test/otb/siamfcpp_alexnet-otb.yaml'#OTB
if model==3:
    config=root_path+'/models/siamfcpp/test/dtb70/siamfcpp_alexnet-dtb70.yaml'#DTB
if model==4:
    config=root_path+'/models/siamfcpp/test/uav123/siamfcpp_alexnet-uav123.yaml'#UAV123
if model==5:
    config=root_path+'/models/siamfcpp/test/uav20l/siamfcpp_alexnet-uav20l.yaml'#UAV20L
if model==6:
    config=root_path+'/models/siamfcpp/test/uavdt/siamfcpp_alexnet-uavdt.yaml'#UAVDT+*-6

if model==7:
    config=root_path+'/models/siamfcpp/test/visdrone/siamfcpp_alexnet-visdrone2019.yaml'#visdrone
if model==8:
    config=root_path+'/models/siamfcpp/test/uav123@10fps/siamfcpp_alexnet-uav123@10fps.yaml'#UAV123@10fps
if model==9:
    config=root_path+'/models/siamfcpp/test/visdrone/siamfcpp_alexnet-visdrone2018.yaml'#visdrone2018
            #H:/PW-SiamFC++//PW-SiamFC++/models/siamfcpp/test/visdrone/siamfcpp_alexnet-visdrone2018.yaml
if __name__ == '__main__':
    # parsing
    #parser = make_parser()
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('-cfg', '--config', default=config,type=str,help='experiment configuration')
    parser.add_argument('-model_path', '--model_path', default=root_path+'/bin/models_FBS_all0.6/snapshots/siamfcpp_alexnet-got/epoch-17.pkl',type=str,help='experiment configuration')  # 为空则选择yaml里面默认的模型
    parsed_args = parser.parse_args()

    args=parser.parse_args() 

    # experiment config  #abspath 绝对路径bin/my_test.py:69

    exp_cfg_path = os.path.realpath(parsed_args.config)


    root_cfg.merge_from_file(exp_cfg_path)
    
   # print(root_cfg.test.track.model.task_model.SiamTrack.pretrain_model_path)

    if args.model_path:
        root_cfg.test.track.model.task_model.SiamTrack.pretrain_model_path=args.model_path 

    #print(root_cfg.test.track.model.task_model.SiamTrack.pretrain_model_path)

    logger.info("Load experiment configuration at: %s" % exp_cfg_path) 
    ROOT_PATH  =os.getcwd()
    ROOT_PATH = root_path

    root_cfg = complete_path_wt_root_in_cfg(root_cfg,ROOT_PATH ) #把cfg中的相对路径,变成了绝对路径
    

    root_cfg = root_cfg.test #获取test的yaml root_cfg['test']
    #root_cfg['track']
    #root_cfg['vos']
    task, task_cfg = specify_task(root_cfg) #获取任务 track or vos ,


    task_cfg.freeze()

    torch.multiprocessing.set_start_method('spawn', force=True)
    
    # build_siamfcpp_tester
    model = model_builder.build("track", task_cfg.model)
    # build pipeline
    pipeline = pipeline_builder.build("track", task_cfg.pipeline, model)#配置超参数
    # build tester
    testers = tester_builder("track", task_cfg.tester, "tester", pipeline)
    
    for tester in testers:
        tester.test()
