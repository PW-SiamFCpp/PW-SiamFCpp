import numpy as np
# ranks_path = '/home/lsw/model-compression/FisherInfo/SiamFCpp-video_analyst_cmp_Fisher/rank_conv/ranks_avg_fisher.npy'
# dist_ranks_avg = np.load(ranks_path, allow_pickle=True).item()

# ranks_path = '/home/lsw/PycharmProjects/PW-SiamFC++/rank_conv/ranks_avg.npy'
# dist_ranks_avg1 = np.load(ranks_path, allow_pickle=True).item()
#
# dist_ranks_thred ={
#     'feature_result_conv1': 20,
#     'feature_result_conv2': 10,
#     'feature_result_conv3': 8,
#     'feature_result_conv4': 2,
#     'feature_result_conv5': 15,
#     'feature_result_c_x': 24,
#     'feature_result_r_x': 24,
#     'feature_result_bbox_p5_conv1': 15,
#     'feature_result_bbox_p5_conv2': 0.000001,
#     'feature_result_bbox_p5_conv3': 14,
#     'feature_result_cls_p5_conv1': 15,
#     'feature_result_cls_p5_conv2': 0.000001,
#     'feature_result_cls_p5_conv3': 14,
# }
# # for conv in dist_ranks_avg:ko
# #     if conv == 'feature_result_cls_p5_conv3':
# #         print(conv)
# #     print(conv)
# #     print(dist_ranks_avg[conv].shape)
# #     print(dist_ranks_avg[conv])
# #     print((dist_ranks_avg[conv] < 10).sum())
#
#
# # dist_cmprate = {
# # 'backbone_conv1': 0.8,
# # 'backbone_conv2': 0.8,
# # 'backbone_conv3': 0.8,
# # 'backbone_conv4': 0.8,
# # 'backbone_conv5': 0.8,
# #
# # 'tranf_c_x': 0.8,
# # 'tranf_r_x': 0.8,
# # 'tranf_c_z_k': 0.8, # has to equal tranf_c_x
# # 'tranf_r_z_k': 0.8, # has to equal tranf_r_x
# #
# # 'head_bbox_conv3x3_1': 0.8,
# # 'head_bbox_conv3x3_2': 0.8,
# # 'head_bbox_conv3x3_3': 0.8,
# #
# # 'head_cls_conv3x3_1': 0.8,
# # 'head_cls_conv3x3_2': 0.8,
# # 'head_cls_conv3x3_3': 0.8,
# # }
#
#
# dist_cmprate0 = {
# 'backbone_conv1': 1-(dist_ranks_avg['feature_result_conv1'] < dist_ranks_thred['feature_result_conv1']).sum()/dist_ranks_avg['feature_result_conv1'].shape[0],
# 'backbone_conv2': 1-(dist_ranks_avg['feature_result_conv2'] < dist_ranks_thred['feature_result_conv2']).sum()/dist_ranks_avg['feature_result_conv2'].shape[0],
# 'backbone_conv3': 1-(dist_ranks_avg['feature_result_conv3'] < dist_ranks_thred['feature_result_conv3']).sum()/dist_ranks_avg['feature_result_conv3'].shape[0],
# 'backbone_conv4': 1-(dist_ranks_avg['feature_result_conv4'] < dist_ranks_thred['feature_result_conv4']).sum()/dist_ranks_avg['feature_result_conv4'].shape[0],
# 'backbone_conv5': 1-(dist_ranks_avg['feature_result_conv5'] < dist_ranks_thred['feature_result_conv5']).sum()/dist_ranks_avg['feature_result_conv5'].shape[0],
#
# 'tranf_c_x': 1-(dist_ranks_avg['feature_result_c_x'] < dist_ranks_thred['feature_result_c_x']).sum()/dist_ranks_avg['feature_result_c_x'].shape[0],
# 'tranf_r_x': 1-(dist_ranks_avg['feature_result_r_x'] < dist_ranks_thred['feature_result_r_x']).sum()/dist_ranks_avg['feature_result_r_x'].shape[0],
# 'tranf_c_z_k': 1-(dist_ranks_avg['feature_result_c_x'] < dist_ranks_thred['feature_result_c_x']).sum()/dist_ranks_avg['feature_result_c_x'].shape[0], # has to equal tranf_c_x
# 'tranf_r_z_k': 1-(dist_ranks_avg['feature_result_r_x'] < dist_ranks_thred['feature_result_r_x']).sum()/dist_ranks_avg['feature_result_r_x'].shape[0], # has to equal tranf_r_x
#
# 'head_bbox_conv3x3_1': 1-(dist_ranks_avg['feature_result_bbox_p5_conv1'] < dist_ranks_thred['feature_result_bbox_p5_conv1']).sum()/dist_ranks_avg['feature_result_bbox_p5_conv1'].shape[0],
# 'head_bbox_conv3x3_2': 1-(dist_ranks_avg['feature_result_bbox_p5_conv2'] < dist_ranks_thred['feature_result_bbox_p5_conv2']).sum()/dist_ranks_avg['feature_result_bbox_p5_conv2'].shape[0],
# 'head_bbox_conv3x3_3': 1-(dist_ranks_avg['feature_result_bbox_p5_conv3'] < dist_ranks_thred['feature_result_bbox_p5_conv3']).sum()/dist_ranks_avg['feature_result_bbox_p5_conv3'].shape[0],
#
# 'head_cls_conv3x3_1': 1-(dist_ranks_avg['feature_result_cls_p5_conv1'] < dist_ranks_thred['feature_result_cls_p5_conv1']).sum()/dist_ranks_avg['feature_result_cls_p5_conv1'].shape[0],
# 'head_cls_conv3x3_2': 1-(dist_ranks_avg['feature_result_cls_p5_conv2'] < dist_ranks_thred['feature_result_cls_p5_conv2']).sum()/dist_ranks_avg['feature_result_cls_p5_conv2'].shape[0],
# 'head_cls_conv3x3_3': 1-(dist_ranks_avg['feature_result_cls_p5_conv3'] < dist_ranks_thred['feature_result_cls_p5_conv3']).sum()/dist_ranks_avg['feature_result_cls_p5_conv3'].shape[0],
# }
#
# print(dist_cmprate0)

# dist_cmprate1 = {
# 'backbone_conv1': 0.6,
# 'backbone_conv2': 0.6,
# 'backbone_conv3': 0.6,
# 'backbone_conv4': 0.6,
# 'backbone_conv5': 0.6,
#
# 'tranf_c_x': 0.6,
# 'tranf_r_x': 0.6,
# 'tranf_c_z_k': 0.6, # has to equal tranf_c_x
# 'tranf_r_z_k': 0.6, # has to equal tranf_r_x
#
# 'head_bbox_conv3x3_1': 0.6,
# 'head_bbox_conv3x3_2': 0.6,
# 'head_bbox_conv3x3_3': 0.6,
#
# 'head_cls_conv3x3_1': 0.6,
# 'head_cls_conv3x3_2': 0.6,
# 'head_cls_conv3x3_3': 0.6,
# }

dist_cmprate = {
'backbone_conv1': 0.6,
'backbone_conv2': 0.6,
'backbone_conv3': 0.6,
'backbone_conv4': 0.6,
'backbone_conv5': 0.6,

'tranf_c_x': 0.6,
'tranf_r_x': 0.6,
'tranf_c_z_k': 0.6, # has to equal tranf_c_x
'tranf_r_z_k': 0.6, # has to equal tranf_r_x


'head_bbox_conv3x3_1': 0.6,
'head_bbox_conv3x3_2': 0.6,
'head_bbox_conv3x3_3': 0.6,

'head_cls_conv3x3_1': 0.6,
'head_cls_conv3x3_2': 0.6,
'head_cls_conv3x3_3': 0.6,
}
# dist_cmprate = {
# 'backbone_conv1': 0.8,
# 'backbone_conv2': 0.85,
# 'backbone_conv3': 0.85,
# 'backbone_conv4': 0.85,
# 'backbone_conv5': 1.0,
#
# 'tranf_c_x': 1.0,
# 'tranf_r_x': 1.0,
# 'tranf_c_z_k': 1.0, # has to equal tranf_c_x
# 'tranf_r_z_k': 0.6, # has to equal tranf_r_x
#
# 'head_bbox_conv3x3_1': 0.85,
# 'head_bbox_conv3x3_2': 0.6,
# 'head_bbox_conv3x3_3': 0.85,
#
# 'head_cls_conv3x3_1': 0.9,
# 'head_cls_conv3x3_2': 0.5,
# 'head_cls_conv3x3_3': 0.85,
# }