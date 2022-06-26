import json
import os

data = dict()
data['bike1']={"name":"bike1","path":"data_seq/UAV123@10fps/bike1","start_frame":1,"end_frame":1029}
data['bike2']={"name":"bike2","path":"data_seq/UAV123@10fps/bike2","start_frame":1,"end_frame":185}
data['bike3']={"name":"bike3","path":"data_seq/UAV123@10fps/bike3","start_frame":1,"end_frame":145}
data['bird1_1']={"name":"bird1_1","path":"data_seq/UAV123@10fps/bird1","start_frame":1,"end_frame":85}
data['bird1_2']={"name":"bird1_2","path":"data_seq/UAV123@10fps/bird1","start_frame":259,"end_frame":493}
data['bird1_3']={"name":"bird1_3","path":"data_seq/UAV123@10fps/bird1","start_frame":525,"end_frame":813}
data['boat1']={"name":"boat1","path":"data_seq/UAV123@10fps/boat1","start_frame":1,"end_frame":301}
data['boat2']={"name":"boat2","path":"data_seq/UAV123@10fps/boat2","start_frame":1,"end_frame":267}
data['boat3']={"name":"boat3","path":"data_seq/UAV123@10fps/boat3","start_frame":1,"end_frame":301}
data['boat4']={"name":"boat4","path":"data_seq/UAV123@10fps/boat4","start_frame":1,"end_frame":185}
data['boat5']={"name":"boat5","path":"data_seq/UAV123@10fps/boat5","start_frame":1,"end_frame":169}
data['boat6']={"name":"boat6","path":"data_seq/UAV123@10fps/boat6","start_frame":1,"end_frame":269}
data['boat7']={"name":"boat7","path":"data_seq/UAV123@10fps/boat7","start_frame":1,"end_frame":179}
data['boat8']={"name":"boat8","path":"data_seq/UAV123@10fps/boat8","start_frame":1,"end_frame":229}
data['boat9']={"name":"boat9","path":"data_seq/UAV123@10fps/boat9","start_frame":1,"end_frame":467}
data['building1']={"name":"building1","path":"data_seq/UAV123@10fps/building1","start_frame":1,"end_frame":157}
data['building2']={"name":"building2","path":"data_seq/UAV123@10fps/building2","start_frame":1,"end_frame":193}
data['building3']={"name":"building3","path":"data_seq/UAV123@10fps/building3","start_frame":1,"end_frame":277}
data['building4']={"name":"building4","path":"data_seq/UAV123@10fps/building4","start_frame":1,"end_frame":263}
data['building5']={"name":"building5","path":"data_seq/UAV123@10fps/building5","start_frame":1,"end_frame":161}
data['car1_1']={"name":"car1_1","path":"data_seq/UAV123@10fps/car1","start_frame":1,"end_frame":251}
data['car1_2']={"name":"car1_2","path":"data_seq/UAV123@10fps/car1","start_frame":251,"end_frame":543}
data['car1_3']={"name":"car1_3","path":"data_seq/UAV123@10fps/car1","start_frame":543,"end_frame":877}
data['car2']={"name":"car2","path":"data_seq/UAV123@10fps/car2","start_frame":1,"end_frame":441}
data['car3']={"name":"car3","path":"data_seq/UAV123@10fps/car3","start_frame":1,"end_frame":573}
data['car4']={"name":"car4","path":"data_seq/UAV123@10fps/car4","start_frame":1,"end_frame":449}
data['car5']={"name":"car5","path":"data_seq/UAV123@10fps/car5","start_frame":1,"end_frame":249}
data['car6_1']={"name":"car6_1","path":"data_seq/UAV123@10fps/car6","start_frame":1,"end_frame":163}
data['car6_2']={"name":"car6_2","path":"data_seq/UAV123@10fps/car6","start_frame":163,"end_frame":603}
data['car6_3']={"name":"car6_3","path":"data_seq/UAV123@10fps/car6","start_frame":603,"end_frame":985}
data['car6_4']={"name":"car6_4","path":"data_seq/UAV123@10fps/car6","start_frame":985,"end_frame":1309}
data['car6_5']={"name":"car6_5","path":"data_seq/UAV123@10fps/car6","start_frame":1309,"end_frame":1621}
data['car7']={"name":"car7","path":"data_seq/UAV123@10fps/car7","start_frame":1,"end_frame":345}
data['car8_1']={"name":"car8_1","path":"data_seq/UAV123@10fps/car8","start_frame":1,"end_frame":453}
data['car8_2']={"name":"car8_2","path":"data_seq/UAV123@10fps/car8","start_frame":453,"end_frame":859}
data['car9']={"name":"car9","path":"data_seq/UAV123@10fps/car9","start_frame":1,"end_frame":627}
data['car10']={"name":"car10","path":"data_seq/UAV123@10fps/car10","start_frame":1,"end_frame":469}
data['car11']={"name":"car11","path":"data_seq/UAV123@10fps/car11","start_frame":1,"end_frame":113}
data['car12']={"name":"car12","path":"data_seq/UAV123@10fps/car12","start_frame":1,"end_frame":167}
data['car13']={"name":"car13","path":"data_seq/UAV123@10fps/car13","start_frame":1,"end_frame":139}
data['car14']={"name":"car14","path":"data_seq/UAV123@10fps/car14","start_frame":1,"end_frame":443}
data['car15']={"name":"car15","path":"data_seq/UAV123@10fps/car15","start_frame":1,"end_frame":157}
data['car16_1']={"name":"car16_1","path":"data_seq/UAV123@10fps/car16","start_frame":1,"end_frame":139}
data['car16_2']={"name":"car16_2","path":"data_seq/UAV123@10fps/car16","start_frame":139,"end_frame":665}
data['car17']={"name":"car17","path":"data_seq/UAV123@10fps/car17","start_frame":1,"end_frame":353}
data['car18']={"name":"car18","path":"data_seq/UAV123@10fps/car18","start_frame":1,"end_frame":403}
data['group1_1']={"name":"group1_1","path":"data_seq/UAV123@10fps/group1","start_frame":1,"end_frame":445}
data['group1_2']={"name":"group1_2","path":"data_seq/UAV123@10fps/group1","start_frame":445,"end_frame":839}
data['group1_3']={"name":"group1_3","path":"data_seq/UAV123@10fps/group1","start_frame":839,"end_frame":1309}
data['group1_4']={"name":"group1_4","path":"data_seq/UAV123@10fps/group1","start_frame":1309,"end_frame":1625}
data['group2_1']={"name":"group2_1","path":"data_seq/UAV123@10fps/group2","start_frame":1,"end_frame":303}
data['group2_2']={"name":"group2_2","path":"data_seq/UAV123@10fps/group2","start_frame":303,"end_frame":591}
data['group2_3']={"name":"group2_3","path":"data_seq/UAV123@10fps/group2","start_frame":591,"end_frame":895}
data['group3_1']={"name":"group3_1","path":"data_seq/UAV123@10fps/group3","start_frame":1,"end_frame":523}
data['group3_2']={"name":"group3_2","path":"data_seq/UAV123@10fps/group3","start_frame":523,"end_frame":943}
data['group3_3']={"name":"group3_3","path":"data_seq/UAV123@10fps/group3","start_frame":943,"end_frame":1457}
data['group3_4']={"name":"group3_4","path":"data_seq/UAV123@10fps/group3","start_frame":1457,"end_frame":1843}
data['person1']={"name":"person1","path":"data_seq/UAV123@10fps/person1","start_frame":1,"end_frame":267}
data['person2_1']={"name":"person2_1","path":"data_seq/UAV123@10fps/person2","start_frame":1,"end_frame":397}
data['person2_2']={"name":"person2_2","path":"data_seq/UAV123@10fps/person2","start_frame":397,"end_frame":875}
data['person3']={"name":"person3","path":"data_seq/UAV123@10fps/person3","start_frame":1,"end_frame":215}
data['person4_1']={"name":"person4_1","path":"data_seq/UAV123@10fps/person4","start_frame":1,"end_frame":501}
data['person4_2']={"name":"person4_2","path":"data_seq/UAV123@10fps/person4","start_frame":501,"end_frame":915}
data['person5_1']={"name":"person5_1","path":"data_seq/UAV123@10fps/person5","start_frame":1,"end_frame":293}
data['person5_2']={"name":"person5_2","path":"data_seq/UAV123@10fps/person5","start_frame":293,"end_frame":701}
data['person6']={"name":"person6","path":"data_seq/UAV123@10fps/person6","start_frame":1,"end_frame":301}
data['person7_1']={"name":"person7_1","path":"data_seq/UAV123@10fps/person7","start_frame":1,"end_frame":417}
data['person7_2']={"name":"person7_2","path":"data_seq/UAV123@10fps/person7","start_frame":417,"end_frame":689}
data['person8_1']={"name":"person8_1","path":"data_seq/UAV123@10fps/person8","start_frame":1,"end_frame":359}
data['person8_2']={"name":"person8_2","path":"data_seq/UAV123@10fps/person8","start_frame":359,"end_frame":509}
data['person9']={"name":"person9","path":"data_seq/UAV123@10fps/person9","start_frame":1,"end_frame":221}
data['person10']={"name":"person10","path":"data_seq/UAV123@10fps/person10","start_frame":1,"end_frame":341}
data['person11']={"name":"person11","path":"data_seq/UAV123@10fps/person11","start_frame":1,"end_frame":241}
data['person12_1']={"name":"person12_1","path":"data_seq/UAV123@10fps/person12","start_frame":1,"end_frame":201}
data['person12_2']={"name":"person12_2","path":"data_seq/UAV123@10fps/person12","start_frame":201,"end_frame":541}
data['person13']={"name":"person13","path":"data_seq/UAV123@10fps/person13","start_frame":1,"end_frame":295}
data['person14_1']={"name":"person14_1","path":"data_seq/UAV123@10fps/person14","start_frame":1,"end_frame":283}
data['person14_2']={"name":"person14_2","path":"data_seq/UAV123@10fps/person14","start_frame":283,"end_frame":605}
data['person14_3']={"name":"person14_3","path":"data_seq/UAV123@10fps/person14","start_frame":605,"end_frame":975}
data['person15']={"name":"person15","path":"data_seq/UAV123@10fps/person15","start_frame":1,"end_frame":447}
data['person16']={"name":"person16","path":"data_seq/UAV123@10fps/person16","start_frame":1,"end_frame":383}
data['person17_1']={"name":"person17_1","path":"data_seq/UAV123@10fps/person17","start_frame":1,"end_frame":501}
data['person17_2']={"name":"person17_2","path":"data_seq/UAV123@10fps/person17","start_frame":501,"end_frame":783}
data['person18']={"name":"person18","path":"data_seq/UAV123@10fps/person18","start_frame":1,"end_frame":465}
data['person19_1']={"name":"person19_1","path":"data_seq/UAV123@10fps/person19","start_frame":1,"end_frame":415}
data['person19_2']={"name":"person19_2","path":"data_seq/UAV123@10fps/person19","start_frame":415,"end_frame":931}
data['person19_3']={"name":"person19_3","path":"data_seq/UAV123@10fps/person19","start_frame":931,"end_frame":1453}
data['person20']={"name":"person20","path":"data_seq/UAV123@10fps/person20","start_frame":1,"end_frame":595}
data['person21']={"name":"person21","path":"data_seq/UAV123@10fps/person21","start_frame":1,"end_frame":163}
data['person22']={"name":"person22","path":"data_seq/UAV123@10fps/person22","start_frame":1,"end_frame":67}
data['person23']={"name":"person23","path":"data_seq/UAV123@10fps/person23","start_frame":1,"end_frame":133}
data['truck1']={"name":"truck1","path":"data_seq/UAV123@10fps/truck1","start_frame":1,"end_frame":155}
data['truck2']={"name":"truck2","path":"data_seq/UAV123@10fps/truck2","start_frame":1,"end_frame":129}
data['truck3']={"name":"truck3","path":"data_seq/UAV123@10fps/truck3","start_frame":1,"end_frame":179}
data['truck4_1']={"name":"truck4_1","path":"data_seq/UAV123@10fps/truck4","start_frame":1,"end_frame":193}
data['truck4_2']={"name":"truck4_2","path":"data_seq/UAV123@10fps/truck4","start_frame":193,"end_frame":421}
data['uav1_1']={"name":"uav1_1","path":"data_seq/UAV123@10fps/uav1","start_frame":1,"end_frame":519}
data['uav1_2']={"name":"uav1_2","path":"data_seq/UAV123@10fps/uav1","start_frame":519,"end_frame":793}
data['uav1_3']={"name":"uav1_3","path":"data_seq/UAV123@10fps/uav1","start_frame":825,"end_frame":1157}
data['uav2']={"name":"uav2","path":"data_seq/UAV123@10fps/uav2","start_frame":1,"end_frame":45}
data['uav3']={"name":"uav3","path":"data_seq/UAV123@10fps/uav3","start_frame":1,"end_frame":89}
data['uav4']={"name":"uav4","path":"data_seq/UAV123@10fps/uav4","start_frame":1,"end_frame":53}
data['uav5']={"name":"uav5","path":"data_seq/UAV123@10fps/uav5","start_frame":1,"end_frame":47}
data['uav6']={"name":"uav6","path":"data_seq/UAV123@10fps/uav6","start_frame":1,"end_frame":37}
data['uav7']={"name":"uav7","path":"data_seq/UAV123@10fps/uav7","start_frame":1,"end_frame":125}
data['uav8']={"name":"uav8","path":"data_seq/UAV123@10fps/uav8","start_frame":1,"end_frame":101}
data['wakeboard1']={"name":"wakeboard1","path":"data_seq/UAV123@10fps/wakeboard1","start_frame":1,"end_frame":141}
data['wakeboard2']={"name":"wakeboard2","path":"data_seq/UAV123@10fps/wakeboard2","start_frame":1,"end_frame":245}
data['wakeboard3']={"name":"wakeboard3","path":"data_seq/UAV123@10fps/wakeboard3","start_frame":1,"end_frame":275}
data['wakeboard4']={"name":"wakeboard4","path":"data_seq/UAV123@10fps/wakeboard4","start_frame":1,"end_frame":233}
data['wakeboard5']={"name":"wakeboard5","path":"data_seq/UAV123@10fps/wakeboard5","start_frame":1,"end_frame":559}
data['wakeboard6']={"name":"wakeboard6","path":"data_seq/UAV123@10fps/wakeboard6","start_frame":1,"end_frame":389}
data['wakeboard7']={"name":"wakeboard7","path":"data_seq/UAV123@10fps/wakeboard7","start_frame":1,"end_frame":67}
data['wakeboard8']={"name":"wakeboard8","path":"data_seq/UAV123@10fps/wakeboard8","start_frame":1,"end_frame":515}
data['wakeboard9']={"name":"wakeboard9","path":"data_seq/UAV123@10fps/wakeboard9","start_frame":1,"end_frame":119}
data['wakeboard10']={"name":"wakeboard10","path":"data_seq/UAV123@10fps/wakeboard10","start_frame":1,"end_frame":157}
data['car1_s']={"name":"car1_s","path":"data_seq/UAV123@10fps/car1_s","start_frame":1,"end_frame":492}
data['car2_s']={"name":"car2_s","path":"data_seq/UAV123@10fps/car2_s","start_frame":1,"end_frame":107}
data['car3_s']={"name":"car3_s","path":"data_seq/UAV123@10fps/car3_s","start_frame":1,"end_frame":434}
data['car4_s']={"name":"car4_s","path":"data_seq/UAV123@10fps/car4_s","start_frame":1,"end_frame":277}
data['person1_s']={"name":"person1_s","path":"data_seq/UAV123@10fps/person1_s","start_frame":1,"end_frame":534}
data['person2_s']={"name":"person2_s","path":"data_seq/UAV123@10fps/person2_s","start_frame":1,"end_frame":84}
data['person3_s']={"name":"person3_s","path":"data_seq/UAV123@10fps/person3_s","start_frame":1,"end_frame":169}

meta_file='/home/lsw/lsw/siamfc++/siamfcpp/evaluation/got_benchmark/datasets/uav123.json'
with open(meta_file) as f:
    seq_metas = json.load(f)
seq_metas['UAV123@10FPS']=dict()
for video in data.keys():
    seq_metas['UAV123@10FPS'][video] = {
        'start_frame': data[video]['start_frame'],
        'end_frame': data[video]['end_frame'],
        'folder_name': data[video]['path'],
    }
seq_metas['UAV123@10fps'] = seq_metas['UAV123@10FPS']
json.dump(seq_metas, open(meta_file, "w"))

meta_file='/home/lsw/lsw/siamfc++/siamfcpp/evaluation/got_benchmark/datasets/uav123.json'
# with open(meta_file) as f:
#     seq_metas = json.load(f)
# seq_metas['UAV123@10FPS']=dict()
# datapath = '/home/lsw/data/Dataset_UAV123_10fps/UAV123_10fps/data_seq/UAV123_10fps'
# videos = os.listdir(datapath)
# videos.sort()
# for video in videos:
#     images = os.listdir(os.path.join(datapath,video))
#     images.sort()
#     seq_metas['UAV123@10FPS'][video]={
#         'start_frame': 1,
#         'end_frame': images.__len__(),
#         'folder_name': video,
#     }
#
# json.dump(seq_metas, open(meta_file, "w"))
#
# meta_file='/home/lsw/data/Dataset_UAV123_10fps/UAV123_10fps'
#
# with open(meta_file) as f:
#     seq_metas = json.load(f)