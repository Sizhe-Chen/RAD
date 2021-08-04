import json
from collections import defaultdict

name_box_id = defaultdict(list)
id_name = dict()
base_path = "../2020-05-18-16-59-36_AoA_SI_YOLO3_Index0to5000_Eps2_Iter10"
json_path = base_path + "/cache/instances_val2017_resized.json"
img_path = base_path + "/adv-resized/"

f = open(json_path, encoding='utf-8')
data = json.load(f)

annotations = data['annotations']
for ant in annotations:
    id = ant['image_id']
    name = img_path + '%012d.png' % id
    cat = ant['category_id']

    if cat >= 1 and cat <= 11:
        cat = cat - 1
    elif cat >= 13 and cat <= 25:
        cat = cat - 2
    elif cat >= 27 and cat <= 28:
        cat = cat - 3
    elif cat >= 31 and cat <= 44:
        cat = cat - 5
    elif cat >= 46 and cat <= 65:
        cat = cat - 6
    elif cat == 67:
        cat = cat - 7
    elif cat == 70:
        cat = cat - 9
    elif cat >= 72 and cat <= 82:
        cat = cat - 10
    elif cat >= 84 and cat <= 90:
        cat = cat - 11

    name_box_id[name].append([ant['bbox'], cat])

f = open('adv_train_coco_yolov3.txt', 'w')
for key in name_box_id.keys():
    f.write(key)
    box_infos = name_box_id[key]
    for info in box_infos:
        x_min = int(info[0][0])
        y_min = int(info[0][1])
        x_max = x_min + int(info[0][2])
        y_max = y_min + int(info[0][3])

        box_info = " %d,%d,%d,%d,%d" % (
            x_min, y_min, x_max, y_max, int(info[1]))
        f.write(box_info)
    f.write('\n')
f.close()