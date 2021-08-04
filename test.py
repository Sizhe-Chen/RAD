import os
import cv2
import json
import time
import shutil
import argparse
import numpy as np
import PIL.Image
from copy import deepcopy
import mmcv
from mmdet.apis import init_detector, inference_detector, show_result

# install mmdet v1 in https://github.com/open-mmlab/mmdetection
# download correspongding pretrained models from https://mmdetection.readthedocs.io/en/latest/model_zoo.html


config_dir = 'configs'
config_files = {
    'ssd': config_dir + '/ssd512_coco.py',
    'faster_rcnn': config_dir + '/faster_rcnn_r101_fpn_1x.py',
    'mask_rcnn': config_dir + '/mask_rcnn_x101_64x4d_fpn_1x.py',
    'retinanet': config_dir + '/retinanet_r101_fpn_1x.py',
    'cascade_rcnn': config_dir + '/cascade_rcnn_r101_fpn_1x.py',
    'cascade_mask_rcnn': config_dir + '/cascade_mask_rcnn_x101_64x4d_fpn_1x.py',
    'htc': config_dir + '/htc/htc_x101_64x4d_fpn_20e_16gpu.py',
}
config_files_ori = deepcopy(config_files)

checkpoint_dir = 'models'
checkpoint_files = {
    'ssd': checkpoint_dir + '/ssd512_coco_vgg16_caffe_120e_20181221-d48b0be8.pth',
    'faster_rcnn': checkpoint_dir + '/faster_rcnn_r101_fpn_2x_20181129-73e7ade7.pth',
    'mask_rcnn': checkpoint_dir + '/mask_rcnn_x101_64x4d_fpn_1x_20181218-cb159987.pth',
    'retinanet': checkpoint_dir + '/retinanet_r101_fpn_2x_20181129-72c14526.pth',
    'cascade_rcnn': checkpoint_dir + '/cascade_rcnn_r101_fpn_20e_20181129-b46dcede.pth',
    'cascade_mask_rcnn': checkpoint_dir + '/cascade_mask_rcnn_x101_64x4d_fpn_20e_20181218-630773a7.pth',
    'htc': checkpoint_dir + '/htc_x101_64x4d_fpn_20e_20190408-497f2561.pth',
}

model_order = list(config_files.keys())
assert model_order == list(checkpoint_files.keys())

paths = {'Annot': 'COCO/annotations', 'mmdet': 'mmdetection/tools/test.py'}
for key in paths: assert os.path.exists(paths[key]), paths[key] + ' does not exist'
for key in config_files: assert os.path.exists(config_files[key]), config_files[key] + ' does not exist'
for key in checkpoint_files: assert os.path.exists(checkpoint_files[key]), checkpoint_files[key] + ' does not exist'
dirs = ['adv', 'cache', 'index', 'detection']
mask = ['mask_rcnn', 'cascade_mask_rcnn', 'htc']


def calculate_rmse(dir_name):
    # calculate the RMSE for all samples
    rmses = []
    for root, _, files in os.walk(dir_name):
        if 'sample_adv.png' not in files or 'sample_ori.png' not in files: continue#print('Not found in', root); continue
        adv = np.array(PIL.Image.open(root + '/sample_adv.png')).astype(np.float32)
        ori = np.array(PIL.Image.open(root + '/sample_ori.png').resize((adv.shape[1], adv.shape[0]))).astype(np.float32)
        rmse = np.sqrt(np.mean(np.square(adv-ori)))
        if rmse < 20: rmses.append(rmse)
    print('RMSE is %.3f in %d samples' % (sum(rmses)/(len(rmses)+0.001), len(rmses)))


def re_annotation(dir_name):
    data = json.load(open(paths['Annot'] + '/instances_val2017.json', 'r', encoding='utf-8'))
    scales = {}
    size = 416 if ('MaskRCNN' not in dir_name) else 448

    existing = [] # record the existing samples
    for file in os.listdir(dir_name + '/' + dirs[0]): existing.append(file)

    # record the resized scale for each sample
    abandoned = []
    for i in range(len(data['images'])):
        new_name = data['images'][i]['file_name'][:-4] + '.png'
        if new_name not in existing: abandoned.append(i)
        data['images'][i]['file_name'] = new_name
        ih, iw = data['images'][i]['height'], data['images'][i]['width']
        scale = min(size/ih, size/iw)
        data['images'][i]['height'], data['images'][i]['width'] = int(ih*scale), int(iw*scale)
        scales[data['images'][i]['id']] = scale
    for i, index in enumerate(abandoned): data['images'].remove(data['images'][index-i])

    # resize the annotations for detection and segmentation
    abandoned = []
    for i in range(len(data['annotations'])):
        image_id = data['annotations'][i]['image_id']
        scale = scales[image_id]
        new_name = str(image_id).zfill(12) + '.png'
        if new_name not in existing: abandoned.append(i)
        for j in range(len(data['annotations'][i]['segmentation'])):
            try:
                data['annotations'][i]['segmentation'][j] = list(np.array(data['annotations'][i]['segmentation'][j])*scale)
            except KeyError: continue
        data['annotations'][i]['area'] = data['annotations'][i]['area'] * (scale ** 2)
        data['annotations'][i]['bbox'] = list(np.array(data['annotations'][i]['bbox']) * scale)
    for i, index in enumerate(abandoned): data['annotations'].remove(data['annotations'][index-i])

    result_dir = dir_name + '/' + dirs[1]
    os.makedirs(result_dir, exist_ok=True)
    json.dump(data, open(result_dir + '/instances_val2017_resized.json', 'w', encoding='utf-8'))


def change_config(model_name, dir_name):
    global config_files, config_files_ori
    # change the config files to test the generated adversarial samples
    ori_config = config_files_ori[model_name]
    py_file = open(ori_config, 'r').read()
    py_file = py_file.replace("data_root + 'val2017/'", "'" + dir_name + "/" + dirs[0] + "'")
    py_file = py_file.replace("data_root + 'annotations/instances_val2017.json'", "'" + dir_name + "/" + dirs[1] + "/instances_val2017_resized.json'")

    new_config = dir_name + '/' + dirs[1] + '/' + os.path.basename(config_files_ori[model_name])
    with open(new_config, 'w') as f: f.write(py_file)
    config_files[model_name] = new_config


def test_index(model_name, dir_name, metric='bbox', unique_metric=False):
    # test the performance of mmdetection models on adversarial samples
    if 'MaskRCNN' in dir_name and model_name == 'mask_rcnn': return
    result_dir = dir_name + '/' + dirs[2]
    os.makedirs(result_dir, exist_ok=True)
    file_name = 'test.py' if not unique_metric else 'test_ours.py'
    command = 'python mmdetection/tools/%s %s %s --out %s --eval %s' % \
        (file_name, config_files[model_name], checkpoint_files[model_name], result_dir + '/' + model_name + '.pickle', metric if (model_name not in mask or unique_metric) else (metric + ' segm'))
    print(command)
    os.system(command)


def test_bbox(model_name, dir_name, sample_num):
    # generate visual results for sample_num samples
    source_dir = dir_name + '/' + dirs[0]
    result_dir = dir_name + '/' + dirs[3] + '/' + model_name
    os.makedirs(result_dir, exist_ok=True)

    config_file = config_files[model_name]
    checkpoint_file = checkpoint_files[model_name]
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    model_id, model_num = model_order.index(model_name) + 1, len(model_order)

    for i, file in enumerate(sorted(os.listdir(source_dir), key=lambda x: int(os.path.splitext(os.path.splitext(x)[0])[0]))):
        if i >= sample_num: break
        img = source_dir + '/' + file
        try:
            result = inference_detector(model, img)
            final = show_result(img, result, model.CLASSES, show=False)
        except: continue
        PIL.Image.fromarray(final[:, :, ::-1]).save(result_dir + '/' + os.path.splitext(file)[0] + '.png')
        print('[ Model %d/%d %s ] [ No %d/%d ] [ File %s ]' % (model_id, model_num, model_name, i+1, sample_num, file), end='\r')


def test_pipeline():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('dataset', type=str, help='dir name of the tested experiment')
    parser.add_argument('gpu_id', help='GPU(s) used')
    args, _ = parser.parse_known_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    assert os.path.exists(args.dataset)
    print('Calculating RMSE for', args.dataset, 'with', len(os.listdir(args.dataset + '/' + dirs[0])), 'samples...')
    calculate_rmse(args.dataset)
    re_annotation(dir_name=args.dataset) # resize annotations for existing adversarial samples to dir_name/dirs[1]/instances_val2017_resized.json
    # change paths in config file and saved in dir_name/dirs[1]/.py
    for model_name in config_files: change_config(model_name=model_name, dir_name=args.dataset)
    # run mAP, mAR for samples to dir_name/dirs[2]
    for model_name in config_files: test_index(model_name=model_name, dir_name=args.dataset)
    # get bbox detection result images in dir_name/dirs[3]/model_name
    for model_name in config_files: test_bbox(model_name=model_name, dir_name=args.dataset, sample_num=500)
    # run accuracy, IoU for samples to dir_name/dirs[2]
    for model_name in config_files: test_index(model_name=model_name, dir_name=args.dataset, unique_metric=True)


if __name__ == "__main__":
    test_pipeline()