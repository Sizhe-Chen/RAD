# Environment
```
conda create -n rad python=3.6.2
conda activate rad
conda install tensorflow-gpu==1.13.1
pip install pillow==6.2.1 opencv-python==4.1.2.30 scikit-image==0.15.0 matplotlib==2.2.2
```
* COCO/val2017 [download](http://cocodataset.org)
* COCO/annotations/instances_val2017_resized.json

# Attact YOLOv3
* model_data/yolo.h5 [download](https://pjreddie.com/media/files/yolov3.weights)  [config](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg) [convert](https://github.com/qqwweee/keras-yolo3/blob/master/convert.py)

* Run default parameters on GPU 0
```
python yolov3.py 0
```
* Run customized parameters on GPU 1
```
python yolov3.py 1 --alpha=1 --epsilon=10 --num_iteration=5
```

# Attact RetinaNet
* model_data/resnet50_coco_best_v2.1.0.h5 [download](https://github.com/fizyr/keras-retinanet/releases/download/0.5.1/resnet50_coco_best_v2.1.0.h5)
* install [keras-retinanet](https://github.com/fizyr/keras-retinanet)
* Run default parameters on GPU 0
```
python retinanet.py 0
```
* Run customized parameters on GPU 1
```
python retinanet.py 1 --alpha=1 --epsilon=10 --num_iteration=5
```

# Test
* pytorch 1.3.1
* install [mmdetection 1.1.0](https://github.com/open-mmlab/mmdetection)
* models [download](https://mmdetection.readthedocs.io/en/latest/model_zoo.html)
* Test on GPU0 for 2020-06-09-15-19-07_RAD_SI_YOLOv3_Index0to5000_Eps2_Iter10
```
python test.py 2020-06-09-15-19-07_RAD_SI_YOLOv3_Index0to5000_Eps2_Iter10 0
```
