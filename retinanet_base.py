"""rewrite from https://github.com/fizyr/keras-retinanet"""


import argparse
import os
os.environ['HDF5_DISABLE_VERSION_CHECK'] = '1'
import sys
import warnings
import numpy as np
import PIL.Image

import keras
import keras.preprocessing.image
import tensorflow as tf

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'keras_retinanet', 'keras_retinanet'))
    import keras_retinanet.bin  # noqa: F401
    __package__ = "keras_retinanet.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from keras_retinanet import models, layers
from keras_retinanet.models.retinanet import retinanet#_bbox_attack
from keras_retinanet import initializers
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
import keras_resnet.models


def create_pyramid_features_attack(C3, C4, C5, feature_size=256):
    # upsample C5 to get P5 from the FPN paper
    P5           = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C5_reduced')(C5)
    P5_upsampled = keras.layers.UpSampling2D(2, name='P5_upsampled')(P5) #layers.UpsampleLike(name='P5_upsampled')([P5, C4])
    P5           = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P5')(P5)

    # add P5 elementwise to C4
    P4           = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced')(C4)
    P4           = keras.layers.Add(name='P4_merged')([P5_upsampled, P4])
    P4_upsampled = keras.layers.UpSampling2D(2, name='P4_upsampled')(P4) #layers.UpsampleLike(name='P4_upsampled')([P4, C3])
    P4           = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4')(P4)

    # add P4 elementwise to C3
    P3 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced')(C3)
    P3 = keras.layers.Add(name='P3_merged')([P4_upsampled, P3])
    P3 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3')(P3)

    # "P6 is obtained via a 3x3 stride-2 conv on C5"
    P6 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P6')(C5)

    # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
    P7 = keras.layers.Activation('relu', name='C6_relu')(P6)
    P7 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P7')(P7)

    return [P3, P4, P5, P6, P7]


def attack_classification_model(
    num_classes,
    num_anchors,
    pyramid_feature_size=256,
    prior_probability=0.01,
    classification_feature_size=256,
    name='classification_submodel'
):
    options = {
        'kernel_size' : 3,
        'strides'     : 1,
        'padding'     : 'same',
    }

    def tensor_exists(sub_net_id):
        try: tf.get_default_graph().get_tensor_by_name('%s_pyramid_classification_0/Relu:0' % sub_net_id)
        except KeyError: return False
        return True

    def forward(inputs):
        outputs = inputs
        sub_net_id = 0
        while tensor_exists(sub_net_id): sub_net_id += 1
        
        for i in range(4):
            outputs = keras.layers.Conv2D(
                filters=classification_feature_size,
                activation='relu',
                name='{}_pyramid_classification_{}'.format(sub_net_id, i),
                kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
                bias_initializer='zeros',
                **options
            )(outputs)

        outputs = keras.layers.Conv2D(
            filters=num_classes * num_anchors,
            kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
            bias_initializer='zeros',#initializers.PriorProbability(probability=prior_probability),
            name='{}_pyramid_classification'.format(sub_net_id),
            **options
        )(outputs)

        # reshape output and apply sigmoid
        outputs = keras.layers.Reshape((-1, num_classes))(outputs)
        outputs = keras.layers.Activation('sigmoid')(outputs)
        return outputs
    return lambda x: forward(x)


def create_models_attack(
    inputs                = None,
    model                 = None,
    nms                   = True,
    class_specific_filter = True,
    name                  = 'retinanet-attack',
    anchor_params         = None,
    nms_threshold         = 0.5,
    score_threshold       = 0.05,
    max_detections        = 300,
    parallel_iterations   = 32,
    **kwargs):
    # backbone_retinanet = models.backbone('resnet50').retinanet; weights='../snapshots/resnet50_coco_best_v2.1.0.h5'
    # model = model_with_weights(backbone_retinanet(80, num_anchors=None, modifier=None), weights=weights, skip_mismatch=True)
    # if no anchor parameters are passed, use default values
    num_classes = 80
    num_anchors = 9
    if inputs is None: inputs = keras.layers.Input(shape=(None, None, 3))
    resnet = keras_resnet.models.ResNet50(inputs, include_top=False, freeze_bn=True)
    backbone_layers = resnet.outputs[1:]
    
    # create RetinaNet model
    model = retinanet(
        inputs=inputs,
        backbone_layers=backbone_layers,
        num_classes=num_classes, 
        submodels=[('classification', attack_classification_model(num_classes, num_anchors))], 
        name='retinanet_attack',
        create_pyramid_features=create_pyramid_features_attack,
        **kwargs)
    return keras.models.Model(inputs=inputs, 
        outputs=keras.layers.Reshape((-1,), name='attack_reshape')(model.outputs[0]), 
        name=name)


def load_net(model_path):
    model_detect = models.load_model(model_path, backbone_name='resnet50')
    pyramid_classification_x = []
    for i in range(4):
        pyramid_classification_x.append(model_detect.get_layer("classification_submodel").get_layer("pyramid_classification_%d" % i).get_weights())
    pyramid_classification = model_detect.get_layer("classification_submodel").get_layer("pyramid_classification").get_weights()

    inputs = keras.layers.Input(batch_shape=(1, 416, 416, 3))
    model = create_models_attack(inputs)
    model.load_weights(model_path, by_name=True, skip_mismatch=True)
    
    for sub_net_id in range(5):
        model.get_layer("%d_pyramid_classification" % sub_net_id).set_weights(pyramid_classification)
        for i in range(4):
            model.get_layer("%d_pyramid_classification_%d" % (sub_net_id, i)).set_weights(pyramid_classification_x[i])
    #model.summary()
    return model, model_detect


def read_image(path, size):
    """
    image_ori:    original image in BGR                                -> (W, H, 3)
    image_resized: resized image in BGR, long side = size              -> (W1, size, 3) or (size, H1, 3)
    image_input:  padded square image with batch dimension             -> (1, size, size, 3)
    val_image:    bool indexes where there is contents in image_input  -> (size, size, 3)
    image_input[0][val_image].reshape(image_resized.shape)[:, :, ::-1] -> (W, H, 3)  resized image in RGB
    """
    image_ori = read_image_bgr(path)
    image_resized, scale = resize_image(image_ori, max_side=size)
    image = preprocess_image(image_resized)
    image_input = np.zeros((1, size, size, 3), dtype=np.float32)
    val_image = np.zeros((1, size, size, 3), dtype=bool)
    if image.shape[1] >= image.shape[0]: val_image[0, int((size-image.shape[0])/2): image.shape[0]+int((size-image.shape[0])/2), :, :] = True
    else:                                val_image[0, :, int((size-image.shape[1])/2): image.shape[1]+int((size-image.shape[1])/2), :] = True
    image_input[val_image] = image.reshape(image_input[val_image].shape)
    return image_resized, image_input, val_image[0]


def test_lrp():
    sess = tf.InteractiveSession()
    model, _ = load_net()
    index_place = tf.placeholder(tf.int32, [None, ])
    analysis = build_lrp(model, index_place)

    def get_index(pred):
        pred_reshape = pred.reshape(-1, 80) # 32526 * 80
        pred_label = np.argmax(pred_reshape, axis=1)
        pred_score = np.max(pred_reshape, axis=1)
        indexes = np.argsort(pred_score)[::-1] * 80 + pred_label
        return indexes[:20]

    size = 416
    test_num = 100
    result_dir = 'LRPRetina'
    os.makedirs(result_dir, exist_ok=True)
    for i, file in enumerate(sorted(os.listdir(paths['Val']))[:test_num]):
        image_resized, image_input, val_image = read_image(paths['Val'] + '/' + file, size)
        pred = model.predict(image_input)
        indexes = get_index(pred)
        feed_dict = {model.input: image_input, index_place: indexes}
        result = visualize_lrp(sess.run(analysis, feed_dict), size=size)
        PIL.Image.fromarray(result[val_image].reshape(image_resized.shape)).save(result_dir + "/" + file)
        print('[ Done %d/%d ]' % (i+1, test_num), end='\r')


def detect_image(model_detect, image_input, image_resized, val_image):
    labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
    boxes, scores, labels = model_detect.predict_on_batch(image_input)
    size = image_input.shape[2]
    draw = np.zeros((size, size, 3), dtype=np.uint8)
    #image_resized = image_input[0][val_image].reshape(resized)[:, :, ::-1] ####
    draw[val_image] = image_resized.reshape(draw[val_image].shape)

    bbox_number = 0
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        if score < 0.5: break # scores are sorted so we can break
        bbox_number += 1
        color = label_color(label)
        b = box.astype(int)
        draw_box(draw, b, color=color)
        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)
    detection = PIL.Image.fromarray(draw[val_image].reshape(image_resized.shape))
    return detection, bbox_number