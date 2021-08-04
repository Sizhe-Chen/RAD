"""
Retrain the YOLO model for your own dataset.
"""
import os
import numpy as np
import time
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils.multi_gpu_utils import multi_gpu_model # https://www.jianshu.com/p/4203a6435ab5

from yolov3_base import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss, get_random_data

def get_time(middle='-'): return time.strftime('%Y-%m-%d' + middle + '%H-%M-%S', time.localtime(time.time()))

def _main():
    annotation_path = '../adv_train_coco_yolov3.txt'
    epoch = 10
    batch_size = 128
    log_dir = get_time() + '_AdvTrain_YOLOv3_Epoch%d_Batch%d' % (epoch, batch_size) #'logs/000/'
    classes_path = 'model_data/coco_classes.txt'
    anchors_path = 'model_data/yolo_anchors.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = (416, 416) # multiple of 32, hw

    is_tiny_version = len(anchors) == 6 # default setting
    if is_tiny_version:
        model = create_tiny_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path='model_data/tiny_yolo_weights.h5')
    else:
        model = create_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path='model_data/yolo.h5') # make sure you know what you freeze

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + '/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=1) #period=3
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    multi_model = multi_gpu_model(model, gpus=4) ###########
    def multi_loss(y_true, y_pred):  return K.sum(y_pred)/K.cast(K.shape(y_pred)[0],K.dtype(y_pred)) ############

    # https://blog.csdn.net/pestzhang/article/details/102815068
    """
    qqwweee/keras-yolo3模型默认采用的是一块GPU，在直接使用model = multi_gpu_model(model,gpus=N)时，模型会报错tensorflow.python.framework.errors_impl.InvalidArgumentError: Can’t concatenate scalars (use tf.stack instead) for ‘yolo_loss_1/concat’ (op: ‘ConcatV2’) with input shapes: [], [], [], [].，这是因为该模型的设计的loss输出时一个标量，需要多输出进行修改，才能实现并行。具体修改方式如下：
    yolo_loss函数中的
    xy_loss = K.sum(xy_loss) / mf
    wh_loss = K.sum(wh_loss) / mf
    confidence_loss = K.sum(confidence_loss) / mf
    class_loss = K.sum(class_loss) / mf
    loss += xy_loss + wh_loss + confidence_loss +class_loss
    替换成下面的代码
    xy_loss = K.sum(K.sum(xy_loss,axis=[2,3,4]),1,keepdims=True)
    wh_loss = K.sum(K.sum(wh_loss,axis=[2,3,4]),1,keepdims=True)
    confidence_loss = K.sum(K.sum(confidence_loss,axis=[2,3,4]),1,keepdims=True)
    class_loss = K.sum(K.sum(class_loss,axis=[2,3,4]),1,keepdims=True)
    loss += xy_loss + wh_loss + confidence_loss +class_loss
    即实现对loss的改进，保证输入N张图片，输出时（N，1）对应的时N张图片的loss
    这个函数在yolo3文件夹model.py里。

    然后将编译语句model.compile(optimizer=Adam(lr=1e-3), loss={ ‘yolo_loss’: lambda y_true, y_pred: y_pred})改为
    model.compile(optimizer=Adam(lr=1e-3), loss=totalloss),其中totalloss为下面定义的损失函数，下面这个定义要添加进train.py里。
    def totalloss(y_true, y_pred):
    return K.sum(y_pred)/K.cast(K.shape(y_pred)[0],K.dtype(y_pred))
    在train.py里添加和修改，记住两个if True的model.compile都要改loss。

    再将model_loss = Lambda(yolo_loss, output_shape=(1,),name=‘yolo_loss’,arguments={‘anchors’: anchors, ‘num_classes’: num_classes, ‘ignore_thresh’: 0.5})( [*model_body.output, *y_true])中的, output_shape=(1,)去掉，
    直接替换为model_loss = Lambda(yolo_loss,name=‘yolo_loss’,arguments={‘anchors’: anchors, ‘num_classes’: num_classes, ‘ignore_thresh’: 0.5})( [*model_body.output, *y_true])
    model_loss在train.py的def create_model函数定义里。
    """
    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if False:
        multi_model.compile(optimizer=Adam(lr=1e-3), loss=multi_loss)
            # use custom yolo_loss Lambda layer.
            #{'yolo_loss': lambda y_true, y_pred: y_pred})

        batch_size = 32
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        multi_model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                validation_steps=max(1, num_val//batch_size),
                epochs=50,
                initial_epoch=0,
                callbacks=[logging, checkpoint])
        model.save_weights(log_dir + '/trained_weights_stage_1.h5')

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        for i in range(len(multi_model.layers)):
            multi_model.layers[i].trainable = True
        #model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
        multi_model.compile(optimizer=Adam(lr=1e-4), loss=multi_loss)
        print('Unfreeze all of the layers.')

        #batch_size = 32 # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        multi_model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val//batch_size),
            epochs=epoch+100, #100
            initial_epoch=100, #50
            callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(log_dir + '/trained_weights_final.h5')

    # Further training if needed.


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/yolo_weights.h5'):
    '''create the training model'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, ###########output_shape=(1,), 
        name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/tiny_yolo_weights.h5'):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
        num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers)-2)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)

if __name__ == '__main__':
    _main()