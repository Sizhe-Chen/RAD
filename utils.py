import os
import cv2
import time
import json
import shutil
import argparse
import PIL.Image
import numpy as np
import tensorflow as tf
from copy import deepcopy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras import backend as K

from interpreters import SGLRP

paths = {'Val':   'COCO/val2017',
         'Annot': 'COCO/annotations/instances_val2017.json',
         'font':  'font/FiraMono-Medium.otf',
         'yolo_anchors': 'model_data/yolo_anchors.txt',
         'coco_classes': 'model_data/coco_classes.txt'}
for key in paths: assert os.path.exists(paths[key]), paths[key] + ' does not exist'


def build_direction(loss, input_place, TI=False, transform='1norm'):
    import scipy.stats as st
    kernlen = 15; nsig = 3
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    kernel = kernel.astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
    stack_kernel = np.expand_dims(stack_kernel, 3)
    
    grad = tf.gradients(loss, input_place)[0]
    if TI: grad = tf.nn.depthwise_conv2d(grad, stack_kernel, strides=[1, 1, 1, 1], padding='SAME')
    if   transform == 'sign':  direction = tf.sign(grad)
    elif transform == '1norm': direction = grad * tf.cast(tf.size(grad), tf.float32) / (tf.norm(grad, ord=1) + 1e-20)
    elif transform == 'pnorm': direction = grad / tf.reduce_max(tf.abs(grad) + 1e-20)
    return direction


def random_padding(adv_image):
    # p = 1
    #if np.random.rand() < 0.7: return adv_image
    original_size = (adv_image.shape[1], adv_image.shape[2])
    resized_factor = np.random.uniform(0.9, 1)
    new_size = (int(adv_image.shape[1]*resized_factor), int(adv_image.shape[2]*resized_factor))
    resized_image = cv2.resize(adv_image[0], (new_size[1], new_size[0]))[np.newaxis, ...]
    new_corner = (int(np.random.uniform(0, original_size[0] - new_size[0])), int(np.random.uniform(0, original_size[1] - new_size[1])))
    new_image = np.zeros(adv_image.shape)
    new_image[0, new_corner[0]:new_corner[0]+new_size[0], new_corner[1]:new_corner[1]+new_size[1], :] = resized_image[0]
    return new_image.astype(adv_image.dtype)


def calculate_direction(adv_image, run_direction, DI=False, SI=False): 
    direction_values = [run_direction(adv_image)]
    if SI: 
        for i in range(4):
            direction_values.append(run_direction(adv_image * (0.5 ** i)))
    if DI:
        for i in range(4):
            direction_values.append(run_direction(random_padding(adv_image)))
    return sum(direction_values) / len(direction_values)


def initialize_uninitialized(sess):
    global_vars          = tf.global_variables()
    is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
    if len(not_initialized_vars): sess.run(tf.variables_initializer(not_initialized_vars))


def convert_second_to_time(sec):
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return "%02d:%02d:%02d" % (h, m, s)


def get_time(middle='-'): return time.strftime('%Y-%m-%d' + middle + '%H-%M-%S', time.localtime(time.time()))


def output(value_dict, stream=None, bit=3):
    output_str = ''
    for key, value in value_dict.items():
        if isinstance(value, list):
            for i in range(len(value)): value[i] = round(value[i], bit)
        if isinstance(value, float) or isinstance(value, np.float32) or isinstance(value, np.float64): value = round(value, bit)
        output_str += '[ ' + str(key) + ' ' + str(value) + ' ] '
    print(output_str)
    if stream is not None: print(output_str, file=stream)


def copy_files(result_dir, forms=['.py'], eliminated=['-', '__pycache__']):
    for root, _, files in os.walk('.'):
        do_continue = False
        for item in eliminated:
            if item in root: do_continue = True
        if do_continue: continue
        for file in files:
            do_copy = False
            for item in forms:
                if item in file: do_copy = True
            if not do_copy: continue
            destiny_path = result_dir + root[1:]
            os.makedirs(destiny_path, exist_ok=True)
            shutil.copyfile(root + '/' + file, destiny_path + '/' + file)


def save_images(images, result_dir, name):
    if len(images) == 0: return
    assert images[0].dtype == np.uint8, 'images must be uint8'
    number = len(images)
    n_img_x = int(np.sqrt(number)) if int(np.sqrt(number)**2) == number else int(np.sqrt(number)) + 1
    plot = Plot(result_dir, n_img_x=n_img_x, img_w=images[0].shape[1], img_h=images[0].shape[0], img_c=images[0].shape[2])
    plot.add_image(images)
    plot.save_images(name)
    plot.clear()


def heatmap(heatmap, cmap="seismic", interpolation="none", colorbar=False, M=None):
    if M is None:
        M = np.abs(heatmap).max()
        if M == 0: M = 1
    plt.imshow(heatmap, cmap=cmap, vmax=M, vmin=-M, interpolation=interpolation)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    if colorbar: plt.colorbar()


def visualize_lrp(analysis, size=440, get_signature=False):
    if get_signature: return cv2.resize(analysis / np.max(np.abs(analysis)) * 255, (size, size))
    heatmap(analysis.sum(axis=2))
    buffer_path = str(time.time() + np.random.choice(range(1000))).replace('.', '') + '.png'
    plt.savefig(buffer_path)
    img = PIL.Image.open(buffer_path).convert('RGB')
    os.remove(buffer_path)
    plt.clf()
    return cv2.resize(np.array(img)[18:458, 100:540, ...], (size, size))


def build_lrp(partial_model, out=None, out_ori=None):
    inp = partial_model.input
    multi = False
    if out is None: 
        assert out_ori is None
        out = partial_model.output
    if out_ori is not None:
        class_num = int(out.shape[1])
        indics_ori = tf.argmax(out_ori[0])
        out_modified = tf.where(tf.equal(tf.range(class_num), tf.cast(indics_ori, tf.int32)), tf.zeros(class_num), out[0])
        target_id = tf.argmax(out_modified)
    else:
        # judge multi target_id
        try: 
            int(out.shape[0])
            target_id = tf.argmax(out[0])
        except TypeError:
            target_id = out # target_id -> (None,) shape, tf.int32 placeholder is specified, which contains interesting indices for heatmap
            multi = True
    analyzer = SGLRP(partial_model, target_id=target_id, multi=multi, relu=True, low=tf.reduce_min(inp), high=tf.reduce_max(inp), allow_lambda_layers=True)
    analysis = analyzer.analyze(inp)
    return analysis[0]


class Plot:
    def __init__(self, directory, n_img_x, img_w, img_h, img_c=3, interval=1):
        self.directory = directory
        if not os.path.exists(directory): os.makedirs(directory)
        assert isinstance(interval, int)
        assert interval >= 1
        self.interval = interval
        assert n_img_x > 0
        self.n_img_x = n_img_x
        assert img_w > 0 and img_h > 0
        self.img_w = img_w
        self.img_h = img_h
        assert img_c == 1 or img_c == 3
        self.img_c = img_c
        self.img_list = []

    def save_images(self, name):
        PIL.Image.fromarray(self._merge(self.img_list[::self.interval]).astype(np.uint8)).save(os.path.join(self.directory, name))

    def _merge(self, image_list):
        size_y = len(image_list) // self.n_img_x + (1 if len(image_list) % self.n_img_x != 0 else 0)
        size_x = self.n_img_x
        h_ = int(self.img_h)
        w_ = int(self.img_w)
        img = np.zeros((h_ * size_y, w_ * size_x, self.img_c))
        
        for idx, image in enumerate(image_list):
            i = int(idx % size_x)
            j = int(idx / size_x)
            img[j*h_:j*h_+h_, i*w_:i*w_+w_, :] = image.reshape((self.img_h, self.img_w, self.img_c))
        return img.squeeze()

    def add_image(self, img):
        if isinstance(img, list): self.img_list += img
        else: self.img_list.append(img)
        return self

    def clear(self):
        self.img_list = []