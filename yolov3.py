
from rad import *
from yolov3_base import YOLO


class YOLOv3(Model):
    def __init__(self, model_attack, model_detect):
        super().__init__(model_attack, model_detect, 'YOLOv3')
    
    def preprocess_image(self, image_path):
        image, self.val_image, self.resized = self.model_detect.preprocess_image(PIL.Image.open(image_path).convert('RGB'))
        return image

    def extract_valid_image(self, image):
        return image[self.val_image].reshape(self.resized)

    def de_preprocess_image(self, image):
        return self.extract_valid_image((image[0] * 255).astype(np.uint8))

    def detect(self, image):
        detection, bbox_number = self.model_detect.detect_image(PIL.Image.fromarray(self.de_preprocess_image(image)), return_box_number=True)
        return detection, bbox_number

    def attack(self, adv_image, alpha, direction_value, ori_image, epsilon):
        adv_image[0][self.val_image] = np.clip(adv_image - alpha / 255 * direction_value, 0, 1)[0][self.val_image]
        adv_image = np.clip(adv_image, ori_image - epsilon / 255, ori_image + epsilon / 255)
        return adv_image


if __name__ == "__main__":
    # weights from https://pjreddie.com/media/files/yolov3.weights 
    # cfg from https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
    # convert to .h5 by https://github.com/qqwweee/keras-yolo3/blob/master/convert.py
    # python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
    assert os.path.exists('model_data/yolo.h5')
    def load_model():
        yolo = YOLO()
        model = YOLOv3(yolo.yolo_model_reshape, yolo)
        return model
    def get_index(pred):
        get_sorted_index = lambda x: np.argsort(x[4::85])[::-1] * 85 + 4
        indexes = [x for x in get_sorted_index(pred)]
        return indexes[:20]
    rad_coco(load_model, get_index, group_dimension=85, attack_dimension=4, transfer_enhance=['SI'])