from rad import *
from retinanet_base import load_net, read_image, detect_image


class RetinaNet(Model):
    def __init__(self, model_attack, model_detect):
        super().__init__(model_attack, model_detect, 'RetinaNet')
        self.low = - np.array([103.939, 116.779, 123.68])
        self.high = 255 + self.low
    
    def preprocess_image(self, image_path):
        image_resized, ori_image, self.val_image = read_image(image_path, 416)
        self.resized = image_resized.shape
        return ori_image

    def extract_valid_image(self, image):
        return image[self.val_image].reshape(self.resized)

    def de_preprocess_image(self, image):
        img = deepcopy(image-self.low)
        return self.extract_valid_image(img[0, :, :, ::-1].astype(np.uint8))

    def detect(self, image):
        detection, bbox_number = detect_image(self.model_detect, image, self.de_preprocess_image(image), self.val_image)
        return detection, bbox_number

    def attack(self, adv_image, alpha, direction_value, ori_image, epsilon):
        adv_image[0][self.val_image] = np.clip(adv_image - alpha * direction_value, self.low, self.high)[0][self.val_image]
        adv_image = np.clip(adv_image, ori_image - epsilon, ori_image + epsilon)
        return adv_image


if __name__ == "__main__":
    # downloaded from https://github.com/fizyr/keras-retinanet/releases/download/0.5.1/resnet50_coco_best_v2.1.0.h5
    model_path = 'model_data/resnet50_coco_best_v2.1.0.h5'
    assert os.path.exists(model_path)
    def load_model():
        model_attack, model_detect = load_net(model_path)
        model = RetinaNet(model_attack, model_detect)
        return model
    def get_index(pred):
        pred_reshape = pred.reshape(-1, 80)
        pred_label = np.argmax(pred_reshape, axis=1)
        pred_score = np.max(pred_reshape, axis=1)
        indexes = np.argsort(pred_score)[::-1] * 80 + pred_label
        return indexes[:20]
    rad_coco(load_model, get_index, group_dimension=80, attack_dimension=0, transfer_enhance=['SI'])