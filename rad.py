from utils import *


# The encapulation of model to be attacked
class Model():
    def __init__(self, model_attack, model_detect, name):
        self.model_attack = model_attack
        self.model_detect = model_detect
        self.name = name
    def preprocess_image(self, image_path): raise NotImplementedError
    def extract_valid_image(self, image): raise NotImplementedError
    def de_preprocess_image(self, image): raise NotImplementedError
    def detect(self, image): raise NotImplementedError
    def attack(self, adv_image, alpha, direction_value, ori_image, epsilon): raise NotImplementedError


# Cyclic data generator
class DataGenerator():
    def __init__(self, data_directory, start_id, end_id, sort_key=None):
        self.data_directory = data_directory
        self.start_id = start_id
        self.end_id = end_id
        self.id = 0
        self.iter = 0
        self.length = self.end_id - self.start_id
        self.file_list = os.listdir(self.data_directory)
        self.file_list.sort(key=sort_key)
        self.file_list = self.file_list[self.start_id:self.end_id]
    

    def yield_sample_path(self):
        sample_path = self.file_list[self.id]
        self.id = (self.id + 1) % self.length
        if self.id == 0: self.iter += 1
        return sample_path


# RAD builder and attacker
class RAD():
    def __init__(self,
                 data_generator,
                 model,
                 get_index,
                 group_dimension, 
                 attack_dimension, 
                 num_class, 
                 transfer_enhance,
                 ):
        self.data_generator = data_generator
        self.model = model
        self.get_index = get_index
        self.group_dimension = group_dimension
        self.attack_dimension = attack_dimension
        self.num_class = num_class
        self.transfer_enhance = transfer_enhance
        
        # build the graph for RAD
        if self.group_dimension is None or self.attack_dimension is None:
            # In this case, the graph for attack is and must be built in self.model
            try:
                self.index_place = self.model.index_place
                self.analysis = self.model.analysis
                self.iou_sorted_index = self.model.iou_sorted_index
                self.iou_sorted_index_place = self.model.iou_sorted_index_place
                self.direction = self.model.direction
            except AttributeError:
                print('Please specify group_dimension and attack_dimension for an auto build')
                print('Or specify iou_sorted_index, iou_sorted_index_place and direction in model')
                exit()
        else:
            # build the attack graph automatically
            self.index_place = tf.placeholder(tf.int32, [None, ])
            self.analysis = build_lrp(self.model.model_attack, self.index_place)
            out_reshaped = tf.reshape(self.model.model_attack.output, (-1, self.group_dimension))
            self.iou_sorted_index = tf.argsort(out_reshaped[:, self.attack_dimension], direction='DESCENDING')
            self.iou_sorted_index_place = tf.placeholder(tf.int32, self.iou_sorted_index.shape) # feed with the iou_sorted_index value of original image
            iou_sorted_bbox = tf.gather(out_reshaped, self.iou_sorted_index_place)
            attacked_nodes = tf.reduce_max(iou_sorted_bbox[:3000, self.group_dimension-self.num_class:], axis=1)
            loss = tf.reduce_mean(attacked_nodes) + tf.reduce_mean(self.analysis) / 1e-21 # use RAD loss except that it reaches its minimum
            self.direction = build_direction(loss, self.model.model_attack.input, TI=('TI' in self.transfer_enhance), transform='1norm')
    

    def attack(self, alpha, epsilon, num_iteration):
        # prepare the result dir
        result_dir_base = get_time(middle='-') + \
            '_RAD_%s_%s_Index%dto%d_Eps%d_Iter%d' % \
            (''.join(self.transfer_enhance), 
             self.model.name, 
             self.data_generator.start_id, 
             self.data_generator.end_id, 
             epsilon, 
             num_iteration)
        copy_files(result_dir_base + '/src')
        result_dir_adv = result_dir_base + '/adv-resized'
        result_dir_detection = result_dir_base + '/detection/' + self.model.name.lower()
        os.makedirs(result_dir_adv)
        os.makedirs(result_dir_detection)
        print('\n\n' + result_dir_base)

        # prepare variables
        sess = K.get_session()
        start_iter = self.data_generator.iter
        start = time.time()
        bbox_first, bbox_last, bbox0_record, rmse_record = [], [], [], []
        
        # attack in each sample
        while self.data_generator.iter == start_iter:
            file = self.data_generator.yield_sample_path()
            result_dir = result_dir_base + '/detail/' + os.path.splitext(file)[0]
            os.makedirs(result_dir, exist_ok=True)
            log = open(result_dir + '/log.txt', 'w')
            ori_image = self.model.preprocess_image(self.data_generator.data_directory + '/' + file)
            adv_image = deepcopy(ori_image)
            iou_sorted_index_value = sess.run(self.iou_sorted_index, {self.model.model_attack.input: ori_image})
            ori_detection, ori_bbox_number = self.model.detect(adv_image)
            heatmaps, imgs = [], []
            print()
            
            # attack in each iteration
            for step in range(num_iteration+1):
                # output
                adv_detection, adv_bbox_number = self.model.detect(adv_image)
                rmse = np.sqrt(np.mean(np.square(
                    self.model.de_preprocess_image(adv_image).astype(np.float32) -
                    self.model.de_preprocess_image(ori_image).astype(np.float32) + 1e-12)))
                output({'Iter': '%d/%d' % (step, num_iteration), 
                        'BBox': adv_bbox_number, 
                        'rmse': rmse,
                        }, stream=log)

                # append visualization
                imgs.append(np.array(adv_detection))
                feed_dict ={self.model.model_attack.input: adv_image, 
                            self.index_place:              self.get_index(self.model.model_attack.predict(adv_image)[0]), 
                            self.iou_sorted_index_place:   iou_sorted_index_value}
                analysis_value = sess.run(self.analysis, feed_dict)
                heatmaps.append(self.model.extract_valid_image(visualize_lrp(analysis_value, size=ori_image.shape[1])))
                if step == num_iteration: break # visualize the final result with no further update

                # update sample
                def run_direction(image):
                    feed_dict[self.model.model_attack.input] = image
                    return sess.run(self.direction, feed_dict)
                direction_value = calculate_direction(adv_image, run_direction, DI=('DI' in self.transfer_enhance), SI=('SI' in self.transfer_enhance))
                adv_image = self.model.attack(adv_image, alpha, direction_value, ori_image, epsilon)

            # final operation
            bbox_first.append(ori_bbox_number)
            bbox_last.append(adv_bbox_number)
            bbox0_record.append(adv_bbox_number == 0)
            rmse_record.append(rmse)

            # output result for this sample
            output({'TimeRm': convert_second_to_time((time.time()-start) / len(bbox_first) * (self.data_generator.length-len(bbox_first))),
                    'No': '%d/%d' % (self.data_generator.id + self.data_generator.start_id, self.data_generator.end_id),
                    'File': os.path.splitext(file)[0],
                    }, stream=log)
            
            # output runing average result
            output({'BBoxAVG': '%.2f->%.2f' % (sum(bbox_first) / len(bbox_first), sum(bbox_last) / len(bbox_last)),
                    'BBbox0AVG': sum(bbox0_record) / len(bbox0_record) * 100, 
                    'rmseAVG': sum(rmse_record) / len(rmse_record),
                    }, stream=log)
            
            # save record
            ori_detection.save(result_dir + '/detection_ori_%d.png' % ori_bbox_number)
            adv_detection.save(result_dir + '/detection_adv_%d.png' % adv_bbox_number)
            adv_detection.save(result_dir_detection + '/' + os.path.splitext(file)[0] + '.png')
            PIL.Image.fromarray(self.model.de_preprocess_image(ori_image)).save(result_dir + '/sample_ori.png')
            adv_image = PIL.Image.fromarray(self.model.de_preprocess_image(adv_image))
            adv_image.save(result_dir + '/sample_adv.png')
            adv_image.save(result_dir_adv + '/' +  os.path.splitext(file)[0] + '.png')
            save_images(imgs, result_dir, 'detection.jpg')
            save_images(heatmaps, result_dir, 'heatmap.jpg')
            log.close()


def rad_coco(load_model, get_index, **kwargs):
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('--alpha',            default=2,   type=int, help='attack step length')
    parser.add_argument('--epsilon',          default=16,  type=int, help='l_infty epsilon')
    parser.add_argument('--num_iteration',    default=10,  type=int, help='number of attack iterations')
    parser.add_argument('--start_id',         default=0,   type=int, help='Starting sample ID')
    parser.add_argument('--end_id',           default=5000,type=int, help='Ending sample ID')
    parser.add_argument('gpu_id', help='GPU(s) used')
    args, _ = parser.parse_known_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    
    rad = RAD(
                data_generator=DataGenerator(data_directory=paths['Val'], start_id=args.start_id, end_id=args.end_id, sort_key=lambda x: int(os.path.splitext(x)[0])), 
                model=load_model(), 
                get_index=get_index,
                num_class=80,
                **kwargs,
                )
    rad.attack(alpha=args.alpha, epsilon=args.epsilon, num_iteration=args.num_iteration)