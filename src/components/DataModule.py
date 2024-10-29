import json
import joblib
import numpy as np
import tensorflow as tf
from datasets import load_from_disk

from src.entity.entity import DataModuleConfig

class DataCollator:
    def __init__(self, config:DataModuleConfig):
        self.config=config
        self.train_transform=joblib.load(config.PREPROCESSOR_DIR+ '/train_transform.pkl')
        self.test_transform=joblib.load(config.PREPROCESSOR_DIR+ '/test_transform.pkl')
        self.transform=self.train_transform
        
        self.iou_threshold=0.5
        self.num_scale=len(config.ANCHORS)
        self.num_anchor_per_scale=len(config.ANCHORS[0])
        self.anchors = np.array(config.ANCHORS).reshape(-1, 2)
        self.grid_sizes= [config.IMAGE_SIZE//32, config.IMAGE_SIZE//16, config.IMAGE_SIZE//8]

        with open(config.DATA_DIR+'/label_to_id.json') as file:
            self.label_to_id=json.load(file)

    def mode(self, split):
        if split=='train':
            self.transform=self.train_transform
        else:
            self.transform=self.test_transform

    def __call__(self, batch):
        processed = [self.single_process(x) for x in batch]
        image = [i[0] for i in processed]
        target_1 =[i[1] for i in processed]
        target_2 =[i[2] for i in processed]
        target_3 =[i[3] for i in processed]
        return {'image': image,
                'target_1': target_1,
                'target_2': target_2,
                'target_3': target_3,
                'label_bbox': np.array(len(batch))}

    def single_process(self, sample):
        image=sample['image']
        bboxes=[x['bbox'] for x in sample['label_bbox']]
        labels=[self.label_to_id[x['label']] for x in sample['label_bbox']]

        trasform_input = self.transform(image=image, bboxes=bboxes, labels=labels)
        image, bboxes, labels = trasform_input['image'], trasform_input['bboxes'], trasform_input['labels']
        targets=self.prepare_target(bboxes, labels)
        return image, *(targets)

    def prepare_target(self, bboxes, labels):
        bboxes=[self.coco_to_yolo_format(box) for box in bboxes]
        targets = [np.zeros([self.num_anchor_per_scale, grid_size, grid_size, 6]) for grid_size in self.grid_sizes]

        for box, label in zip(bboxes, labels):
            iou_scores = self.anchor_iou(box[2:4], self.anchors)
            anchor_indices=np.argsort(iou_scores)[::-1]
            x,y,w,h = box

            has_anchor=[False]*self.num_scale
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx//self.num_anchor_per_scale
                anchor_on_scale = anchor_idx%self.num_anchor_per_scale

                grid_size=self.grid_sizes[scale_idx]
                i, j = int(y*grid_size), int(x*grid_size)
                anchor_taken = targets[scale_idx][anchor_on_scale,i,j,0]

                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0]=1

                    x_cell = grid_size*x-j
                    y_cell = grid_size*y-i
                    w_cell = grid_size*w
                    h_cell = grid_size*h
                    box_cell = [x_cell, y_cell, w_cell, h_cell]
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_cell
                    targets[scale_idx][anchor_on_scale, i, j, 5] = label
                    has_anchor[scale_idx] = True
                elif not anchor_taken and iou_scores[anchor_idx]>self.iou_threshold:
                    targets[scale_idx][anchor_on_scale, i, j, 0]=-1

        return targets

    @staticmethod
    def anchor_iou(box1, box2):
        box1=np.array(box1)
        intersection =np.minimum(box1[0], box2[..., 0]) * np.minimum(box1[1], box2[..., 1])
        box1_area = box1[0]*box1[1]
        box2_area = box2[..., 0]*box2[..., 1]

        union=box1_area+box2_area-intersection
        return intersection/union

    def coco_to_yolo_format(self, box):
        x,y,w,h = box

        x_center = x+w/2
        y_center = y+h/2

        x_center=x_center/self.config.IMAGE_SIZE
        y_center=y_center/self.config.IMAGE_SIZE
        width=w/self.config.IMAGE_SIZE
        height=h/self.config.IMAGE_SIZE

        return [x_center, y_center, width, height]

class DataModule:
    def __init__(self, config: DataModuleConfig):
        self.dataset=load_from_disk(config.DATA_DIR)
        with open(config.DATA_DIR+'/label_to_id.json', 'r') as file:
            self.label_to_id=json.load(file)
        with open(config.DATA_DIR+'/id_to_label.json', 'r') as file:
            self.id_to_label=json.load(file)

        self.setup(config.TRAIN_SIZE)
        self.collator=DataCollator(config)

    def setup(self, train_size):
        tmp=self.dataset['train'].train_test_split(train_size=train_size)
        self.dataset['validation']=tmp.pop('test')
        self.dataset['train']=tmp.pop('train')

    def get_dataloader(self, split, batch_size=32, shuffle=False):
        self.collator.mode(split)
        return self.dataset[split].to_tf_dataset(columns=['image', 'label_bbox'], label_cols=['target_1','target_2','target_3'], collate_fn=self.collator,
                                                   batch_size=batch_size, drop_remainder=True, shuffle=shuffle, prefetch=tf.data.AUTOTUNE)