import tensorflow as tf
from src.utils.yolo import *
from src.entity.entity import LossConfig

class YOLOLoss(tf.keras.losses.Loss):
    def __init__(self, config:LossConfig):
        super().__init__()
        self.lambda_obj=config.LAMBDA_OBJ
        self.lambda_noobj=config.LAMBDA_NOOBJ
        self.lambda_coord=config.LAMBDA_COORD
        self.lambda_class=config.LAMBDA_CLASS
        self.anchors=tf.convert_to_tensor(config.ANCHORS)

        self.mse=tf.keras.losses.MeanSquaredError()
        self.bce=tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.ce=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def call(self, target, pred):
        loss=0.0
        for idx, val in enumerate(target.values()):
            loss=loss+self.single_loss(val, pred[idx], self.anchors[idx])

        return loss

    def single_loss(self, target, pred, anchors):
        obj = target[..., 0]==1
        noobj = target[..., 0]==0

        # no object loss
        noobj_loss= self.bce(target[..., 0][noobj], pred[..., 0][noobj])

        # obj loss
        anchors = tf.reshape(anchors,[1,3,1,1,2])
        pred_boxes = tf.concat([tf.math.sigmoid(pred[...,1:3]), tf.math.exp(pred[..., 3:5])*anchors], axis=-1)
        iou_score = yolo_iou(pred_boxes[obj], target[..., 1:5][obj])
        obj_loss=self.bce(iou_score*target[..., 0][obj], pred[..., 0][obj])

        # box loss
        pred_boxes = tf.concat([tf.math.sigmoid(pred[..., 1:3]), pred[..., 3:5]], axis=-1)
        true_boxes = tf.concat([pred[..., 1:3], tf.math.log(1e-6+ target[..., 3:5]/anchors)], axis=-1)
        coord_loss = self.mse(true_boxes[obj], pred_boxes[obj])

        # class loss
        class_loss = self.ce(target[..., 5:][obj], pred[..., 5:][obj])

        return self.lambda_noobj*noobj_loss + self.lambda_obj*obj_loss + self.lambda_coord*coord_loss + self.lambda_class*class_loss