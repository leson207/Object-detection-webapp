import numpy as np
import tensorflow as tf
from einops import rearrange,repeat

from utils.yolo import yolo_iou

def get_pred_boxes(pred, anchors, grid_size, obj_threshold):
    obj_pred=pred[..., 0:1]
    center_pred=pred[..., 1:3]
    size_pred=pred[..., 3:5]*tf.reshape(anchors, [1, len(anchors), 1, 1, 2])
    best_prob = tf.math.reduce_max(tf.nn.softmax(pred[..., 5:], axis=-1), axis=-1, keepdims=True)
    best_class = tf.cast(tf.expand_dims(tf.math.argmax(pred[..., 5:], axis=-1), -1), dtype=tf.float32)

    cell_indices=repeat(tf.range(grid_size, dtype=tf.float32), 'h -> 1 a h w 1', a=len(anchors), w=grid_size)
    x=1/grid_size*(center_pred[..., 0:1]+cell_indices)
    y=1/grid_size*(center_pred[..., 1:2]+rearrange(cell_indices, 'b a h w 1 -> b a w h 1'))
    wh=1/grid_size*size_pred

    pred_boxes = tf.concat([obj_pred, x, y, wh, best_class, best_prob], axis=-1)
    filtered_boxes = []
    for batch in pred_boxes:
        filtered = tf.boolean_mask(batch, batch[..., 0] >= obj_threshold)
        filtered_boxes.append(filtered[..., 1:])
    return filtered_boxes

def get_true_boxes(target, anchors, grid_size):
    # boxes = batch_size, anchor, s ,s 6
    # box = [obj, x ,y, w ,h, class_idx]
    center = target[..., 1:3]
    size = target[..., 3:5]
    label_idx= target[..., 5:6]

    cell_indices=repeat(tf.range(grid_size, dtype=tf.float32), 'h -> 1 a h w 1', a=len(anchors), w=grid_size)
    x=1/grid_size*(center[..., 0:1]+cell_indices)
    y=1/grid_size*(center[..., 1:2]+rearrange(cell_indices, 'b a h w 1 -> b a w h 1'))
    wh=1/grid_size*size
    true_boxes=tf.concat([x,y,wh, label_idx], axis=-1)
    filtered_boxes = []
    for batch in true_boxes:
        filtered = tf.boolean_mask(batch, batch[..., 0] ==1)
        filtered_boxes.append(filtered[..., 1:])
    return filtered_boxes

def nms(boxes, iou_threshold=0.5, obj_threshold=0.5, class_threshold=0.5):
    # boxes = num_box_of anser 6
    # box = [x,y,w,h,class_idx, class_prob]
    boxes = [box for box in boxes if box[5]>class_threshold and box[0]>obj_threshold]
    boxes = sorted(boxes, key=lambda x: x[5], reverse=True)

    selected_boxes=[]
    while boxes:
        selected_box = boxes.pop(0)
        selected_boxes.append(selected_box)
        for idx in range(len(boxes)-1, -1,-1):
            if boxes[idx][4]==selected_box[4] and yolo_iou(selected_box[0:4], boxes[idx][0:4])<iou_threshold:
                boxes.pop(idx)

    return selected_boxes

def mean_average_precision(pred_boxes, true_boxes, iou_thresholds=[0.5], prob_thresholds=[0.5], num_classes=20):
    # box = [[x1,y1,x2,y2,calss_idx,prob]]

    TP=np.zeros([num_classes, len(iou_thresholds), len(prob_thresholds)])
    FP=np.zeros([num_classes, len(iou_thresholds), len(prob_thresholds)])
    num_true_boxes = np.zeros([num_classes])
    for single_pred_boxes, single_true_boxes in zip(pred_boxes, true_boxes):
        single_pred_boxes=sorted(single_pred_boxes, key = lambda x: x[5], reverse=True)
        for true_box in single_true_boxes:
            num_true_boxes[int(true_box[4])] += 1

        matched = [False]*len(single_true_boxes)
        for pred_box in single_pred_boxes:
            best_score=0
            best_idx=0
            for idx, true_box in enumerate(single_true_boxes):
                iou_score = yolo_iou(pred_box[0:4], true_box[0:4])
                if iou_score>best_score and pred_box[4]==true_box[4]:
                    best_score=iou_score
                    best_idx=idx

            for iou_idx, iou_threshold in enumerate(iou_thresholds):
                for prob_idx, prob_threshold in enumerate(prob_thresholds):
                    if best_score>iou_threshold and pred_box[5]>=prob_threshold:
                        if matched[best_idx]==False:
                            TP[pred_box[4]][iou_idx][prob_idx]+=1
                            matched[best_idx]=True
                        else:
                            FP[pred_box[4]][iou_idx][prob_idx]+=1

    precision = np.divide(TP, TP + FP, out=np.zeros_like(TP), where=(TP + FP) != 0)
    precision = np.mean(precision, axis=0)
    recall = np.divide(TP, num_true_boxes[:, None, None], out=np.zeros_like(TP), where=num_true_boxes[:, None, None] != 0)
    recall = np.mean(recall, axis=0)

    metrics={}
    for iou_idx, iou_threshold in enumerate(iou_thresholds):
        for prob_idx, prob_threshold in enumerate(prob_thresholds):
            key=f'iou@{iou_threshold}_prob@{prob_threshold}'
            metrics['Precision_'+key]=precision[iou_idx][prob_idx]
            metrics['Recall_'+key]=recall[iou_idx][prob_idx]

    return metrics

def get_metrics(pred, target, anchors, grid_size):
    true_boxes=get_true_boxes(target, anchors, grid_size)
    pred_boxes=get_pred_boxes(pred, anchors, grid_size, obj_threshold=0.05)
    pred_boxes=[nms(batch, iou_threshold=0.05, obj_threshold=0.05, class_threshold=0.05) for batch in pred_boxes]
    metrics=mean_average_precision(pred_boxes, true_boxes, iou_thresholds=[0., 0.075], prob_thresholds=[0., 0.175], num_classes=208)
    return metrics