import numpy as np
import tensorflow as tf
from einops import rearrange,repeat

import random
import tensorflow as tf
from PIL import ImageDraw, ImageFont


def pascal_iou(box1, box2):
    x1 = tf.maximum(box1[..., 0], box2[..., 0])
    y1 = tf.maximum(box1[..., 1], box2[..., 1])
    x2 = tf.minimum(box1[..., 2], box2[..., 2])
    y2 = tf.minimum(box1[..., 3], box2[..., 3])

    intersect = tf.maximum(0.0, x2 - x1) * tf.maximum(0.0, y2 - y1)

    box1_area = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    box2_area = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])

    union = box1_area + box2_area - intersect
    iou = intersect / union

    return iou

def yolo_to_pascal(box):
    x1 = box[..., 0]-box[..., 2]/2
    y1 = box[..., 1]-box[..., 3]/2
    x2 = box[..., 0]+box[..., 2]/2
    y2 = box[..., 1]+box[..., 3]/2
    return tf.stack([x1, y1, x2, y2], axis=-1)

def yolo_iou(box1, box2):
    box1 = yolo_to_pascal(box1)
    box2 = yolo_to_pascal(box2)
    return pascal_iou(box1, box2)

def draw_boxes(image, annotations):
    colors = {}
    draw = ImageDraw.Draw(image)
    for annotation in annotations:
        x_min, y_min, width, height = annotation['bbox']
        x_max=x_min+width
        y_max=y_min+height
        box=[x_min, y_min, x_max,y_max]
        label = annotation['label']

        if label not in colors:
            colors[label] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        color = colors[label]
        draw.rectangle(box, outline=color, width=2)

        text_position = (box[0], box[1] - 10)
        font = ImageFont.load_default()

        text_bbox = font.getbbox(label)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        draw.rectangle(
            [text_position, (text_position[0] + text_width, text_position[1] + text_height)],
            fill=color
        )
        draw.text(text_position, label, fill=(255, 255, 255), font=font)

    return image

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

def smth(pred, ):
    anchors = [[0.28, 0.22], [0.38, 0.48], [0.9, 0.78],[0.07, 0.15], [0.15, 0.11], [0.14, 0.29],[0.02, 0.03], [0.04, 0.07], [0.08, 0.06]]
    anchors = tf.cast(anchors, dtype=tf.float32)
    boxes=[]
    for x in pred:
        boxes.extend(get_pred_boxes(x, anchors, grid_size=x.shape[2], obj_threshold=0.5))

    boxes = nms(boxes, iou_threshold=0.5, obj_threshold=0.5, class_threshold=0.5)

    return boxes

