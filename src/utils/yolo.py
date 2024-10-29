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