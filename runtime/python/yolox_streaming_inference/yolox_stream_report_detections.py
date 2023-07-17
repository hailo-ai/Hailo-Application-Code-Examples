import numpy as np
import cv2

def get_label(class_id):
    with open('yolox_s_labels.json','r') as f:
        labels = eval(f.read())         
        return labels[str(class_id)]

def draw_detection(image, d, c, color, scale_factor_x, scale_factor_y):
    """Draw box and label for 1 detection."""
    label = get_label(c)    
    ymin, xmin, ymax, xmax = d
    ymin, xmin, ymax, xmax = int(ymin * scale_factor_y), int(xmin * scale_factor_x), int(ymax * scale_factor_y), int(xmax * scale_factor_x)    
    image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness=2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    image = cv2.putText(image, label, (xmin + 2, ymin - 2), font, font_scale, (255, 255, 255), 1)
    return (label, image)

def report_detections(detections, image, min_score=0.45, scale_factor_x=1, scale_factor_y=1):
    """Reports, saves and draws all confident detections"""
    np.random.seed(2) # this seed yeilds nice bb colors
    COLORS = np.random.randint(0, 255, size=(91, 3), dtype=np.uint8)
    boxes = detections['boxes']
    classes = detections['classes'].astype(int)
    scores = detections['scores']
    draw = image.copy()        
    for idx in range(detections['num_detections']):
        if scores[idx] >= min_score:
            color = tuple(int(c) for c in COLORS[classes[idx]])
            label, draw = draw_detection(draw, boxes[idx], classes[idx], color, scale_factor_x, scale_factor_y)
    return draw