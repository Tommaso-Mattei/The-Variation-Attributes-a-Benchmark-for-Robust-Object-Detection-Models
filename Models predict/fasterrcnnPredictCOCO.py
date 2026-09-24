import os
import torch
import cv2
import numpy as np
import colorsys
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms.functional import to_tensor

#This code takes a COCO pre-trained Faster R-CNN model and does inference on the benchmark; additionally it also
#replicates a YOLO style annotation by drawing bounding boxes and classes on the images themselves, beyond the .txt labels

# codifies different colors per class
def get_class_colors(num_classes):
    colors = []
    for i in range(num_classes):
        hue = i / num_classes
        r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9) 
        colors.append((int(b * 255), int(g * 255), int(r * 255)))  
    return colors



source_path = "EntireSet" 

images_dir = "runs_fasterrcnn_COCO/images"

os.makedirs(images_dir, exist_ok=True)


weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
model = fasterrcnn_resnet50_fpn(weights=weights)
model.eval().to('cuda')

categories = weights.meta["categories"]

class_colors = get_class_colors(len(categories))


image_paths = sorted(
    os.path.join(source_path, f)
    for f in os.listdir(source_path)
    if f.lower().endswith(".jpg")
)

labels_dir = "runs_fasterrcnn_COCO/labels"
os.makedirs(labels_dir, exist_ok=True)


with torch.no_grad():
    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        img_tensor = to_tensor(img).to('cuda')
 
        prediction = model([img_tensor])[0]  # dictionary: 'boxes' = x1,y1,x2,y2, 'labels', 'scores'


        keep = prediction['scores'] >= 0.25  # confidence cutoff
        boxes = prediction['boxes'][keep].tolist()
        labels = prediction['labels'][keep].tolist()
        scores = prediction['scores'][keep].tolist()
 

        filename = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        txt_path = os.path.join(labels_dir, filename)

        #Writes info in a txt
        with open(txt_path, "w") as f:
            for label, box, score in zip(labels, boxes, scores):
                x1, y1, x2, y2 = box
                f.write(f"{label} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {score:.4f}\n")

        img_np = np.array(img)  
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) 

        h, w = img_bgr.shape[:2]
        font_scale = max(0.5, w / 1500)  
        thickness = max(1, w // 500)


        #loop responsable for drawing bounding boxes and classes on the actual images
        for label, box, score in zip(labels, boxes, scores):
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            class_name = categories[label]
            text = f"{class_name} {score:.2f}"
            color = class_colors[label]

            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thickness)

            text_y = y1 - 10 if y1 - 10 > 15 else y1 + 20

            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(img_bgr, (x1, text_y - text_h - 4), (x1 + text_w, text_y + 4), color, -1)
            cv2.putText(img_bgr, text, (x1, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        out_img_path = os.path.join(images_dir, os.path.basename(img_path))
        cv2.imwrite(out_img_path, img_bgr)
 
        print(os.path.basename(img_path), "->", len(boxes), "detections")