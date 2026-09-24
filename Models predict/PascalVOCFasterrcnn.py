import os
import glob
from gluoncv import model_zoo, data
from PIL import Image

#This code takes from GluonCV the Pascal VOC pre-trained model of Faster R-CNN and runs it on the benchmark

IMAGE_DIR = "EntireSet"
OUTPUT_DIR = "runs_fasterrcnn_PascalVOC/labels"
CONF_THRESH = 0.25   

 
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
net = model_zoo.get_model('faster_rcnn_resnet50_v1b_voc', pretrained=True)
print("Classes:", net.classes)
 
image_paths = sorted(
    glob.glob(os.path.join(IMAGE_DIR, "*.jpg")) +
    glob.glob(os.path.join(IMAGE_DIR, "*.jpeg")) +
    glob.glob(os.path.join(IMAGE_DIR, "*.png"))
)
print(f"Found {len(image_paths)} images in {IMAGE_DIR}")

#In the first part the actual inference is done, later to match with the rest of the experiments 
#a rescaling is applied, before writing the result on a .txt
for i, img_path in enumerate(image_paths):
    x, orig_img = data.transforms.presets.rcnn.load_test(img_path)
 
    box_ids, scores, bboxes = net(x)
 
    box_ids = box_ids[0].asnumpy()
    scores = scores[0].asnumpy()
    bboxes = bboxes[0].asnumpy()

    resized_h, resized_w = orig_img.shape[0], orig_img.shape[1]
    with Image.open(img_path) as im:
        raw_w, raw_h = im.size  
 
    scale_x = raw_w / resized_w
    scale_y = raw_h / resized_h
 
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(OUTPUT_DIR, base_name + ".txt")
 
    n_written = 0
    with open(out_path, "w") as f:
        for cls_id, score, box in zip(box_ids, scores, bboxes):
            s = float(score[0])
            if s < 0 or s < CONF_THRESH:
                continue  
            x1, y1, x2, y2 = box
            x1 *= scale_x
            x2 *= scale_x
            y1 *= scale_y
            y2 *= scale_y

            #box clipping
            x1 = max(0.0, min(x1, raw_w))
            x2 = max(0.0, min(x2, raw_w))
            y1 = max(0.0, min(y1, raw_h))
            y2 = max(0.0, min(y2, raw_h))
            f.write(f"{int(cls_id[0])} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {s:.4f}\n")
            n_written += 1
 
    print(f"[{i+1}/{len(image_paths)}] {base_name}: {n_written} detections -> {out_path}")
 
print("Done.")