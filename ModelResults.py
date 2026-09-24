import cv2
import torch
from torchvision.ops import box_iou
import os


IMAGES_DIR = "EntireSet" #Here should go the unlabeled samples
GROUND_TRUTH_DIR = "LabeledTestSetForYOLO11\\test\\labels"  #In here should go the downloaded dataset from roboflow
RESULTS_DIR = "results"
IOU_THRESHOLD = 0.5
IOU_SWEEP_THRESHOLDS = [round(0.5 + 0.05 * i, 2) for i in range(10)]
CONF_SWEEP_THRESHOLDS = [round(0.25 + 0.05 * i, 2) for i in range(15)]

CLASS_NAMES = ['bicycle', 'bottle', 'chair', 'monitor', 'pottedplant']

# Mapping classes between models and ground truth
CLASS_MAPS = {
    "YOLO11": {
        1: 0,   # bicycle
        39: 1,  # bottle
        56: 2,  # chair
        62: 3,  # tv/monitor
        58: 4,  # potted plant
    },
    "FasterRCNN": {
        2: 0,   # bicycle
        44: 1,  # bottle
        62: 2,  # chair
        72: 3,  # tv/monitor
        64: 4,  # potted plant
    },
    "YOLO11VOC": {
        1: 0,   # bicycle
        4: 1,   # bottle
        8: 2,   # chair
        19: 3,  # tv/monitor
        15: 4,  # potted plant
    }
}
CLASS_MAPS["RT-DETR"] = CLASS_MAPS["YOLO11"] #Same base format
CLASS_MAPS["RT-DETRVOC"] = CLASS_MAPS["FasterRCNNVOC"] = CLASS_MAPS["YOLO11VOC"] #Same base format

MODEL_CONFIGS = {
    "YOLO11": {
        "predictions_dir": "runs_YOLO11_COCO\\detect\\predict\\labels",
        "class_map": CLASS_MAPS["YOLO11"],
        "results_file": os.path.join(RESULTS_DIR, "results_YOLO11.txt"),
        "pred_format": "xywh_norm",
    },
    "FasterRCNN": {
        "predictions_dir": "runs_fasterrcnn_COCO\\detect\\predict\\labels",
        "class_map": CLASS_MAPS["FasterRCNN"],
        "results_file": os.path.join(RESULTS_DIR, "results_FasterRCNN.txt"),
        "pred_format": "xyxy_pixels",
    },
    "RT-DETR": {
        "predictions_dir": "runs_RTDETR_COCO\\detect\\predict\\labels",
        "class_map": CLASS_MAPS["RT-DETR"],
        "results_file": os.path.join(RESULTS_DIR, "results_RTDETR.txt"),
        "pred_format": "xywh_norm",
    },
    "YOLO11VOC": {
        "predictions_dir": "runs_YOLO11l_PascalVOC\\detect\\predict\\labels",
        "class_map": CLASS_MAPS["YOLO11VOC"],
        "results_file": os.path.join(RESULTS_DIR, "results_YOLO11_PascalVOC.txt"),
        "pred_format": "xywh_norm",
    },
    "FasterRCNNVOC": {
        "predictions_dir": "runs_fasterrcnn_PascalVOC\\labels",
        "class_map": CLASS_MAPS["FasterRCNNVOC"],
        "results_file": os.path.join(RESULTS_DIR, "results_FasterRCNN_PascalVOC.txt"),
        "pred_format": "xyxy_pixels",
    },
    "RT-DETRVOC": {
        "predictions_dir": "runs_RTDETR_PascalVOC\\detect\\predict\\labels",
        "class_map": CLASS_MAPS["RT-DETRVOC"],
        "results_file": os.path.join(RESULTS_DIR, "results_RTDETR_PascalVOC.txt"),
        "pred_format": "xywh_norm",
    }
}
#Decide which model to analyze by substituing model to run with the right name
#"YOLO11"  #"FasterRCNN" #"RT-DETR" #AddVOC at the end for the pascal voc predictions
MODEL_TO_RUN = "FasterRCNNVOC"

PREDICTIONS_DIR = MODEL_CONFIGS[MODEL_TO_RUN]["predictions_dir"]
CLASS_MAP = MODEL_CONFIGS[MODEL_TO_RUN]["class_map"]
RESULTS_FILE = MODEL_CONFIGS[MODEL_TO_RUN]["results_file"]
PRED_FORMAT = MODEL_CONFIGS[MODEL_TO_RUN]["pred_format"]
MODEL_NAME = MODEL_TO_RUN

LIGHT_NAMES = {1: "day", 2: "night"}
VIEW_NAMES = {1: "front", 2: "lateral", 3: "rear", 4: "top"}
DISTANCE_NAMES = {1: "close", 2: "far"}
OCCLUSION_NAMES = {1: "occluded", 2: "not_occluded"}
TRUNCATION_NAMES = {1: "truncated", 2: "not_truncated"}



#filenames taken from the roboflow version
def extract_base_name(filename):
    name = filename[:-4] if filename.endswith(".txt") else filename  
    lower = name.lower()
    idx = lower.find("_jpg.rf.")
    return name[:idx] if idx != -1 else name #splits the filename to leave only the relevant variation numbers

#Reads the coordinates from a txt torchvision shaped file
def get_boxInfo_xyxy_pixels(filePath):
    boxes = []
    if not os.path.exists(filePath):
        return boxes
    with open(filePath, 'r') as file:
        for line in file:
            parts = line.split()
            if len(parts) < 5:
                continue
            class_id = int(float(parts[0]))
            x1, y1, x2, y2 = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
            confidence = float(parts[5]) if len(parts) >= 6 else None
            boxes.append((class_id, [x1, y1, x2, y2], confidence))
    return boxes

#Reads the coordinates from a txt YOLO shaped file
def get_boxInfo_YOLO(filePath):
    boxes = []
    if not os.path.exists(filePath):
        return boxes  
    with open(filePath, 'r') as file:
        for line in file:
            parts = line.split()
            if len(parts) < 5:
                continue
            class_id = int(float(parts[0]))
            confidence = float(parts[5]) if len(parts) >= 6 else None
            coordinates = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
            boxes.append((class_id, [round(v, 6) for v in coordinates],confidence))
    return boxes

#transforms based on the format
def to_pixel_box(BoxInfo, img_H, img_W, box_format):
    if box_format == "xywh_norm":
        return coordinates_converter(img_H, img_W, BoxInfo)  # YOLO logic
    elif box_format == "xyxy_pixels":
        x1, y1, x2, y2 = BoxInfo
        return torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)
    else:
        raise ValueError(f"Unknown box_format: {box_format}")

#parses a filename and stores the relevant info
def parse_attributes(base_name):
    parts = base_name.split("_")
    if len(parts) != 7:
        print(f"WARNING: unexpected filename format for {base_name}, skipping attribute parse")
        return None
    class_num, obj_num, light, view, distance, occlusion, truncation = (int(p) for p in parts)
    return {
        "class_num": class_num,
        "object_num": obj_num,
        "light": LIGHT_NAMES.get(light, light),
        "view": VIEW_NAMES.get(view, view),
        "distance": DISTANCE_NAMES.get(distance, distance),
        "occlusion": OCCLUSION_NAMES.get(occlusion, occlusion),
        "truncation": TRUNCATION_NAMES.get(truncation, truncation),
    }


#Gets the pixel coordinates of the rectangle upper left and lower right corners, necessary for Intersection over Union
def coordinates_converter(img_H,img_W, BoxInfo):
    x,y,w,h = BoxInfo
    x1 = (x - w/2)*img_W
    y1 = (y - h/2)*img_H
    x2 = (x + w/2)*img_W
    y2 = (y + h/2)*img_H
    return torch.tensor([[x1, y1, x2, y2]], dtype = torch.float32)

#Gets the best IoU of all boxes over the ground truth with the correct class
#It also checks for NMS failures with same class good matches, and 
#it also differentiates between IoU and confidence of the right class or the wrong class when the detection is right
def get_matching_iou(image, truth_class,truth_box,predictions, pred_format):
    img_H, img_W = image.shape[:2]
    ground_truth_box = to_pixel_box(truth_box, img_H, img_W, "xywh_norm")
    best_same_class_iou = 0
    best_same_class_confidence = None
    same_class_good_matches = 0

    best_wrong_class_iou = 0
    best_wrong_class_confidence = None
    for predicted_class_og, predicted_box_og, predicted_confidence in predictions:
        predicted_class = CLASS_MAP.get(predicted_class_og) #same notation of class ; if it's none than it's outside of our 5 classes.

        predicted_box = to_pixel_box(predicted_box_og, img_H, img_W, pred_format)
        iou = box_iou(ground_truth_box,predicted_box).item()
        if  predicted_class is not None and predicted_class == truth_class:
            if iou >= IOU_THRESHOLD:
                same_class_good_matches += 1
            if iou > best_same_class_iou:
                best_same_class_iou = iou
                best_same_class_confidence = predicted_confidence
        else:
            if iou > best_wrong_class_iou:
                best_wrong_class_iou = iou
                best_wrong_class_confidence = predicted_confidence
    return {
        "same_class_iou": best_same_class_iou,
        "same_class_confidence": best_same_class_confidence,
        "same_class_good_matches": same_class_good_matches,
        "wrong_class_iou": best_wrong_class_iou,
        "wrong_class_confidence": best_wrong_class_confidence,
    }

#accuracy vs IoU
def accuracy_at_threshold(records, threshold, conditions=None):
    if conditions:
        subset = [r for r in records if all(r["attributes"].get(k) == v for k, v in conditions.items())]
    else:
        subset = records
    total = len(subset)
    if total == 0:
        return 0.0
    correct = sum(1 for r in subset if r["iou"] >= threshold)
    return correct / total

#accuracy vs Confidence
def accuracy_at_confidence(records, conf_threshold, conditions=None):
    if conditions:
        subset = [r for r in records if all(r["attributes"].get(k) == v for k, v in conditions.items())]
    else:
        subset = records
    total = len(subset)
    if total == 0:
        return 0.0
    kept_correct = sum(
        1 for r in subset
        if r["is_correct"] and r["confidence"] is not None and r["confidence"] >= conf_threshold
    )
    return kept_correct / total

def format_iou_sweep(records, conditions=None):
    accs = [accuracy_at_threshold(records, t, conditions) for t in IOU_SWEEP_THRESHOLDS]
    per_threshold = ", ".join(f"acc@{t:.2f}={a:.4f}" for t, a in zip(IOU_SWEEP_THRESHOLDS, accs))
    mean_acc = sum(accs) / len(accs)
    return f"{per_threshold} | mean_acc[0.5:0.95]={mean_acc:.4f}"

def format_confidence_sweep(records, conditions=None):
    return ", ".join(
        f"acc@conf{c:.2f}={accuracy_at_confidence(records, c, conditions):.4f}"
        for c in CONF_SWEEP_THRESHOLDS
    )

#Only the specific slice of conditions is used for the main stats, if specified, otherwise it's applied on the entire dataset
def compute_stats(records, conditions=None):
    if conditions:
        subset = [r for r in records if all(r["attributes"].get(k) == v for k, v in conditions.items())]
    else:
        subset = records

    total = len(subset)
    if total == 0:
        return {
            "total": 0, "correct": 0, "accuracy": 0.0,
            "mean_iou_all": 0.0, "mean_iou_correct": 0.0,
            "mean_iou_wrong": 0.0,"good_localization_wrong_class_count": 0,
            "conf_good_loc_good_class": 0.0,
            "conf_good_loc_bad_class": 0.0,
            "conf_good_loc_any_class": 0.0,
            "avg_boxes_per_image": 0.0,
            "nms_failures": 0, "nms_failure_rate": 0.0,
        }

    correct = sum(1 for r in subset if r["is_correct"])
     
    ious_correct = [r["iou"] for r in subset if r["is_correct"]]
    ious_wrong = [r["wrong_class_iou"] for r in subset if r["wrong_class_iou"] >= IOU_THRESHOLD and not r["is_correct"]] 
    # ----> i thought abou this metric to check how right the model is in its bounding box even when the class is wrong.
    #While you can't tell precisely if it's the good localization with the wrong class and it may sometime falsly flag an occluding object, it's still worth using

    ious_all = ious_correct + ious_wrong

    #3 different confidences are calculated, all when localization is good: correct class, wrong class and overall
    good_class_confs = [r["confidence"] for r in subset
                         if r["iou"] >= IOU_THRESHOLD and r["confidence"] is not None]

    bad_class_confs = [r["wrong_class_confidence"] for r in subset
                        if r["wrong_class_iou"] >= IOU_THRESHOLD and r["wrong_class_confidence"] is not None and not r["is_correct"]]

    any_class_confs = []
    for r in subset:
        if r["iou"] >= r["wrong_class_iou"]:
            best_iou, best_conf = r["iou"], r["confidence"]
        else:
            best_iou, best_conf = r["wrong_class_iou"], r["wrong_class_confidence"]
        if best_iou >= IOU_THRESHOLD and best_conf is not None:
            any_class_confs.append(best_conf)

    nms_failures = sum(1 for r in subset if r.get("nms_failure"))
    nms_failure_rate = nms_failures / total

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total,
        "mean_iou_all": sum(ious_all) / len(ious_all) if ious_all else 0.0,
        "mean_iou_correct": sum(ious_correct) / len(ious_correct) if ious_correct else 0.0,
        "mean_iou_wrong": sum(ious_wrong) / len(ious_wrong) if ious_wrong else 0.0,
        "good_localization_wrong_class_count" : len(ious_wrong),
        "conf_good_loc_good_class": sum(good_class_confs) / len(good_class_confs) if good_class_confs else 0.0,
        "conf_good_loc_bad_class": sum(bad_class_confs) / len(bad_class_confs) if bad_class_confs else 0.0,
        "conf_good_loc_any_class": sum(any_class_confs) / len(any_class_confs) if any_class_confs else 0.0,
        "avg_boxes_per_image": sum(r["pred_box_count"] for r in subset) / total,
        "nms_failures": nms_failures, "nms_failure_rate": nms_failure_rate,
    }

#Formatting
def format_stats_line(label, stats):
    prefix = f"{label}: " if label else ""
    return (f"{prefix}accuracy={stats['accuracy']:.4f}, "
            f"mean_iou_all_true={stats['mean_iou_all']:.4f}, "
            f"mean_iou_correct_only={stats['mean_iou_correct']:.4f}, "
            f"mean_iou_wrong_only={stats['mean_iou_wrong']:.4f}, "
            f"conf_good_loc_good_class={stats['conf_good_loc_good_class']:.4f}, "
            f"conf_good_loc_bad_class={stats['conf_good_loc_bad_class']:.4f}, "
            f"conf_good_loc_any_class={stats['conf_good_loc_any_class']:.4f}, "
            f"avg_boxes_per_image={stats['avg_boxes_per_image']:.4f}, "
            f"nms_failures={stats['nms_failures']}/{stats['total']} (rate={stats['nms_failure_rate']:.4f}), "
            f"correct={stats['correct']}/{stats['total']}, "
            f"good_localization_wrong_class_count={stats['good_localization_wrong_class_count']}")



if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    label_files = [file for file in os.listdir(GROUND_TRUTH_DIR) if file.endswith(".txt")]
    records = []

    #Executes the entire logic, comparing the label with the prediction and appending in the records all the metric
    for label_file in label_files:
        base_name = extract_base_name(label_file)
        truth_path = os.path.join(GROUND_TRUTH_DIR,label_file)
        truth_info = get_boxInfo_YOLO(truth_path)
        if len(truth_info) == 0:
            print(f"WARNING: no ground truth box in {label_file}, skipping") #Should not happen at all but just to be sure.
            continue
        truth_class, truth_box, _ = truth_info[0] #Only one box

        #Matching prediction image
        image_path = os.path.join(IMAGES_DIR, base_name + ".JPG")
        if not os.path.exists(image_path):
            alt_path = os.path.join(IMAGES_DIR, base_name + ".jpg")
            image_path = alt_path if os.path.exists(alt_path) else None
        if image_path is None:
            print(f"WARNING: image not found for {base_name}, skipping")
            continue
        image = cv2.imread(image_path)
        pred_path = os.path.join(PREDICTIONS_DIR, base_name + ".txt")
        predictions = (get_boxInfo_YOLO(pred_path) if PRED_FORMAT == "xywh_norm" else get_boxInfo_xyxy_pixels(pred_path))

        match_result = get_matching_iou(image, truth_class, truth_box, predictions, PRED_FORMAT)

        iou_value, confidence = match_result["same_class_iou"], match_result["same_class_confidence"]

        is_correct = iou_value >= IOU_THRESHOLD

        attributes = parse_attributes(base_name)
        if attributes is None:
            continue

        records.append({
            "base_name": base_name,
            "class_name": CLASS_NAMES[truth_class],
            "iou": iou_value,
            "is_correct": is_correct,
            "confidence": confidence,
            "wrong_class_iou": match_result["wrong_class_iou"],
            "wrong_class_confidence": match_result["wrong_class_confidence"],
            "pred_box_count": len(predictions),
            "same_class_good_matches": match_result["same_class_good_matches"],
            "nms_failure": match_result["same_class_good_matches"] > 1,
            "attributes": attributes,
        })    

        print(f"{base_name} ({CLASS_NAMES[truth_class]}): best IoU = {iou_value:.3f}, confidence={confidence}")

    #This last section writes inside the .txt files the results. Here also the single object instances appears.
    overall = compute_stats(records)

    print(f"\n{MODEL_NAME} -> {format_stats_line('Overall', overall)}")

    with open(RESULTS_FILE, "a") as f:
        f.write(f"\n=== {MODEL_NAME} ===\n")
        f.write(format_stats_line("Overall", overall) + f", iou_threshold={IOU_THRESHOLD}\n")
        f.write("IoU sweep (Overall): " + format_iou_sweep(records) + "\n")
        f.write("Confidence sweep (Overall): " + format_confidence_sweep(records) + "\n")

        f.write("\n-- By class --\n")
        for class_name in CLASS_NAMES:
            class_num = CLASS_NAMES.index(class_name) + 1
            cond = {"class_num": class_num}
            stats = compute_stats(records, cond)
            f.write("  " + format_stats_line(class_name, stats) + "\n")
            f.write("    IoU sweep: " + format_iou_sweep(records, cond) + "\n")
            f.write("    Confidence sweep: " + format_confidence_sweep(records, cond) + "\n")

        f.write("\n-- By light --\n")
        for val in ["day", "night"]:
            cond = {"light": val}
            f.write("  " + format_stats_line(val, compute_stats(records, {"light": val})) + "\n")
            f.write("    IoU sweep: " + format_iou_sweep(records, cond) + "\n")
            f.write("    Confidence sweep: " + format_confidence_sweep(records, cond) + "\n")

        f.write("\n-- By view --\n")
        
        for val in ["front", "lateral", "rear", "top"]:
            cond = {"view": val}
            f.write("  " + format_stats_line(val, compute_stats(records, {"view": val})) + "\n")
            f.write("    IoU sweep: " + format_iou_sweep(records, cond) + "\n")
            f.write("    Confidence sweep: " + format_confidence_sweep(records, cond) + "\n")

        f.write("\n-- By distance --\n")
        for val in ["close", "far"]:
            cond = {"distance": val}
            f.write("  " + format_stats_line(val, compute_stats(records, {"distance": val})) + "\n")
            f.write("    IoU sweep: " + format_iou_sweep(records, cond) + "\n")
            f.write("    Confidence sweep: " + format_confidence_sweep(records, cond) + "\n")

        f.write("\n-- By occlusion --\n")
        for val in ["occluded", "not_occluded"]:
            cond = {"occlusion": val}
            f.write("  " + format_stats_line(val, compute_stats(records, {"occlusion": val})) + "\n")
            f.write("    IoU sweep: " + format_iou_sweep(records, cond) + "\n")
            f.write("    Confidence sweep: " + format_confidence_sweep(records, cond) + "\n")

        f.write("\n-- By truncation --\n")
        for val in ["truncated", "not_truncated"]:
            cond = {"truncation": val}
            f.write("  " + format_stats_line(val, compute_stats(records, {"truncation": val})) + "\n")
            f.write("    IoU sweep: " + format_iou_sweep(records, cond) + "\n")
            f.write("    Confidence sweep: " + format_confidence_sweep(records, cond) + "\n")


        
        f.write("\n-- By class + object instance --\n")
        for class_name in CLASS_NAMES:
            class_num = CLASS_NAMES.index(class_name) + 1
            for obj_num in [1, 2, 3]:
                cond = {"class_num": class_num, "object_num": obj_num}
                stats = compute_stats(records, cond)
                f.write(f"  {class_name} obj#{obj_num}: " + format_stats_line("", stats) + "\n")
                f.write("    IoU sweep: " + format_iou_sweep(records, cond) + "\n")
                f.write("    Confidence sweep: " + format_confidence_sweep(records, cond) + "\n")