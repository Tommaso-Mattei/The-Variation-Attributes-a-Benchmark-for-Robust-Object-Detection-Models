import re
import glob
import os
import matplotlib.pyplot as plt

#This code analyzes the all the.txt files resulting from the ModelResults.py file and
#it creates for each model result a plot for the confidence sweep and a plot for the IoU sweep
#at last, it also creates two overall plots merging all results
UPLOAD_DIR = "results"
OUT_DIR = "Plots"

#capable of finding the correct lines to parse from the results .txt
IOU_LINE_RE = re.compile(r"IoU sweep.*?:\s*(.*)")
CONF_LINE_RE = re.compile(r"Confidence sweep.*?:\s*(.*)")
 
IOU_PAIR_RE = re.compile(r"acc@([\d.]+)=([\d.]+)")
CONF_PAIR_RE = re.compile(r"acc@conf([\d.]+)=([\d.]+)")
 

#gets thresholds and accuracies
def extract_first_sweep(text, line_re, pair_re):
    for line in text.splitlines():
        m = line_re.search(line)
        if m:
            pairs = pair_re.findall(line)
            if pairs:
                thresholds = [float(t) for t, _ in pairs]
                accuracies = [float(a) for _, a in pairs]
                return thresholds, accuracies
    return None, None
 
 
FILENAME_RE = re.compile(r"^results_(.+)_([^_]+)$")
 
 
def model_name_from_filename(path):
    base = os.path.basename(path)
    stem = re.sub(r"\.txt$", "", base)
    m = FILENAME_RE.match(stem)
    if m:
        model, dataset = m.group(1), m.group(2)
        return f"{model} ({dataset})"
    return re.sub(r"^results_", "", stem)
 
#makes names safe for filenames
def slugify(label):
    s = label.replace(" (", "_").replace(")", "")
    s = re.sub(r"\s+", "-", s.strip())
    s = re.sub(r"[^A-Za-z0-9_\-]", "", s)
    return s
 
#creates filename properly 
def build_output_filename(prefix, labels):
    """e.g. prefix='iou_sweep', labels=['FasterRCNN (COCO)', 'YOLO11 (COCO)']
    -> 'iou_sweep_FasterRCNN-COCO_YOLO11-COCO.png'
    Falls back to a generic name if there are too many models (keeps filenames sane)."""
    if not labels:
        return f"{prefix}_plot.png"
    if len(labels) > 4:
        return f"{prefix}_comparison_{len(labels)}models.png"
    slugs = "_".join(slugify(l) for l in labels)
    return f"{prefix}_{slugs}.png"
 
#scans a directory in which are all the .txt for results, creating all the relevant plots in the output directory 
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
 
    txt_files = sorted(glob.glob(os.path.join(UPLOAD_DIR, "*.txt")))
    if not txt_files:
        print("No .txt files found in", UPLOAD_DIR)
        return
 
    iou_data = {}
    conf_data = {}
 
    for path in txt_files:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
 
        iou_th, iou_acc = extract_first_sweep(text, IOU_LINE_RE, IOU_PAIR_RE)
        conf_th, conf_acc = extract_first_sweep(text, CONF_LINE_RE, CONF_PAIR_RE)
 
        name = model_name_from_filename(path)
 
        if iou_th:
            iou_data[name] = (iou_th, iou_acc)
        else:
            print(f"[WARN] No IoU sweep found in {path}")
 
        if conf_th:
            conf_data[name] = (conf_th, conf_acc)
        else:
            print(f"[WARN] No Confidence sweep found in {path}")
 
    #Accuracy vs IoU global
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for name, (th, acc) in iou_data.items():
        ax.plot(th, acc, marker='o', linewidth=2, label=name)
    ax.set_xlabel('IoU Threshold', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy vs. IoU Threshold', fontsize=13)
    if iou_data:
        any_th = next(iter(iou_data.values()))[0]
        ax.set_xticks(any_th)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(fontsize=9)
    fig.tight_layout()
    iou_filename = build_output_filename("iou_sweep", list(iou_data.keys()))
    fig.savefig(os.path.join(OUT_DIR, iou_filename), dpi=300)
    plt.close(fig)
 
    #Accuracy vs Confidence global
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for name, (th, acc) in conf_data.items():
        ax.plot(th, acc, marker='o', linewidth=2, label=name)
    ax.set_xlabel('Confidence Threshold', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy vs. Confidence Threshold', fontsize=13)
    if conf_data:
        any_th = next(iter(conf_data.values()))[0]
        ax.set_xticks(any_th)
        ax.tick_params(axis='x', rotation=45)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(fontsize=9)
    fig.tight_layout()
    conf_filename = build_output_filename("confidence_sweep", list(conf_data.keys()))
    fig.savefig(os.path.join(OUT_DIR, conf_filename), dpi=300)
    plt.close(fig)
 
    #individual accuracy vs IoU
    for name, (th, acc) in iou_data.items():
        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.plot(th, acc, marker='o', linewidth=2, color='#1f77b4')
        ax.set_xlabel('IoU Threshold', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(f'Accuracy vs. IoU Threshold — {name}', fontsize=13)
        ax.set_xticks(th)
        ax.grid(True, linestyle=':', alpha=0.6)
        fig.tight_layout()
        fname = build_output_filename("iou_sweep", [name])
        fig.savefig(os.path.join(OUT_DIR, fname), dpi=300)
        plt.close(fig)
 
    #individual accuracy vs Confidence
    for name, (th, acc) in conf_data.items():
        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.plot(th, acc, marker='o', linewidth=2, color='#ff7f0e')
        ax.set_xlabel('Confidence Threshold', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(f'Accuracy vs. Confidence Threshold — {name}', fontsize=13)
        ax.set_xticks(th)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, linestyle=':', alpha=0.6)
        fig.tight_layout()
        fname = build_output_filename("confidence_sweep", [name])
        fig.savefig(os.path.join(OUT_DIR, fname), dpi=300)
        plt.close(fig)
 
    print("Done. Models processed:", list(iou_data.keys()))
    print("Saved combined:", iou_filename, "and", conf_filename)
    print(f"Saved {len(iou_data)} individual IoU plots and {len(conf_data)} individual Confidence plots.")
 
 
if __name__ == "__main__":
    main()