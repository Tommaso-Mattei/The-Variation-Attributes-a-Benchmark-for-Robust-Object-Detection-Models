# The-Variation-Attributes-a-Benchmark-for-Robust-Object-Detection-Models
Master's degree thesis about the creation of a dataset/benchmark with complex conditions to test/fine tune detection models.

## The goal

TVA (The Variation Attributes) is a dataset created to be used as a benchmark for models, testing themselves on complex samples full of variations. The naming convention of the dataset allows to analyze beyond the simple global metrics, focusing on particularly difficult variations to see how much the model is affected.

The variations presented are the following: class(bicycle, bottle, chair, monitor, pottedplant), instance(3 different object instances), light (day/night), POV (front, lateral, rear, top), distance (close/far), occlusion (present/not present), truncation (present/not present).

The code presented here can be used mainly to gather results and metrics from a model run on the dataset.

## Reproducibility info

The dataset itself can be found here: https://universe.roboflow.com/datasets-jqohk/the-variation-attributes-a-benchmark-for-robust-object-detection-models.

To make use of this repository it is possible to either: replicate the experiments and results or to test a different model against the benchmark. Beyond what's written inside the requirements.txt, it is necessary to download the labeled dataset itself inside the repository and the same should be done, for completeness, for the unlabeled one, which can be found here: https://drive.google.com/file/d/1tXFkX2Ic8jWSMvwPSIP2aZ2x7HS8mwgR/view?usp=sharing.

Additionally, all of the inference outputs used in the study are stored here (https://drive.google.com/file/d/1h50tBBfS-XGoe-0OdnnJxti6X_4FPjM0/view?usp=sharing) and can be analyzed again with ModelResults.py and plot.py if needed.

The Models predict folder contains the code used to get Faster R-CNN, YOLO11 and RT-DETR, pre-trained on Pascal VOC. It also contains the code for the pre-trained Faster R-CNN with COCO. To get YOLO11l and RT-DETRl it is advised to check Ultralytics very intutitive commands (https://docs.ultralytics.com/models/yolo11 ; https://docs.ultralytics.com/models/rtdetr).

Inside the results/original_results folder, there are the results of the previously mentione models on the benchmark.

