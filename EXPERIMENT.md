# YOLOv5 COCO1000 Parameter Experiment

This repository is a reproducible YOLOv5s inference-parameter experiment based on a fixed 1,000-image subset of the
COCO 2017 validation set. It compares one factor at a time, selects the best value of each tested factor by
threshold-conditioned `mAP@0.5:0.95`, and validates the selected combination.

The original YOLOv5 documentation remains in [README.md](README.md). The full Chinese experiment report is available
at [docs/YOLOv5_COCO1000_experiment_report.pdf](docs/YOLOv5_COCO1000_experiment_report.pdf).

## Experiment Design

All groups use `yolov5s.pt`, the same 1,000 images, CPU inference, batch size 4, and a maximum of 1,000 detections per
image. The baseline is `imgsz=640`, `conf=0.25`, `IoU=0.45`, and augmented inference disabled.

| Group | Changed setting     | Precision | Recall |  mAP50 | mAP50-95 | ms/image |
| ----- | ------------------- | --------: | -----: | -----: | -------: | -------: |
| A     | Baseline            |    0.6343 | 0.5275 | 0.6140 |   0.4434 |     81.4 |
| S1    | `imgsz=512`         |    0.6395 | 0.5110 | 0.6055 |   0.4397 |     54.6 |
| S2    | `imgsz=768`         |    0.6444 | 0.5362 | 0.6188 |   0.4450 |    122.7 |
| S3    | `conf=0.15`         |    0.6410 | 0.5339 | 0.6090 |   0.4314 |     80.0 |
| S4    | `conf=0.35`         |    0.7210 | 0.4753 | 0.6185 |   0.4541 |     81.6 |
| S5    | `IoU=0.35`          |    0.6414 | 0.5246 | 0.6146 |   0.4432 |     79.0 |
| S6    | `IoU=0.55`          |    0.6194 | 0.5291 | 0.6102 |   0.4404 |     82.4 |
| S7    | Augmented inference |    0.6356 | 0.5298 | 0.6102 |   0.4380 |    188.4 |
| Z     | Final combination   |    0.7278 | 0.4759 | 0.6190 |   0.4557 |    115.4 |

The selected combination is:

```text
weights=yolov5s.pt
imgsz=768
conf_thres=0.35
iou_thres=0.45
augment=False
```

This setting prioritizes strict localization accuracy. It is not the best setting for recall or CPU speed. The AP
values are internal comparisons conditioned on confidence thresholds between 0.15 and 0.35, not official COCO
leaderboard results.

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

PyTorch may be installed separately according to the target CPU or CUDA environment. YOLOv5 downloads
`yolov5s.pt` automatically when it is first required, so model weights are not stored in Git.

## Rebuild the COCO1000 Images

The repository contains the fixed YOLO labels and [manifest](datasets/coco1000/manifest.csv), but not the 1,000 COCO
JPEG files. Download the exact image list recorded by the experiment:

```powershell
python experiment/download_coco1000.py
```

The script writes images to `datasets/coco1000/images/val` and verifies every downloaded file. The dataset config is
[data/coco1000.yaml](data/coco1000.yaml).

## Run the Final Configuration

```powershell
python val.py --data data/coco1000.yaml --weights yolov5s.pt --imgsz 768 --conf-thres 0.35 `
  --iou-thres 0.45 --batch-size 4 --device cpu --max-det 1000 --save-txt --save-conf `
  --project runs/val --name coco1000_Z_final_combined
```

Run all single-factor groups and select the final combination automatically:

```powershell
python experiment/run_coco1000_experiments.py
```

Raw machine-readable results are in [experiment/results/coco1000_metrics.json](experiment/results/coco1000_metrics.json).

## Repository Contents

- `data/coco1000.yaml`: YOLO dataset configuration.
- `datasets/coco1000/labels/val`: 1,000 fixed YOLO-format label files.
- `datasets/coco1000/manifest.csv`: image IDs, official URLs, sizes, categories, and sample origin.
- `experiment/download_coco1000.py`: deterministic image downloader and verifier.
- `experiment/run_coco1000_experiments.py`: baseline, single-factor, selection, and final validation workflow.
- `experiment/results`: metrics and compact summary figures.
- `docs/YOLOv5_COCO1000_experiment_report.pdf`: detailed experiment report.

## Data and License

The source images and annotations are from the [COCO 2017 dataset](https://cocodataset.org/#download). Users are
responsible for following the COCO terms and the licenses associated with individual images. YOLOv5 source code in
this repository remains subject to the license in [LICENSE](LICENSE).
