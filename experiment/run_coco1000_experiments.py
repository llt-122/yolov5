from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
val = importlib.import_module("val")


PROJECT = ROOT / "runs" / "val"
RESULTS = ROOT / "experiment" / "results" / "coco1000_metrics_latest.json"
BASE = {"imgsz": 640, "conf_thres": 0.25, "iou_thres": 0.45, "augment": False}
EXPERIMENTS = [
    {"id": "A", "name": "coco1000_A_baseline", **BASE},
    {"id": "S1", "name": "coco1000_S1_img512", **BASE, "imgsz": 512},
    {"id": "S2", "name": "coco1000_S2_img768", **BASE, "imgsz": 768},
    {"id": "S3", "name": "coco1000_S3_conf015", **BASE, "conf_thres": 0.15},
    {"id": "S4", "name": "coco1000_S4_conf035", **BASE, "conf_thres": 0.35},
    {"id": "S5", "name": "coco1000_S5_iou035", **BASE, "iou_thres": 0.35},
    {"id": "S6", "name": "coco1000_S6_iou055", **BASE, "iou_thres": 0.55},
    {"id": "S7", "name": "coco1000_S7_augment", **BASE, "augment": True},
]


def run_group(config):
    started = time.perf_counter()
    results, _, timing = val.run(
        data=str(ROOT / "data" / "coco1000.yaml"),
        weights=str(ROOT / "yolov5s.pt"),
        batch_size=4,
        imgsz=config["imgsz"],
        conf_thres=config["conf_thres"],
        iou_thres=config["iou_thres"],
        max_det=1000,
        device="cpu",
        workers=4,
        augment=config["augment"],
        save_txt=True,
        save_conf=True,
        project=PROJECT,
        name=config["name"],
        exist_ok=True,
        half=False,
        plots=True,
    )
    return {
        **config,
        "precision": results[0],
        "recall": results[1],
        "map50": results[2],
        "map50_95": results[3],
        "preprocess_ms": timing[0],
        "inference_ms": timing[1],
        "nms_ms": timing[2],
        "elapsed_seconds": time.perf_counter() - started,
    }


def best(groups, candidates, key):
    eligible = [group for group in groups if group["id"] in candidates]
    return max(eligible, key=lambda group: (group["map50_95"], -group["inference_ms"]))[key]


def main():
    groups = []
    for config in EXPERIMENTS:
        print(f"Running {config['id']}: {config['name']}", flush=True)
        groups.append(run_group(config))

    final = {
        "id": "Z",
        "name": "coco1000_Z_final_combined",
        "imgsz": best(groups, {"A", "S1", "S2"}, "imgsz"),
        "conf_thres": best(groups, {"A", "S3", "S4"}, "conf_thres"),
        "iou_thres": best(groups, {"A", "S5", "S6"}, "iou_thres"),
        "augment": best(groups, {"A", "S7"}, "augment"),
    }
    print(f"Running selected combination: {final}", flush=True)
    groups.append(run_group(final))

    document = {
        "dataset": "COCO 2017 validation fixed 1000-image subset",
        "dataset_yaml": "data/coco1000.yaml",
        "selection_metric": "threshold-conditioned mAP@0.5:0.95",
        "groups": groups,
        "selected_config": {key: final[key] for key in ["imgsz", "conf_thres", "iou_thres", "augment"]},
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(document, indent=2), encoding="utf-8")
    print(f"Saved metrics to {RESULTS}")


if __name__ == "__main__":
    main()
