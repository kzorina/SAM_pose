import os
from pathlib import Path
import json

eval_dir = '/home/ros/sandbox_mf/bop_toolkit/data/evals'
# gtsam_ycbv-test_mega_1_1_1.0_0.1_1_0.1_0.17453292519943295_5.00E-06_9.37E-04
root, dirs, files = next(os.walk(eval_dir))
for dir in dirs:
    print(dir)
    method, dataset, backbone = dir.split('_')[:3]
    tvt, rvt = dir.split('_')[-2:]
    scores = json.load(open(str(Path(eval_dir) / dir / 'scores_bop24.json')))
    # print(f"{method}, {dataset}, {backbone} | {tvt}, {rvt} recall = {}")
    print(f"{method}, {dataset}, {backbone} | {tvt}, {rvt} precision = {scores['bop24_mAP']}")
