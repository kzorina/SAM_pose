import os
from pathlib import Path
import json

# for rvt in [0.000937, 0.00187]: # precision oriented, recall oriented for dynamic
# for rvt in [0.0000125, 0.00012]: # precision oriented, recall oriented for static
rvt_indic = {
    0.000937: ('dynamic', 'precision'),
    0.00187: ('dynamic', 'recall'),
    0.0000125: ('static', 'precision'),
    0.00012: ('static', 'reall')
}

eval_dir = '/home/ros/sandbox_mf/bop_toolkit/data/evals'
# gtsam_ycbv-test_mega_1_1_1.0_0.1_1_0.1_0.17453292519943295_5.00E-06_9.37E-04
root, dirs, files = next(os.walk(eval_dir))
for dir in dirs:
    # print(dir)
    method, dataset, backbone = dir.split('_')[:3]
    tvt, rvt = dir.split('_')[-2:]
    dyn_stat, orient = rvt_indic[float(rvt)]
    scores = json.load(open(str(Path(eval_dir) / dir / 'scores_bop24.json')))
    # print(f"{method}, {dataset}, {backbone} | {tvt}, {rvt} recall = {}")
    print(f"{method}, {dataset}, {backbone} | {dyn_stat}, {orient} precision = {scores['bop24_mAP']}")
