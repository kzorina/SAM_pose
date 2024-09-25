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
save_list = []
for dir in dirs:
    # print(dir)
    method, dataset, backbone = dir.split('_')[:3]
    if 'noreject' in dir:
        method += '_noreject'
    tvt, rvt = dir.split('_')[-2:]
    try:
        dyn_stat, orient = rvt_indic[float(rvt)]
    except:
        dyn_stat, orient = 'unknown', 'unknown'
    scores = json.load(open(str(Path(eval_dir) / dir / 'scores_bop24.json')))
    save_list.append((method, dataset, backbone, dyn_stat, orient, scores['bop24_mAP']))
save_list.sort(key=lambda x: '_'.join(x[:5]), reverse=True)
# print(f"{method}, {dataset}, {backbone} | {tvt}, {rvt} recall = {}")
for (method, dataset, backbone, dyn_stat, orient, score) in save_list:
    print(f"{method}, {dataset}, {backbone} | {dyn_stat}, {orient} precision = {score}")
