import os
from pathlib import Path
import json
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Create Configuration')
parser.add_argument('--which_bop', type=str, help='Which bop results to print',
       default="bop19")

args = parser.parse_args()
rvt_indic = {
    0.000937: ('dynamic', 'precision'),
    0.00187: ('dynamic', 'recall'),
    0.0000125: ('static', 'precision'),
    0.00012: ('static', 'reall')
}

if args.which_bop == 'bop19':
    eval_dir = '/home/ros/kzorina/vojtas/bop_eval'
    metrics = ['ad', 'adi', 'add']
elif args.which_bop == 'bop24':
    eval_dir = '/home/ros/sandbox_mf/bop_toolkit/data/evals'
else:
    raise ValueError(f"Unknown bop {args.which_bop}")

root, dirs, files = next(os.walk(eval_dir))
save_list = []
for dir in dirs:
    # print(dir)
    if len(dir.split('_')) < 3:
        print(dir)
        continue
    method, dataset, backbone = dir.split('_')[:3]
    if 'noreject' in dir:
        method += '_noreject'
    tvt, rvt = dir.split('_')[-2:]
    try:
        dyn_stat, orient = rvt_indic[float(rvt)]
    except:
        dyn_stat, orient = 'unknown', 'unknown'
    scores = json.load(open(str(Path(eval_dir) / dir / f'scores_{args.which_bop}.json')))
    if args.which_bop == 'bop19':
        # print(scores.keys())
        metrics_present = True
        for m in metrics:
            if 'bop19_average_recall_' + m not in scores.keys():
                metrics_present = False
        if not metrics_present:
            continue
        avg_rec = np.mean([scores['bop19_average_recall_' + m] for m in metrics])
        avg_prec = np.mean([scores['bop19_average_precision_' + m] for m in metrics])
        save_list.append((method, dataset, backbone, dyn_stat, orient, 'recall', avg_rec))
        save_list.append((method, dataset, backbone, dyn_stat, orient, 'precision', avg_prec))
    if args.which_bop == 'bop24':
        save_list.append((method, dataset, backbone, dyn_stat, orient, 'precision', scores['bop24_mAP']))
save_list.sort(key=lambda x: '_'.join(x[:5]), reverse=True)
# print(f"{method}, {dataset}, {backbone} | {tvt}, {rvt} recall = {}")
for (method, dataset, backbone, dyn_stat, orient, metric, score) in save_list:
    print(f"{method}, {dataset}, {backbone} | {dyn_stat}, {orient} {metric} = {score}")
