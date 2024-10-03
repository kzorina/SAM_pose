import pathlib
import pickle
from pathlib import Path
from GlobalParams import GlobalParams
import numpy as np
import json

def display_table(data_dict, col_width=10):
    row_names = sorted(list(set([t for (t, r) in data_dict.keys()])))
    column_names = sorted(list(set([r for (t, r) in data_dict.keys()])))

    # Header row (column names in scientific notation)
    header = " " * col_width  # Empty space for the corner where row and column names meet
    header += "".join(f"{name:.1e}".rjust(col_width) for name in column_names)
    print(header)

    # Print the table
    for row in row_names:
        # Print row header (row name in scientific notation)
        row_str = f"{row:.1e}".rjust(col_width)

        # Print row values
        for col in column_names:
            # (tvt, rvt): metric
            value = data_dict.get((row, col), 0.0)
            row_str += "{:10.2f}".format(value)  # Format the value to 2 decimal places
        # Print the entire row
        print(row_str)


# Variant 2: load from bop saved logs
bop_log_dir = Path('/home/ros/kzorina/vojtas/bop_eval')
METHOD_BACKBONE = ''

COMMENT = ''
DATASET_NAME = 'hopeVideo'
which_modality = 'static'  # 'static', 'dynamic'
metrics = ['vsd', 'mssd', 'msdp']

if which_modality == 'static':
    base_params = GlobalParams(
                        cov_drift_lin_vel=0.00000001,
                        cov_drift_ang_vel=0.0000001,
                        outlier_rejection_treshold_trans = 0.10,
                        outlier_rejection_treshold_rot = 10*np.pi/180,
                        t_validity_treshold=1.,
                        R_validity_treshold=0.00012,
                        # t_validity_treshold=0.000025,
                        # R_validity_treshold=0.00075,
                        max_derivative_order=0,
                        reject_overlaps=0.05)
elif which_modality == 'dynamic':
    base_params = GlobalParams(
                        cov_drift_lin_vel=0.1,
                        cov_drift_ang_vel=1,
                        outlier_rejection_treshold_trans=0.10,
                        outlier_rejection_treshold_rot=10*np.pi/180,
                        t_validity_treshold=0.000005,
                        R_validity_treshold=0.00075,
                        max_derivative_order=1,
                        reject_overlaps=0.05)
# recall_data = {}
# precision_data = {}
metric_save = {}
for metric in metrics:
    metric_save[metric] = {}
for obj_pose_noise_t_std in [0., 0.005, 0.01, 0.05, 0.1]:
    for obj_pose_noise_r_std in [0., 0.01, 0.05, 0.1, 0.25]:
        SAVE_CSV_COMMENT = f'noisy-object-{obj_pose_noise_t_std}-{obj_pose_noise_r_std}'
        eval_dir = f'gtsam{SAVE_CSV_COMMENT}_{DATASET_NAME}-test_{METHOD_BACKBONE}{COMMENT}{str(base_params)}'
        print(bop_log_dir / eval_dir / 'scores_bop19.json')
        scores = str(bop_log_dir / eval_dir / 'scores_bop19.json')
        if not pathlib.Path(scores).exists():
            print("Folder does not exist: ", eval_dir)
            continue
        data = json.load(open(scores))
        for metric in metrics:
            metric_save[metric][(obj_pose_noise_t_std, obj_pose_noise_r_std)] = data['bop19_average_recall_' + metric]
        # recall_data[(tvt, rvt)] = np.mean([data['bop19_average_recall_' + m] for m in metrics])
        # precision_data[(tvt, rvt)] = np.mean([data['bop19_average_precision_' + m] for m in metrics])
for metric in metrics:
    print(f"{metric} results")
    display_table(metric_save[metric])
# print("Recall results")
# display_table(recall_data)
# print("Precision results")
# display_table(precision_data)