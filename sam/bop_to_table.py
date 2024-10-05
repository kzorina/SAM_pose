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

# # Variant 1: load from manually saved file
# recall_file = '/home/ros/kzorina/vojtas/ycbv/ablation_kz_recall.p'
# precision_file = '/home/ros/kzorina/vojtas/ycbv/ablation_kz_precision.p'
#
# recall_data = pickle.load(open(recall_file, 'rb'))
# precision_data = pickle.load(open(precision_file, 'rb'))

# Variant 2: load from bop saved logs
bop_log_dir = Path('/home/ros/kzorina/vojtas/bop_eval_kz')
# METHOD_BACKBONE = 'cosy_'
# # COMMENT = 'synt_real_0.0_threshold_'
# COMMENT = 'synt_real_0.0_threshold_noreject_'
# SAVE_CSV_COMMENT = 'search-parameters2'
# DATASET_NAME = 'ycbv'
# # METHOD_BACKBONE = 'cosy_'
# # COMMENT = 'synt_real_0.0_threshold_'
which_modality = 'static'  # 'static', 'dynamic'
# metrics = ['ad', 'adi']
metrics = ['vsd', 'mssd', 'mspd']

if which_modality == 'static':
    base_params = GlobalParams(
                        cov_drift_lin_vel=0.00000001,
                        cov_drift_ang_vel=0.0000001,
                        outlier_rejection_treshold_trans = 0.10,
                        outlier_rejection_treshold_rot = 10*np.pi/180,
                        t_validity_treshold=1.,
                        R_validity_treshold=1.,
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
recall_data = {}
precision_data = {}
metric_save = {}
for metric in metrics:
    metric_save[metric] = {}

rvt_list = [0.0000125, 1e-5, 0.00012, 1e-4, 1e-3, 1e-2, 1e-1, 1.]
rvt_list +=[1.2e-05, 1.6e-05, 2e-05, 2.5e-05, 3.1e-05, 4e-05, 5e-05, 6.3e-05, 7.9e-05, 0.0001, 0.000125, 0.000158,
            0.000199, 0.00025, 0.000315, 0.000397, 0.0005, 0.000629, 0.000792, 0.000998, 0.001256, 0.001582, 0.001992,
            0.002508, 0.003158, 0.003977, 0.005008, 0.006306, 0.007941]

for rvt in rvt_list:
    for tvt in [1.]:
        base_params.R_validity_treshold = rvt
        base_params.t_validity_treshold = tvt
        # base_params.outlier_rejection_treshold_trans = ortt
        # base_params.outlier_rejection_treshold_rot = ortr
        
        # eval_dir = f'gtsam{SAVE_CSV_COMMENT}_{DATASET_NAME}-test_{METHOD_BACKBONE}{COMMENT}{str(base_params)}'
        eval_dir = f'gtsam-look-for-params_hopeVideo-test_{str(base_params)}'
        print(bop_log_dir / eval_dir / 'scores_bop19.json')
        scores = str(bop_log_dir / eval_dir / 'scores_bop19.json')
        if not pathlib.Path(scores).exists():
            print("Folder does not exist: ", eval_dir)
            continue
        data = json.load(open(scores))
        for metric in metrics:
            metric_save[metric][(rvt, tvt)] = data['bop19_average_recall_' + metric]
            # metric_save[metric][(ortt, ortr)] = data['bop19_average_recall_' + metric]
        recall_data[(tvt, rvt)] = np.mean([data['bop19_average_recall_' + m] for m in metrics])
        precision_data[(tvt, rvt)] = np.mean([data['bop19_average_precision_' + m] for m in metrics])
# for metric in metrics:
#     print(f"{metric} results")
#     display_table(metric_save[metric])
# print("Recall results")
# display_table(recall_data)
# print("Precision results")
# display_table(precision_data)
print("Recall results [tvt=1]")
for tvt in [1.]:
    for rvt in rvt_list:
        print(f"rvt {rvt}   -   {recall_data[(tvt, rvt)]}")
print("Precision results [tvt=1]")
for tvt in [1.]:
    for rvt in rvt_list:
        print(f"rvt {rvt}   -   {precision_data[(tvt, rvt)]}")
pickle.dump(recall_data, open('recall_data.p', 'wb'))
pickle.dump(precision_data, open('precision_data.p', 'wb'))
