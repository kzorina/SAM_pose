import pickle
from pathlib import Path
from GlobalParams import GlobalParams
import numpy as np

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
bop_log_dir = Path('/home/ros/kzorina/vojtas/bop_eval')
DATASET_NAME = 'ycbv'
METHOD_BACKBONE = 'cosy_'
COMMENT = 'synt_real_0.0_threshold_'
which_modality = 'dynamic'  # 'static', 'dynamic'

if which_modality == 'static':
    base_params = GlobalParams(
                        cov_drift_lin_vel=0.00000001,
                        cov_drift_ang_vel=0.0000001,
                        outlier_rejection_treshold_trans = 0.10,
                        outlier_rejection_treshold_rot = 10*np.pi/180,
                        t_validity_treshold=0.000025,
                        R_validity_treshold=0.00075,
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

eval_dir = f'gtsam_{DATASET_NAME}-test_{METHOD_BACKBONE}{COMMENT}{str(base_params)}.csv'
print(bop_log_dir / eval_dir)
# /gtsam_ycbv-test_cosy_synt_real_0.0_threshold_1_1_1.0_0.1_1_0.1_0.174532925│   1.0e-03      0.91      0.76      0.68      0.65      0.64
# 19943295_1.00E-05_1.00E-04/scores_bop19.json
#
# print("Recall results")
# display_table(recall_data)
# print("Precision results")
# display_table(precision_data)