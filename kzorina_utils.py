from pathlib import Path

#####
mode = 'ycbv'
#####


ycbv_path = Path("/home/ros/kzorina/vojtas/ycbv")
ycbv_test_dirs = [i for i in range(48, 60)]


def merge_csv_files(input_file_template: str, sub_folder_ids: list[int], output_file: str):

saving results to: /home/ros/kzorina/vojtas/ycbv/ablation/samshorthorizon_ycbv-test_cosy_synt_real_0.0_threshold_1_3_1_0.1_1_1.00E-09_1.00E-09_10_2.50E-04_2.20E-03_.csv


if __name__ == "__main__":
    DATASET_NAME = 'ycbv'
    METHOD_BACKBONE = 'cosy'
    COMMENT = 'synt_real_0.0_threshold'
    mod = 1
    input_template = str(ycbv_path) + '/0000%d/' + f'samshorthorizon_{DATASET_NAME}-test_{METHOD_BACKBONE}_{COMMENT}_{mod}_{str(sam_settings)}_.csv')
    merge_csv_files()
    # if mode == 'ycbv':
    #     test_dirs = [ycbv_path / test_dir for test_dir in ycbv_test_dirs]
    # else:
    #     raise ValueError(F"Unknown mode {mode}")

    # for test_dir in test_dirs:
    #     for f in test_dir.iterdir():
    #         print(f)