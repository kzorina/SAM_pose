# SAM_pose

This repository contains additional materials related to the thesis "Temporal Consistency for Object Pose Estimation from Images."

## Repository Structure

### `dataset_generator`
This folder contains the scripts used for generating the static and dynamic synthetic datasets presented in the work.

### `robotic_experiment`
This folder contains the ROS package implemented to perform qualitative evaluation.

### `sam`
This folder contains the Python implementation of our method presented in the work.

### `short_horizon`
This folder contains the Python implementation of our short horizon method presented in the work.

## Usage

For detailed instructions on how to use the scripts and packages, refer to the respective folders. Each folder contains a `README.md` file with specific information and instructions.


## Installation instructions

```
conda create -y -n sampose python=3.12 
conda activate sampose
mamba install -y numpy pinocchio gtsam matplotlib scipy  # sam
mamba install -y pyyaml # short_horizon
mamba install -y opencv pytorch # for gtsam processing

python scripts/eval_bop19_pose.py --renderer_type=vispy --result_filenames=/home/kzorina/work/bop_datasets/hopeVideo/ablation/gtsam_hopeVideo-test_1_3_1_0.1_1_1.00E-09_1.00E-09_10_2.50E-04_2.20E-03_.csv
python scripts/eval_bop19_pose.py --renderer_type=vispy --result_filenames=/home/kzorina/work/bop_datasets/hopeVideo/ablation/mod_hopeVideo-test_4_.csv


python scripts/eval_bop19_pose.py --renderer_type=vispy --result_filenames=/home/ros/kzorina/vojtas/ycbv/test/000048/cosy_ycbv-frames_prediction.csv,/home/ros/kzorina/vojtas/ycbv/test/000049/cosy_ycbv-frames_prediction.csv,/home/ros/kzorina/vojtas/ycbv/test/000050/cosy_ycbv-frames_prediction.csv,/home/ros/kzorina/vojtas/ycbv/test/000051/cosy_ycbv-frames_prediction.csv,/home/ros/kzorina/vojtas/ycbv/test/000052/cosy_ycbv-frames_prediction.csv,/home/ros/kzorina/vojtas/ycbv/test/000053/cosy_ycbv-frames_prediction.csv,/home/ros/kzorina/vojtas/ycbv/test/000054/cosy_ycbv-frames_prediction.csv,/home/ros/kzorina/vojtas/ycbv/test/000055/cosy_ycbv-frames_prediction.csv,/home/ros/kzorina/vojtas/ycbv/test/000056/cosy_ycbv-frames_prediction.csv,/home/ros/kzorina/vojtas/ycbv/test/000057/cosy_ycbv-frames_prediction.csv,/home/ros/kzorina/vojtas/ycbv/test/000058/cosy_ycbv-frames_prediction.csv,/home/ros/kzorina/vojtas/ycbv/test/000059/cosy_ycbv-frames_prediction.csv --eval_path /home/ros/kzorina/vojtas/ycbv/all_results.txt


python scripts/eval_bop19_pose.py --renderer_type=vispy --result_filenames=/home/ros/sandbox_mf/data/local_data_happypose/results/ycbv-debug_no_opt/ycbv.bop19/detector+SO3_grid/bop_evaluation/refiner-final_ycbv-test.csv


python scripts/eval_bop19_pose.py --renderer_type=vispy --result_filenames=/home/ros/kzorina/vojtas/ycbv/ablation/gtsam_ycbv-test_1_0_1.0_1e-08_1e-07_0.1_0.17453292519943295_2.50E-05_1.20E-04_.csv

python scripts/eval_bop19_pose.py --renderer_type=vispy --result_filenames=/home/ros/kzorina/vojtas/ycbv/ablation/gtsam_ycbv-test_1_0_1.0_1e-08_1e-07_0.1_0.17453292519943295_2.50E-05_1.25E-05_.csv

python scripts/eval_bop19_pose.py --renderer_type=vispy --result_filenames=/home/ros/kzorina/vojtas/ycbv/cosy_ycbv-test_frames_prediction.csv

```


## Steps to run on YCBV dataset
Run cosypose inference
```
/home/ros/miniconda3/envs/happypose/bin/python /home/ros/kzorina/vojtas/gtsam_playground/scripts/run_inference_on_ycbv.py
```
Combine (merge) all predictions into one file
```
import csv
output_file = '/home/ros/kzorina/vojtas/ycbv/cosy_ycbv-test_frames_prediction.csv'

# Merge CSV files
with open(output_file, 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    for i, test_id in enumerate([48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]):
        with open(f"/home/ros/kzorina/vojtas/ycbv/test/0000{test_id}/cosy_ycbv-test_frames_prediction.csv", 'r') as infile:
            if i != 0:
                infile.readline()
            reader = csv.reader(infile)
            for row in reader:
                writer.writerow(row)
```
Evaluate BOP on subset of predictions (bop env)
```
python scripts/eval_bop19_pose.py --renderer_type=vispy --result_filenames=/home/ros/kzorina/vojtas/ycbv/cosy_ycbv-test_frames_prediction.csv
```

Run short horizon SAM
```
conda activate sampose
python short_horizon/main.py
python scripts/eval_bop19_pose.py --renderer_type=vispy --result_filenames=/home/ros/kzorina/vojtas/ycbv/ablation/samshorthorizon_ycbv-test_cosy_synt_real_0.0_threshold_1_3_1_0.1_1_1.00E-09_1.00E-09_10_2.50E-04_2.20E-03_.csv
```

Run SAM Pose
```
conda activate sampose
python sam/ablation.py
# to repeate const pose values on cosy hope
```