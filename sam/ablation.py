import copy
from pathlib import Path
import json
import numpy as np
import pickle
import time
import utils
from utils import load_scene_camera, load_pickle, merge_T_cos_px_counts
from SamWrapper import SamWrapper
from State import *
from GlobalParams import GlobalParams
# from Vizualization_tools import display_factor_graph, animate_refinement, animate_state
from bop_tools import convert_frames_to_bop, export_bop
import copy
import os
import shutil
import multiprocessing
import argparse
from consts import HOPE_OBJECT_NAME_TO_ID
import pinocchio as pin

parser = argparse.ArgumentParser(description='Create Configuration')
parser.add_argument('--dynamic', action=argparse.BooleanOptionalAction)

args = parser.parse_args()

# METHOD_BACKBONE = 'cosy_'
# COMMENT = 'synt_real_0.0_threshold_'
# COMMENT = 'synt_real_0.0_threshold_noreject_'
# SAVE_CSV_COMMENT = 'search-parameters2'
# METHOD_BACKBONE = 'mega_'
# COMMENT = '0.7-threshold_'

# for hope
METHOD_BACKBONE = ''
COMMENT = ''
# SAVE_CSV_COMMENT = '-measurement-covariance-prime-size-independent'

DATASETS_PATH = Path("/home/ros/kzorina/vojtas")
# DATASET_NAME = "ycbv"
DATASET_NAME = "hopeVideo"
# DATASET_NAME = "SynthDynamicOcclusion"
model_info = json.load(open(DATASETS_PATH / DATASET_NAME / 'models/models_info.json'))
obj_radius = {k: v['diameter'] / 2 for k, v in model_info.items()}
OBJECT_NAME_TO_ID = HOPE_OBJECT_NAME_TO_ID if DATASET_NAME == 'hopeVideo' else None

# CAMERA_POSE_NOISE_T_STD = 0.005
# CAMERA_POSE_NOISE_R_STD = 0.1
# SAVE_CSV_COMMENT = f'noisy-camera-{CAMERA_POSE_NOISE_T_STD}-{CAMERA_POSE_NOISE_R_STD}'

def apply_noise(pose, t_std, r_std):
    pose = pin.SE3(pose[:3, :3], pose[:3, 3])
    noise = np.zeros(6)
    if t_std is not None:
        noise[:3] = np.random.normal(0, t_std, 3 )  
    if r_std is not None:
        noise[3:] = np.random.normal(0, np.deg2rad(r_std), 3)  
    pose_noisy = pose.act(pin.exp6(noise)) 

    return pose_noisy.homogeneous

def __refresh_dir(path):
    """
    Wipes a directory and all its content if it exists. Creates a new empty one.
    :param path:
    """
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=False, onerror=None)
    os.makedirs(path)

def recalculate_validity(results, t_validity_treshold, R_validity_treshold, reject_overlaps):
    recalculated_results = {}
    for result_key in results:
        refined_scene = results[result_key]
        recalculated_refined_scene = []
        for frame in range(len(refined_scene)):
            entry = {}
            for obj_label in refined_scene[frame]:
                entry[obj_label] = []
                for obj_idx in range(len(refined_scene[frame][obj_label])):
                    track = copy.deepcopy(refined_scene[frame][obj_label][obj_idx])

                    validity = State.is_valid(track["Q"], t_validity_treshold, R_validity_treshold)
                    # breakpoint()
                    T_wo = track["T_wo"]
                    Q = track["Q"]
                    #  remove overlapping discrete symmetries
                    if validity and reject_overlaps > 0 or reject_overlaps == -1:  # -1 corresponds to adaptive
                        if reject_overlaps == -1:
                            float_reject_overlap = obj_radius[OBJECT_NAME_TO_ID[obj_label]]
                        else:
                            float_reject_overlap = reject_overlaps
                        for obj_inst in range(len(entry[obj_label])):
                            if entry[obj_label][obj_inst]["valid"]:
                                other_T_wo = entry[obj_label][obj_inst]["T_wo"]
                                other_Q = entry[obj_label][obj_inst]["Q"]
                                dist = np.linalg.norm(T_wo[:3, 3] - other_T_wo[:3, 3])
                                if dist < float_reject_overlap:
                                    if np.linalg.det(Q) > np.linalg.det(other_Q):
                                        validity = False
                                    else:
                                        entry[obj_label][obj_inst]["valid"] = False

                    track["valid"] = validity
                    entry[obj_label].append(track)
            recalculated_refined_scene.append(entry)
        recalculated_results[result_key] = recalculated_refined_scene
    return recalculated_results


def refine_data(scene_camera, frames_prediction, px_counts, params:GlobalParams,
                obj_pose_noise_t_std=None,
                obj_pose_noise_r_std=None,
                cam_pose_noise_t_std=None,
                cam_pose_noise_r_std=None,):
    sam = SamWrapper(params)
    refined_scene = []
    for i in range(len(scene_camera)):  # iter over image_ids
        time_stamp = i/30  # time in secs if fps=30
        T_wc = np.linalg.inv(scene_camera[i]['T_cw'])
        Q_T_wc = np.eye(6)*10**(-6)  # uncertainty in cam pose
        if cam_pose_noise_t_std is not None and cam_pose_noise_r_std is not None:
            np.fill_diagonal(Q_T_wc, [cam_pose_noise_t_std] * 3 + [cam_pose_noise_r_std] *3)
            # breakpoint()
            T_wc = apply_noise(T_wc,
                                t_std=cam_pose_noise_t_std,
                                r_std=cam_pose_noise_r_std
                                )
        Q_T_wc = np.eye(6)*10**(-6)  # uncertainty in cam pose
        
        # Q_T_wc = np.eye(6)*cam_pose_noise_t_std**2 if cam_pose_noise_t_std != 0. else np.eye(6)*10**(-6)  # uncertainty in cam pose
        detections = merge_T_cos_px_counts(frames_prediction[i], px_counts[i])  # T_co and Q for all detected object in a frame.
        if obj_pose_noise_t_std is not None or obj_pose_noise_r_std is not None:
            # breakpoint()
            for obj_label in detections.keys():
                new_list = []
                for el in detections[obj_label]:
                    new_list.append({"T_co":apply_noise(el['T_co'],
                                                        t_std=obj_pose_noise_t_std,
                                                        r_std=obj_pose_noise_r_std
                                                        ), 
                                     "Q":el['Q']})
                detections[obj_label] = new_list

        # add noise to object pose
        sam.insert_detections({"T_wc":T_wc, "Q":Q_T_wc}, detections, time_stamp)
        current_state = sam.get_state()
        # animate_state(current_state, time_stamp)
        refined_scene.append(current_state.get_extrapolated_state(time_stamp, T_wc))
        # display_factor_graph(*utils.parse_variable_index(sam.tracks.factor_graph.isams[sam.tracks.factor_graph.active_chunk].getVariableIndex()))
        # time.sleep(1)
        print(f"\r({(i + 1)}/{len(scene_camera)})", end='')
    return refined_scene

def refine_scene(scene_path, params,
                obj_pose_noise_t_std=None,
                obj_pose_noise_r_std=None,
                cam_pose_noise_t_std=None,
                cam_pose_noise_r_std=None,):
    print('refining scene ', scene_path)
    scene_camera = load_scene_camera(scene_path / "scene_camera.json")
    frames_prediction = load_pickle(scene_path / f"{METHOD_BACKBONE}{COMMENT}frames_prediction.p")
    px_counts = load_pickle(scene_path / f"{METHOD_BACKBONE}{COMMENT}frames_px_counts.p")
    # frames_prediction = load_pickle(scene_path / "frames_prediction.p")
    # px_counts = load_pickle(scene_path / "frames_px_counts.p")
    refined_scene = refine_data(scene_camera, frames_prediction, px_counts, params,
                                obj_pose_noise_t_std=obj_pose_noise_t_std,
                                obj_pose_noise_r_std=obj_pose_noise_r_std,
                                cam_pose_noise_t_std=cam_pose_noise_t_std,
                            cam_pose_noise_r_std=cam_pose_noise_r_std,)
    return refined_scene

def anotate_dataset(DATASETS_PATH, DATASET_NAME, scenes, 
                    params, 
                    dataset_type='hope', 
                    obj_pose_noise_t_std=None,
                    obj_pose_noise_r_std=None,
                    cam_pose_noise_t_std=None,
                    cam_pose_noise_r_std=None,
                    which_modality='static', 
                    load_scene=False):
    results = {}
    print(f"scenes: {scenes}")
    for scene_num in scenes:
        scene_path = DATASETS_PATH/DATASET_NAME/"test"/ f"{scene_num:06}"
        if load_scene:
            with open(scene_path / f'{METHOD_BACKBONE}{COMMENT}frames_refined_prediction.p', 'rb') as file:
                refined_scene = pickle.load(file)
        else:
            refined_scene = refine_scene(scene_path, params,
                                         obj_pose_noise_t_std=obj_pose_noise_t_std,
                                         obj_pose_noise_r_std=obj_pose_noise_r_std,
                                         cam_pose_noise_t_std=cam_pose_noise_t_std,
                            cam_pose_noise_r_std=cam_pose_noise_r_std,)
            with open(scene_path / f'{METHOD_BACKBONE}{COMMENT}frames_refined_prediction.p', 'wb') as file:
                pickle.dump(refined_scene, file)
        results[scene_num] = refined_scene
    # for tvt in [1e-8, 1e-7, 1e-6, 1e-5]:
    for tvt in [1.]:
        # outlier_rejection_treshold_trans = 0.10,
        # outlier_rejection_treshold_rot = 10 * np.pi / 180,
        # for ortt in [1e-4, 1e-3, 1e-2, 1e-1, 1., 2., 3., 5., 10.]:
        # for tvt in [1e-4, 1e-3, 1e-2, 1e-1, 1., 2., 3., 5., 10.]:
        #     for rvt in [1]:
        # for rvt in [0.006,0.003,0.00263,0.00225,0.00187,0.00165,0.0015,0.00113,0.000937,0.00075,0.000563,0.000375,0.000188]
        forked_params = copy.deepcopy(params) 
        # forked_params.outlier_rejection_treshold_trans = ortt
        forked_params.t_validity_treshold = tvt
        # rvt_list = [1., 1.25, 1.5, 1.75, 2., 2.25, 2.5, 2.75, 3]
        # ortr_list = [1e-4, 1e-3, 1e-2, 1e-1, 1., 2., 3., 5., 10.]
        # rvt_list = [0.0000125, 0.00012, 1e-4, 1e-3, 1e-2]
        # rvt_list = [1.2e-05, 1.6e-05, 2e-05, 2.5e-05, 3.1e-05, 4e-05, 5e-05, 6.3e-05, 7.9e-05, 0.0001, 0.000125, 0.000158, 0.000199, 0.00025, 0.000315, 0.000397, 0.0005, 0.000629, 0.000792, 0.000998, 0.001256, 0.001582, 0.001992, 0.002508, 0.003158, 0.003977, 0.005008, 0.006306, 0.007941]
        # rvt_list = [0.0000125, 0.00012] if which_modality == 'static' else [0.000937, 0.00187]
        # for rvt in [0.000937, 0.00187]: # precision oriented, recall oriented for dynamic
        for rvt in [0.0000125, 0.00012]: # precision oriented, recall oriented for static
        # for ortr in ortr_list:
        # for rvt in rvt_list: # precision oriented, recall oriented for static
        # for rvt in [0.00012]: #  recall oriented for static
        # for rvt in [1]: # precision oriented, recall oriented for static
        # for rvt in [0.0006400,0.0003200,0.0001600,0.0001200,0.0000800,0.0000400,0.0000200,0.0000175,0.0000150,0.0000125,0.0000100,0.0000075,0.0000050,0.0000025,0.0000010]:
            # forked_params = copy.deepcopy(params)
            # forked_params.R_validity_treshold = params.R_validity_treshold * rvt
            # forked_params.t_validity_treshold = params.t_validity_treshold * tvt
            # forked_params.outlier_rejection_treshold_rot = ortr
            # forked_params = copy.deepcopy(params)
            forked_params.R_validity_treshold = rvt

            recalculated_results = recalculate_validity(results,
                                                        forked_params.t_validity_treshold,
                                                        forked_params.R_validity_treshold,
                                                        forked_params.reject_overlaps)
            # SAVE_CSV_COMMENT = f'noisy-object-{obj_pose_noise_t_std}-{obj_pose_noise_r_std}'
            # SAVE_CSV_COMMENT = f'noisy-camera-{cam_pose_noise_t_std}-{cam_pose_noise_r_std}'
            # SAVE_CSV_COMMENT = f'noisy-camera-std-based-q-{cam_pose_noise_t_std}-{cam_pose_noise_r_std}'
            # SAVE_CSV_COMMENT = f'-baseline-new-reject-new-cov'
            SAVE_CSV_COMMENT = f'-baseline-new-reject'
            output_name = f'gtsam{SAVE_CSV_COMMENT}_{DATASET_NAME}-test_{METHOD_BACKBONE}{COMMENT}{str(forked_params)}.csv'
            print('saving final result to ', output_name)
            export_bop(convert_frames_to_bop(recalculated_results, dataset_type), DATASETS_PATH / DATASET_NAME / "ablation" / output_name)

def main():

    

    scenes_dict = {
        'hopeVideo': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'SynthDynamicOcclusion': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'ycbv': [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
    }
    start_time = time.time()
    dataset_type = "ycbv" if DATASET_NAME == 'ycbv' else "hope"
    # reject_overlaps = 0.05
    reject_overlaps = -1  # for radius based overlap threshold
    # dataset_type = "hope"

    # DATASET_NAME = "SynthDynamicOcclusion"
    # DATASET_NAME = "SynthStatic"


    scenes = scenes_dict[DATASET_NAME]
    # scenes = [0, 1, 2]
    # scenes = [0]
    # DATASET_NAME = "ycbv_test_bop19"
    # scenes = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
    which_modality = 'dynamic' if args.dynamic else 'static' # 'static', 'dynamic'
    # if 'accel' in SAVE_CSV_COMMENT:
    #     which_modality = 'accel'
    #     print("RUNNING acceleration model")

    pool = multiprocessing.Pool(processes=15)

    # __refresh_dir(DATASETS_PATH / DATASET_NAME / "ablation")
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
                                reject_overlaps=reject_overlaps)
    elif which_modality == 'dynamic':
        base_params = GlobalParams(
                                cov_drift_lin_vel=0.1,
                                cov_drift_ang_vel=1,
                                outlier_rejection_treshold_trans=0.10,
                                outlier_rejection_treshold_rot=10*np.pi/180,
                                t_validity_treshold=0.000005,
                                R_validity_treshold=0.00075,
                                max_derivative_order=1,
                                reject_overlaps=reject_overlaps)
    elif which_modality == 'accel':
        base_params = GlobalParams(
                                cov_drift_lin_vel=0.1,
                                cov_drift_ang_vel=1,
                                outlier_rejection_treshold_trans=0.10,
                                outlier_rejection_treshold_rot=10*np.pi/180,
                                t_validity_treshold=0.000005,
                                R_validity_treshold=0.00075,
                                max_derivative_order=2,
                                reject_overlaps=reject_overlaps)
    else:
        raise ValueError(f"Unknown modality {which_modality}")
    # to not reject anything
    if 'noreject' in COMMENT:
        base_params.reject_overlaps = 0
    forked_params = copy.deepcopy(base_params)
    obj_pose_noise_t_std = None
    obj_pose_noise_r_std = None
    cam_pose_noise_t_std = None
    cam_pose_noise_r_std = None

    # for obj_pose_noise_t_std in [0., 0.005, 0.01, 0.015, 0.02]:
    #     for obj_pose_noise_r_std in [0., 1.75, 2.5, 3.75, 5]:
    # for cam_pose_noise_t_std in [0., 0.005, 0.01, 0.015, 0.02]:
    #     for cam_pose_noise_r_std in [0., 1.75, 2.5, 3.75, 5]:
    #         anotate_dataset(DATASETS_PATH, DATASET_NAME, scenes, forked_params, 
    #                         dataset_type=dataset_type, 
    #                         obj_pose_noise_t_std=obj_pose_noise_t_std,
    #                         obj_pose_noise_r_std=obj_pose_noise_r_std,
    #                         cam_pose_noise_t_std=cam_pose_noise_t_std,
    #                         cam_pose_noise_r_std=cam_pose_noise_r_std,
    #                         which_modality=which_modality,
    #                         load_scene=False)

    forked_params = copy.deepcopy(base_params)
    anotate_dataset(DATASETS_PATH, DATASET_NAME, scenes, forked_params, dataset_type, which_modality=which_modality, 
                    load_scene=False)



    # pool.close()
    # pool.join()

    # refined_scene = load_pickle(DATASETS_PATH/DATASET_NAME/"test"/f"{0:06}"/'frames_refined_prediction.p')
    # scene_gt = utils.load_scene_gt(DATASETS_PATH/DATASET_NAME/"test"/f"{0:06}"/'scene_gt.json', list(utils.HOPE_OBJECT_NAMES.values()))
    # scene_camera = load_scene_camera(DATASETS_PATH/DATASET_NAME/"test"/f"{0:06}" / "scene_camera.json")
    # animate_refinement(refined_scene, scene_gt, scene_camera)

    print(f"elapsed time: {time.time() - start_time:.2f}s")
    print(f"elapsed time: {utils.format_time(time.time() - start_time)}")

if __name__ == "__main__":
    main()
