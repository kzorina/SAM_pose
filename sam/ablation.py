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

# METHOD_BACKBONE = 'cosy_'
# COMMENT = 'synt_real_0.0_threshold_'
# METHOD_BACKBONE = 'mega_'
# COMMENT = ''
# for hope
METHOD_BACKBONE = ''
COMMENT = ''

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
                    T_wo = track["T_wo"]
                    Q = track["Q"]
                    #  remove overlapping discrete symmetries
                    if validity and reject_overlaps > 0:
                        for obj_inst in range(len(entry[obj_label])):
                            if entry[obj_label][obj_inst]["valid"]:
                                other_T_wo = entry[obj_label][obj_inst]["T_wo"]
                                other_Q = entry[obj_label][obj_inst]["Q"]
                                dist = np.linalg.norm(T_wo[:3, 3] - other_T_wo[:3, 3])
                                if dist < reject_overlaps:
                                    if np.linalg.det(Q) > np.linalg.det(other_Q):
                                        validity = False
                                    else:
                                        entry[obj_label][obj_inst]["valid"] = False

                    track["valid"] = validity
                    entry[obj_label].append(track)
            recalculated_refined_scene.append(entry)
        recalculated_results[result_key] = recalculated_refined_scene
    return recalculated_results


def refine_data(scene_camera, frames_prediction, px_counts, params:GlobalParams):
    sam = SamWrapper(params)
    refined_scene = []
    for i in range(len(scene_camera)):
        time_stamp = i/30
        T_wc = np.linalg.inv(scene_camera[i]['T_cw'])
        Q_T_wc = np.eye(6)*10**(-6)
        detections = merge_T_cos_px_counts(frames_prediction[i], px_counts[i])  # T_co and Q for all detected object in a frame.
        sam.insert_detections({"T_wc":T_wc, "Q":Q_T_wc}, detections, time_stamp)
        current_state = sam.get_state()
        # animate_state(current_state, time_stamp)
        refined_scene.append(current_state.get_extrapolated_state(time_stamp, T_wc))
        # display_factor_graph(*utils.parse_variable_index(sam.tracks.factor_graph.isams[sam.tracks.factor_graph.active_chunk].getVariableIndex()))
        # time.sleep(1)
        print(f"\r({(i + 1)}/{len(scene_camera)})", end='')
    return refined_scene

def refine_scene(scene_path, params):
    print('refining scene ', scene_path)
    scene_camera = load_scene_camera(scene_path / "scene_camera.json")
    frames_prediction = load_pickle(scene_path / f"{METHOD_BACKBONE}{COMMENT}frames_prediction.p")
    px_counts = load_pickle(scene_path / f"{METHOD_BACKBONE}{COMMENT}frames_px_counts.p")
    # frames_prediction = load_pickle(scene_path / "frames_prediction.p")
    # px_counts = load_pickle(scene_path / "frames_px_counts.p")
    refined_scene = refine_data(scene_camera, frames_prediction, px_counts, params)
    return refined_scene

def anotate_dataset(DATASETS_PATH, DATASET_NAME, scenes, params, dataset_type='hope', which_modality='static', load_scene=False):
    results = {}
    print(f"scenes: {scenes}")
    for scene_num in scenes:
        scene_path = DATASETS_PATH/DATASET_NAME/"test"/ f"{scene_num:06}"
        if load_scene:
            with open(scene_path / f'{METHOD_BACKBONE}{COMMENT}frames_refined_prediction.p', 'rb') as file:
                refined_scene = pickle.load(file)
        else:
            refined_scene = refine_scene(scene_path, params)
            with open(scene_path / f'{METHOD_BACKBONE}{COMMENT}frames_refined_prediction.p', 'wb') as file:
                pickle.dump(refined_scene, file)
        results[scene_num] = refined_scene
    for tvt in [1.]:
    # for tvt in [1., 1.25, 1.5, 1.75, 2., 2.25, 2.5, 2.75, 3]:
        # for rvt in [1]:
        # for rvt in [0.006,0.003,0.00263,0.00225,0.00187,0.00165,0.0015,0.00113,0.000937,0.00075,0.000563,0.000375,0.000188]
        forked_params = copy.deepcopy(params) 
        forked_params.t_validity_treshold = tvt
        # rvt_list = [1., 1.25, 1.5, 1.75, 2., 2.25, 2.5, 2.75, 3]
        rvt_list = [0.0000125, 0.00012] if which_modality == 'static' else [0.000937, 0.00187]
        # for rvt in [0.000937, 0.00187]: # precision oriented, recall oriented for dynamic
        # for rvt in [0.0000125, 0.00012]: # precision oriented, recall oriented for static
        for rvt in rvt_list: # precision oriented, recall oriented for static
        # for rvt in [0.0006400,0.0003200,0.0001600,0.0001200,0.0000800,0.0000400,0.0000200,0.0000175,0.0000150,0.0000125,0.0000100,0.0000075,0.0000050,0.0000025,0.0000010]:
            # forked_params = copy.deepcopy(params)
            # forked_params.R_validity_treshold = params.R_validity_treshold * rvt
            # forked_params.t_validity_treshold = params.t_validity_treshold * tvt
            forked_params.R_validity_treshold = rvt

            recalculated_results = recalculate_validity(results, forked_params.t_validity_treshold, forked_params.R_validity_treshold, forked_params.reject_overlaps)
            output_name = f'gtsam_{DATASET_NAME}-test_{METHOD_BACKBONE}{COMMENT}{str(forked_params)}.csv'
            print('saving final result to ', output_name)
            export_bop(convert_frames_to_bop(recalculated_results, dataset_type), DATASETS_PATH / DATASET_NAME / "ablation" / output_name)

def main():
    scenes_dict = {
        'hopeVideo': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'ycbv': [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
    }
    start_time = time.time()
    dataset_type = "ycbv"
    # dataset_type = "hope"
    DATASETS_PATH =  Path("/home/ros/kzorina/vojtas")
    # DATASET_NAME = "SynthDynamicOcclusion"
    # DATASET_NAME = "SynthStatic"
    # DATASET_NAME = "hopeVideo"
    DATASET_NAME = "ycbv"
    scenes = scenes_dict[DATASET_NAME]
    # scenes = [0, 1, 2]
    # scenes = [0]
    # DATASET_NAME = "ycbv_test_bop19"
    # scenes = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
    which_modality = 'dynamic'  # 'static', 'dynamic'

    pool = multiprocessing.Pool(processes=15)

    # __refresh_dir(DATASETS_PATH / DATASET_NAME / "ablation")
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
    else:
        raise ValueError(f"Unknown modality {which_modality}")
    # to not reject anything
    base_params.reject_overlaps = 0
    forked_params = copy.deepcopy(base_params)
    anotate_dataset(DATASETS_PATH, DATASET_NAME, scenes, forked_params, dataset_type, which_modality,
                    load_scene=False)
    # for trans in [1]:
    #     for rot in [1]:
    #         forked_params = copy.deepcopy(base_params)
    #         # forked_params.outlier_rejection_treshold_trans = trans
    #         # forked_params.outlier_rejection_treshold_rot = rot
    #         # pool.apply_async(anotate_dataset, args=(DATASETS_PATH, DATASET_NAME, scenes, forked_params, dataset_type))
    #         anotate_dataset(DATASETS_PATH, DATASET_NAME, scenes, forked_params, dataset_type, which_modality, load_scene=True)
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
