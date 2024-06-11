import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
from Utils import calculate_mse_blockwise, save_model_map, Hierarchical_B, Segmentation, load_selected_frames, divide_into_blocks, save_frames_and_txt
import ants
from Model import block_motion_compensation_affine, affine_registration, block_perspective_sift, AKAZE_feature_compensation_perspective, block_motion_compensation_feature_matching_perspective, block_motion_compensation_block_matching, dct_motion_compensation,block_motion_compensation_affine,block_motion_compensation_feature_matching, block_motion_compensation_gradient,block_motion_compensation_phase_correlation,block_motion_compensation_sift,AKAZE_feature_compensation,block_motion_compensation_lucas_kanade

# Part 1: Load necessary images
frames = load_selected_frames('../data')
ground_truth_frames = load_selected_frames('../data')

# Part 2: Hierarchical-B processing order
processing_order = Hierarchical_B()

# Part 3: Global motion compensation
def global_motion_compensation(target, ref1, ref2):
############################### The Methods #####################################
    methods = [
        block_motion_compensation_affine,
        affine_registration,
        block_perspective_sift,
        block_motion_compensation_feature_matching,
        block_motion_compensation_block_matching,
        block_motion_compensation_phase_correlation,
        block_motion_compensation_sift,
        AKAZE_feature_compensation,
        block_motion_compensation_lucas_kanade,
        AKAZE_feature_compensation_perspective,
        block_motion_compensation_feature_matching_perspective
    ]
#################################################################################
    compensated_frames = []
    id = 1
    for method in methods:
        compensated_frame_ref1 = method(ref1,target)
        compensated_frame_ref2 = method(ref2,target)
        compensated_frames.extend([{'frame': compensated_frame_ref1, 'model_id': id}, {'frame': compensated_frame_ref2, 'model_id': id}])
        id += 1
    return compensated_frames

def post_processing(frames):
    post_processed_frames = {}
    for frame_idx, frame in tqdm(frames.items()):
        denoised_frame = cv2.bilateralFilter(frame, 3, 75, 75)
        post_processed_frames[frame_idx] = denoised_frame
    return post_processed_frames

# Part4: Processing
def process_frames(frames, ground_truth_frames, processing_order):
    predictions = {}
    model_maps = {}

    for target, ref1, ref2 in tqdm(processing_order):
        ref1_frame = frames[ref1]
        ref2_frame = frames[ref2]
        gt_frame = ground_truth_frames[target]

        compensated_frames = global_motion_compensation(gt_frame, ref1_frame, ref2_frame)
        gt_blocks = divide_into_blocks(gt_frame)
        comp_frames_blocks = [divide_into_blocks(comp_frame['frame']) for comp_frame in compensated_frames]
        mse_matrix = calculate_mse_blockwise(gt_blocks, comp_frames_blocks)
        
        final_frame = np.zeros_like(gt_frame)
        model_map = []

        num_blocks = len(gt_blocks)
        num_methods = len(comp_frames_blocks)

        for block_idx in range(num_blocks):
            i, j, gt_block = gt_blocks[block_idx]['i'], gt_blocks[block_idx]['j'], gt_blocks[block_idx]['block']
            best_method_idx = np.argmin(mse_matrix[block_idx])
            best_block = comp_frames_blocks[best_method_idx][block_idx]['block']
            best_id = compensated_frames[best_method_idx]['model_id']
            final_frame[i:i+16, j:j+16] = best_block
            model_map.append(best_id)

        predictions[target] = final_frame
        model_maps[target] = model_map

    predictions = post_processing(predictions)
    return predictions, model_maps

predicted_frames, model_maps = process_frames(frames, ground_truth_frames, processing_order)

# Part 5: Save with MSE selection
save_frames_and_txt(predicted_frames, '../output')
save_model_map(model_maps, '../output')
Segmentation()
# 做物件切分，疊圖的方法
