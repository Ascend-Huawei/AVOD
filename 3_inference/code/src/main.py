
"""
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import cv2
import numpy as np
import argparse
import sys
from pathlib import Path

from utils.KittiDataloader import KittiDataloader
import utils.utils as utils

sys.path.append(".")

from atlas_utils.acl_resource import AclResource 
from atlas_utils.acl_model import Model


def get_rpn_proposals_and_scores(rpn_top_anchors, softmax_scores):
    """Returns the proposals and scores stacked for saving to file.

    Args:
        predictions: A list containing the model outputs.
        0: top anchors
        1: PRED_TOP_OBJECTNESS_SOFTMAX

    Returns:
        proposals_and_scores: A numpy array of shape (number_of_proposals,
            8), containing the rpn proposal boxes and scores.
    """

   
    rpn_top_anchors = utils.anchors_to_box_3d(rpn_top_anchors)
    

    proposals_and_scores = np.column_stack((rpn_top_anchors,
                                            softmax_scores))

    return proposals_and_scores

def get_avod_predicted_boxes_3d_and_scores(final_pred_boxes_3d, final_pred_orientations, final_pred_softmax):
    """Returns the predictions and scores stacked for saving to file.

    Args:
        predictions: A dictionary containing the model outputs.
        box_rep: removed; box_4ca only

    Returns:
        predictions_and_scores: A numpy array of shape
            (number_of_predicted_boxes, 9), containing the final prediction
            boxes, orientations, scores, and types.
    """

    # Calculate difference between box_3d and predicted angle
    ang_diff = final_pred_boxes_3d[:, 6] - final_pred_orientations

    # Wrap differences between -pi and pi
    two_pi = 2 * np.pi
    ang_diff[ang_diff < -np.pi] += two_pi
    ang_diff[ang_diff > np.pi] -= two_pi

    def swap_boxes_3d_lw(boxes_3d):
        boxes_3d_lengths = np.copy(boxes_3d[:, 3])
        boxes_3d[:, 3] = boxes_3d[:, 4]
        boxes_3d[:, 4] = boxes_3d_lengths
        return boxes_3d

    pi_0_25 = 0.25 * np.pi
    pi_0_50 = 0.50 * np.pi
    pi_0_75 = 0.75 * np.pi

    # Rotate 90 degrees if difference between pi/4 and 3/4 pi
    rot_pos_90_indices = np.logical_and(pi_0_25 < ang_diff,
                                        ang_diff < pi_0_75)
    final_pred_boxes_3d[rot_pos_90_indices] = \
        swap_boxes_3d_lw(final_pred_boxes_3d[rot_pos_90_indices])
    final_pred_boxes_3d[rot_pos_90_indices, 6] += pi_0_50

    # Rotate -90 degrees if difference between -pi/4 and -3/4 pi
    rot_neg_90_indices = np.logical_and(-pi_0_25 > ang_diff,
                                        ang_diff > -pi_0_75)
    final_pred_boxes_3d[rot_neg_90_indices] = \
        swap_boxes_3d_lw(final_pred_boxes_3d[rot_neg_90_indices])
    final_pred_boxes_3d[rot_neg_90_indices, 6] -= pi_0_50

    # Flip angles if abs difference if greater than or equal to 135
    # degrees
    swap_indices = np.abs(ang_diff) >= pi_0_75
    final_pred_boxes_3d[swap_indices, 6] += np.pi

    # Wrap to -pi, pi
    above_pi_indices = final_pred_boxes_3d[:, 6] > np.pi
    final_pred_boxes_3d[above_pi_indices, 6] -= two_pi

    
    # Find max class score index
    not_bkg_scores = final_pred_softmax[:, 1:]
    final_pred_types = np.argmax(not_bkg_scores, axis=1)

    # Take max class score (ignoring background)
    final_pred_scores = np.array([])
    for pred_idx in range(len(final_pred_boxes_3d)):
        all_class_scores = not_bkg_scores[pred_idx]
        max_class_score = all_class_scores[final_pred_types[pred_idx]]
        final_pred_scores = np.append(final_pred_scores, max_class_score)

    # Stack into prediction format
    predictions_and_scores = np.column_stack(
        [final_pred_boxes_3d,
            final_pred_scores,
            final_pred_types])

    return predictions_and_scores


def main():
    """main"""
    #initialize acl runtime 
    acl_resource = AclResource()
    acl_resource.init()

    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(suppress=True)
    
    model = Model("../model/avod_npu.om")
    # print(vars(model))
    dataloader = KittiDataloader()

    Path("../output/proposals_and_scores").mkdir(parents=True, exist_ok=True)
    Path("../output/final_predictions_and_scores").mkdir(parents=True, exist_ok=True)

    for name, input_list in dataloader:
        # 0: rpn_top_anchors (300, 6)
        # 1: rpn_top_objectness_softmax (300,)
        # 2: avod_nms_num_valid (?,)
        # 3: avod_top_classification_softmax (100, 2)
        # 4: avod_top_prediction_anchors (100, 6)
        # 5: avod_top_prediction_boxes_3d (100, 7)
        # 6: avod_top_prediction_boxes_4c (100, 10)
        # 7: avod_top_orientations (100,)
        out = model.execute(input_list)

        np.set_printoptions(threshold=np.inf)
        np.set_printoptions(suppress=True)

        rpn_top_anchors = out[0]
        softmax_scores = out[1]
        proposals_and_scores = get_rpn_proposals_and_scores(rpn_top_anchors, softmax_scores)
        
        num_valid = out[2]
        
        # boxes_3d from boxes_4c
        final_pred_boxes_3d = out[5][:num_valid]
        # Predicted orientation from layers
        final_pred_orientations = out[-1][:num_valid]
        # Append score and class index (object type)
        final_pred_softmax = out[3][:num_valid]

        predictions_and_scores = get_avod_predicted_boxes_3d_and_scores(final_pred_boxes_3d, final_pred_orientations, final_pred_softmax)
        
        np.savetxt(f"../output/proposals_and_scores/{name}.txt", proposals_and_scores, fmt='%.3f')
        np.savetxt(f"../output/final_predictions_and_scores/{name}.txt", predictions_and_scores, fmt='%.5f')

if __name__ == '__main__':   
    main()
