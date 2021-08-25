# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



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
from multiprocessing import Process

import subprocess
from pathlib import Path

import utils.utils as utils

ROOT_DIR = Path(__file__).resolve().parent.parent
dataset_dir = ROOT_DIR / 'data'
imgs_dir = dataset_dir / 'image_2'
score_threshold = 0.1

gt_labels_dir = dataset_dir / 'label_2'
kitti_predictions_3d_dir = ROOT_DIR / 'kitti_native_eval' / str(score_threshold)

script_dir = ROOT_DIR / 'kitti_native_eval'
results_dir = ROOT_DIR / 'output'

def run_kitti_native_script():
    
    eval_script = script_dir / 'run_eval.sh'
    
    subprocess.call([str(eval_script), 
                     str(script_dir), # $1
                     str(score_threshold), # $2
                     str(gt_labels_dir), # $3
                     str(kitti_predictions_3d_dir), # $4
                     str(results_dir)]) # $5

def run_kitti_native_script_with_05_iou():

    eval_script = script_dir / 'run_eval_05_iou.sh'
    
    subprocess.call([str(eval_script), 
                     str(script_dir), # $1
                     str(score_threshold), # $2
                     str(gt_labels_dir), # $3
                     str(kitti_predictions_3d_dir), # $4
                     str(results_dir)]) # $5

def main():
    # Convert output in Kitti format
    sample_names = [name for name, _ in map(lambda x : x.split('.'), os.listdir(imgs_dir))]
    utils.save_predictions_in_kitti_format(sample_names, score_threshold)
    
    # Create a separate processes to run the native evaluation
    native_eval_proc = Process(target=run_kitti_native_script, args=())
    native_eval_proc_05_iou = Process(target=run_kitti_native_script_with_05_iou, args=())
    
    # Don't call join on this cuz we do not want to block
    # this will cause one zombie process - should be fixed later.
    native_eval_proc.start()
    native_eval_proc_05_iou.start()
    
    native_eval_proc.join()
    native_eval_proc_05_iou.join()

if __name__ == '__main__':
    main()
