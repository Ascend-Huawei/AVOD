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


# !/usr/bin/env python
# -*- coding:utf-8 -*-
#
from ctypes import *
import os
import time
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from atlas_utils.lib.atlasutil_so import libatlas
import atlas_utils.constants as const
from atlas_utils.acl_image import AclImage
from atlas_utils.acl_logger import log_error, log_info

CAMERA_OK = 0
CAMERA_ERROR = 1

CAMERA_CLOSED = 0
CAMERA_OPENED = 1


class CameraOutputC(Structure):
    """Ctypes parameter object for frame data"""
    _fields_ = [
        ('size', c_int),
        ('data', POINTER(c_ubyte))
    ]


class Camera(object):
    """Atlas200dk board camera access class"""
    def __init__(self, camera_id, fps=20, size=(1280, 720)):
        """Create camera instance
        Args:
            camera_id: camera slot
            fps: frame per second
            size: frame resolution
        """
        self._id = camera_id
        self._fps = fps
        self._width = size[0]
        self._height = size[1]
        self._size = int(self._width * self._height * 3 / 2)
        self._status = CAMERA_CLOSED
        if CAMERA_OK == self._open():
            self._status = CAMERA_OPENED
        else:
            log_error("Open camera %d failed" % (camera_id))

    def _open(self):
        ret = libatlas.OpenCameraEx(self._id, self._fps,
                                    self._width, self._height)
        if (ret != CAMERA_OK):
            log_error("Open camera %d failed ,ret = %d" % (self._id, ret))
            return CAMERA_ERROR
        self._status = CAMERA_OPENED
        return CAMERA_OK

    def is_opened(self):
        """Camera is opened or not"""
        return (self._status == CAMERA_OPENED)

    def read(self):
        """Read frame from camera"""
        frame_data = CameraOutputC()
        ret = libatlas.ReadCameraFrame(self._id, byref(frame_data))
        if (ret != CAMERA_OK):
            log_error("Read camera %d failed" % (self._id))
            return None

        return AclImage(
            addressof(frame_data.data.contents),                
            self._width,
            self._height,
            self._size,
            const.MEMORY_DVPP)

    def close(self):
        """Close camera"""
        log_info("Close camera ", self._id)
        libatlas.CloseCameraEx(self._id)

    def __del__(self):
        self.close()


if __name__ == "__main__":
    cap = Camera(camera_id=0, fps=20, size=(1280, 720))

    start = time.time()
    for i in range(0, 100):
        image = cap.read()
    print("Read 100 frame exhaust ", time.time() - start)
