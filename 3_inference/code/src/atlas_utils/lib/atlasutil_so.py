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


import threading
import ctypes
import os
import platform

import acl

from atlas_utils.constants import ACL_HOST, ACL_DEVICE


def _load_lib_atlasutil():
    run_mode, ret = acl.rt.get_run_mode()

    lib = None
    if run_mode == ACL_DEVICE:
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        so_path = os.path.join(cur_dir, 'atlas200dk/libatlasutil.so')
        lib=ctypes.CDLL(so_path)

    return lib


class _AtlasutilLib(object):
    _instance_lock=threading.Lock()
    lib=_load_lib_atlasutil()

    def __init__(self):
        pass

    def __new__(cls, *args, **kwargs):
        if not hasattr(_AtlasutilLib, "_instance"):
            with _AtlasutilLib._instance_lock:
                if not hasattr(_AtlasutilLib, "_instance"):
                    _AtlasutilLib._instance=object.__new__(
                        cls, *args, **kwargs)
        return _AtlasutilLib._instance

libatlas=_AtlasutilLib.lib
