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

REGISTER = 0
UNREGISTER = 1


class _ResourceList(object):
    """Acl resources of current application
    This class provide register inferace of acl resource, when application
    exit, all register resource will release befor acl.rt.reset_device to
    avoid program abnormal 
    """
    _instance_lock = threading.Lock()

    def __init__(self):
        self.resources = []

    def __new__(cls, *args, **kwargs):
        if not hasattr(_ResourceList, "_instance"):
            with _ResourceList._instance_lock:
                if not hasattr(_ResourceList, "_instance"):
                    _ResourceList._instance = object.__new__(
                        cls, *args, **kwargs)
        return _ResourceList._instance

    def register(self, resource):
        """Resource register interface
        Args:
            resource: object with acl resource, the object must be has
                      method destroy()
        """
        item = {"resource": resource, "status": REGISTER}
        self.resources.append(item)

    def unregister(self, resource):
        """Resource unregister interface
        If registered resource release by self and no need _ResourceList 
        release, the resource object should unregister self
        Args:
            resource: registered resource
        """
        for item in self.resources:
            if resource == item["resource"]:
                item["status"] = UNREGISTER

    def destroy(self):
        """Destroy all register resource"""
        for item in self.resources:
            if item["status"] == REGISTER:
                item["resource"].destroy()
                item["status"] = UNREGISTER


resource_list = _ResourceList()
