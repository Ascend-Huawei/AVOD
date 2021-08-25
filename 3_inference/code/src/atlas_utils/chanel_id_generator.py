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


class _ChannelIdGenerator(object):
    """Generate global unique id number, single instance mode class"""
    _instance_lock = threading.Lock()
    channel_id = 0

    def __init__(self):
        pass

    def __new__(cls, *args, **kwargs):
        if not hasattr(_ChannelIdGenerator, "_instance"):
            with _ChannelIdGenerator._instance_lock:
                if not hasattr(_ChannelIdGenerator, "_instance"):
                    _ChannelIdGenerator._instance = object.__new__(
                        cls, *args, **kwargs)
        return _ChannelIdGenerator._instance

    def generator_channel_id(self):
        """Generate global unique id number
        The id number is increase
        """
        curren_channel_id = 0
        with _ChannelIdGenerator._instance_lock:
            curren_channel_id = _ChannelIdGenerator.channel_id
            _ChannelIdGenerator.channel_id += 1

        return curren_channel_id


def gen_unique_channel_id():
    """Interface of generate global unique id number"""
    generator = _ChannelIdGenerator()
    return generator.generator_channel_id()
