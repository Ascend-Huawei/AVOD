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


import sys
import os

import acl


_ACL_DEBUG = 0
_ACL_INFO = 1
_ACL_WARNING = 2
_ACL_ERROR = 3


def log_error(*log_msg):
    """Recode error level log to file
    Args:
        *log_msg: format string and args list
    """
    log_str = [str(i) for i in log_msg]
    log_str = "".join(log_str)

    print(log_str)

    caller_frame = sys._getframe().f_back
    # caller file
    filename = caller_frame.f_code.co_filename
    # caller line no
    line_no = caller_frame.f_lineno
    # caller function
    func_name = caller_frame.f_code.co_name

    message = "[" + filename + ":" + str(line_no) + \
              " " + func_name + "]" + log_str
    acl.app_log(_ACL_ERROR, message)


def log_warning(*log_msg):
    """Recode warning level log to file
    Args:
        *log_msg: format string and args list
    """
    log_str = [str(i) for i in log_msg]
    log_str = "".join(log_str)
    caller_frame = sys._getframe().f_back
    # caller file
    filename = caller_frame.f_code.co_filename
    # caller line no
    line_no = caller_frame.f_lineno
    # caller function
    func_name = caller_frame.f_code.co_name

    message = "[" + filename + ":" + str(line_no) + \
              " " + func_name + "]" + log_str
    acl.app_log(_ACL_WARNING, message)


def log_info(*log_msg):
    """Recode info level log to file
    Args:
        *log_msg: format string and args list
    """
    log_str = [str(i) for i in log_msg]
    log_str = "".join(log_str)
    print(log_str)
    caller_frame = sys._getframe().f_back
    # caller file
    filename = caller_frame.f_code.co_filename
    # caller line no
    line_no = caller_frame.f_lineno
    # caller function
    func_name = caller_frame.f_code.co_name

    message = "[" + filename + ":" + str(line_no) + \
              " " + func_name + "]" + log_str
    acl.app_log(_ACL_INFO, message)


def log_debug(*log_msg):
    """Recode debug level log to file
    Args:
        *log_msg: format string and args list
    """
    log_str = [str(i) for i in log_msg]
    log_str = "".join(log_str)
    caller_frame = sys._getframe().f_back
    # caller file
    filename = caller_frame.f_code.co_filename
    # caller line no
    line_no = caller_frame.f_lineno
    # caller function
    func_name = caller_frame.f_code.co_name

    message = "[" + filename + ":" + str(line_no) + \
              " " + func_name + "]" + log_str

    acl.app_log(_ACL_DEBUG, message)
