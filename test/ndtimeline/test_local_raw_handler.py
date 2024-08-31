################################################################################
#
# Copyright 2023 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################

import os
from vescale.ndtimeline.world_info import WorldInfo
from vescale.ndtimeline.handlers import LocalRawNDHandler
from vescale.ndtimeline.variables import LOCAL_LOGGING_PATH


def test_basic_usage():
    ### run_id: represents an identifier for a specific run or instance of the handler. It is set to 0 in this case.
    ### chunk_sz: stands for "chunk size" and indicates the size of data chunks that the handler will process. It is set to 10 in this case.
    ###           chunk_sz 控制日志文件的最大大小，当日志文件超过这个大小时，会自动进行日志轮转。
    ###           chunk_sz 参数用于控制日志文件的最大大小（以字节为单位）。在 LocalRawNDHandler 类中，
    ###           它被传递给 RotatingFileHandler 的 maxBytes 参数。如果日志文件的大小超过 chunk_sz，则会创建一个新的日志文件，
    ###           并根据 backup_cnt 参数保留一定数量的备份日志文件。
    ### backup_cnt: represents the number of backups to maintain. It is set to 3 in this case.
    h = LocalRawNDHandler(run_id=0, chunk_sz=10, backup_cnt=3)

    file_name = "timeline_run0_raw.log"
    h("test_metric", 1.0, [1.0], [1.0], [{}], range(0, 1), WorldInfo(0, 0), {})
    assert os.path.exists(os.path.join(LOCAL_LOGGING_PATH, file_name))
    for _ in range(4):
        h("test_metric", 1.0, [1.0], [1.0], [{}], range(0, 1), WorldInfo(0, 0), {})
    h("test_metric2", 2.0, [1.0], [1.0], [{}], range(0, 1), WorldInfo(0, 0), {})
    assert os.path.exists(os.path.join(LOCAL_LOGGING_PATH, file_name + ".2"))
    assert not os.path.exists(os.path.join(LOCAL_LOGGING_PATH, file_name + ".4"))
    os.remove(os.path.join(LOCAL_LOGGING_PATH, file_name))
    for i in range(1, 4):
        os.remove(os.path.join(LOCAL_LOGGING_PATH, file_name + "." + str(i)))
    assert not os.path.exists(os.path.join(LOCAL_LOGGING_PATH, file_name + ".2"))
