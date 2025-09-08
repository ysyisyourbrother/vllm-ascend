#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm/vllm/worker/gpu_model_runner.py
#

import torch
from vllm.config import VllmConfig

from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


class NPUTorchairModelRunner(NPUModelRunner):

    # Brandon: 传递了worker进去
    def __init__(self, vllm_config: VllmConfig, device: torch.device, worker=None):
        super().__init__(vllm_config, device, worker=worker)
