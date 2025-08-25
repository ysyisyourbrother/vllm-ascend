import math
import os
from contextlib import contextmanager
from enum import Enum
from typing import Any, Optional

import torch
from vllm.config import VllmConfig
from vllm.distributed import get_dp_group, get_ep_group, get_tp_group
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.logger import init_logger

import vllm_ascend.envs as envs
from vllm_ascend.platform import NPUPlatform

# 初始化logger
logger = init_logger("vllm")

class FusedMoEState(Enum):
    AllGather = 0
    All2All = 1
    MC2 = 2
    AllGatherEP = 3
    NaiveMulticast = 4
    All2AllSeq = 5


# TODO(zzzzwwjj): add soc_version to choose branch
def _get_fused_moe_state(ep_size: int, with_prefill: bool,
                         is_deepseek_v3_r1: bool):
    # 添加MoE状态决策日志
    if os.getenv('VLLM_INFERENCE_PATH_DEBUG', 'false').lower() == 'true':
        logger.info(f"【推理路径定位】开始MoE状态决策 - EP size: {ep_size}, with_prefill: {with_prefill}, is_deepseek_v3_r1: {is_deepseek_v3_r1}")

    # 记录环境变量状态
    if os.getenv('VLLM_INFERENCE_PATH_DEBUG', 'false').lower() == 'true':
        logger.info(f"【推理路径定位】环境变量检查 - VLLM_ENABLE_FUSED_EXPERTS_ALLGATHER_EP: {envs.VLLM_ENABLE_FUSED_EXPERTS_ALLGATHER_EP}")
        logger.info(f"【推理路径定位】环境变量检查 - VLLM_ASCEND_ENABLE_MOE_ALL2ALL_SEQ: {envs.VLLM_ASCEND_ENABLE_MOE_ALL2ALL_SEQ}")

    # the fusion operator torch_npu.npu_grouped_matmul_finalize_routing called by allgather ep
    # only supports deepseek v3/r1
    if (envs.VLLM_ENABLE_FUSED_EXPERTS_ALLGATHER_EP and ep_size > 1
            and is_deepseek_v3_r1):
        if os.getenv('VLLM_INFERENCE_PATH_DEBUG', 'false').lower() == 'true':
            logger.info(f"【推理路径定位】MoE状态决策结果: AllGatherEP - 满足AllGatherEP条件")
        return FusedMoEState.AllGatherEP
    elif ep_size == 1:
        if with_prefill:
            if os.getenv('VLLM_INFERENCE_PATH_DEBUG', 'false').lower() == 'true':
                logger.info(f"【推理路径定位】MoE状态决策结果: NaiveMulticast - EP size=1且有prefill")
            return FusedMoEState.NaiveMulticast
        else:
            if os.getenv('VLLM_INFERENCE_PATH_DEBUG', 'false').lower() == 'true':
                logger.info(f"【推理路径定位】MoE状态决策结果: AllGather - EP size=1且无prefill")
            return FusedMoEState.AllGather
    elif envs.VLLM_ASCEND_ENABLE_MOE_ALL2ALL_SEQ:
        # MC2 Dispatch/Combine performs better than alltoall_seq in decoding stage.
        if ep_size < 16 or with_prefill:
            if os.getenv('VLLM_INFERENCE_PATH_DEBUG', 'false').lower() == 'true':
                logger.info(f"【推理路径定位】MoE状态决策结果: All2AllSeq - EP size<16或有prefill")
            return FusedMoEState.All2AllSeq
        else:
            if os.getenv('VLLM_INFERENCE_PATH_DEBUG', 'false').lower() == 'true':
                logger.info(f"【推理路径定位】MoE状态决策结果: MC2 - EP size>=16且无prefill (通过All2AllSeq分支)")
            return FusedMoEState.MC2
    # NOTE: mc2 need ep_size >= 16 & all2all can't use in torchair graph.
    elif ep_size < 16 or with_prefill:
        if os.getenv('VLLM_INFERENCE_PATH_DEBUG', 'false').lower() == 'true':
            logger.info(f"【推理路径定位】MoE状态决策结果: All2All - EP size<16或有prefill")
        return FusedMoEState.All2All
    else:
        if os.getenv('VLLM_INFERENCE_PATH_DEBUG', 'false').lower() == 'true':
            logger.info(f"【推理路径定位】MoE状态决策结果: MC2 - EP size>=16且无prefill (默认分支)")
        return FusedMoEState.MC2


@contextmanager
def set_ascend_forward_context(
    attn_metadata: Any,
    vllm_config: VllmConfig,
    virtual_engine: int = 0,
    num_tokens: Optional[int] = None,
    num_tokens_across_dp: Optional[torch.Tensor] = None,
    with_prefill: bool = True,
    in_profile_run: bool = False,
    num_actual_tokens: Optional[int] = None,
):
    """A context manager that stores the current forward context,
    can be attention metadata, etc.
    We add some additional param into forward_context.
    """
    with set_forward_context(attn_metadata,
                             vllm_config,
                             virtual_engine=virtual_engine,
                             num_tokens=num_tokens,
                             num_tokens_across_dp=num_tokens_across_dp):
        forward_context = get_forward_context()
        forward_context.with_prefill = with_prefill

        # 添加EP配置检查日志
        if os.getenv('VLLM_INFERENCE_PATH_DEBUG', 'false').lower() == 'true':
            logger.info(f"【推理路径定位】Forward Context设置开始")
            logger.info(f"【推理路径定位】vLLM并行配置检查 - enable_expert_parallel: {vllm_config.parallel_config.enable_expert_parallel}")

        ep_size = (get_ep_group().world_size if
                   vllm_config.parallel_config.enable_expert_parallel else 1)

        if os.getenv('VLLM_INFERENCE_PATH_DEBUG', 'false').lower() == 'true':
            logger.info(f"【推理路径定位】EP组配置 - EP group world_size: {get_ep_group().world_size}")
            logger.info(f"【推理路径定位】EP组配置 - 最终EP size: {ep_size}")

        # 检查模型配置
        model_config = vllm_config.model_config.hf_config
        if os.getenv('VLLM_INFERENCE_PATH_DEBUG', 'false').lower() == 'true':
            logger.info(f"【推理路径定位】模型配置检查 - 模型类型: {type(model_config).__name__}")
            logger.info(f"【推理路径定位】模型配置检查 - 是否有n_routed_experts: {hasattr(model_config, 'n_routed_experts')}")
        if hasattr(model_config, 'n_routed_experts'):
            if os.getenv('VLLM_INFERENCE_PATH_DEBUG', 'false').lower() == 'true':
                logger.info(f"【推理路径定位】模型配置检查 - n_routed_experts: {model_config.n_routed_experts}")

        is_deepseek_v3_r1 = hasattr(
            vllm_config.model_config.hf_config, 'n_routed_experts'
        ) and vllm_config.model_config.hf_config.n_routed_experts == 256

        if os.getenv('VLLM_INFERENCE_PATH_DEBUG', 'false').lower() == 'true':
            logger.info(f"【推理路径定位】模型类型判断 - is_deepseek_v3_r1: {is_deepseek_v3_r1}")

        fused_moe_state = _get_fused_moe_state(ep_size, with_prefill,
                                               is_deepseek_v3_r1)
        forward_context.fused_moe_state = fused_moe_state

        if os.getenv('VLLM_INFERENCE_PATH_DEBUG', 'false').lower() == 'true':
            logger.info(f"【推理路径定位】Forward Context设置完成 - 最终MoE状态: {fused_moe_state}")
        forward_context.in_profile_run = in_profile_run

        # NOTE: This cannot be set using set_forward_context
        # due to multiple warmups before actual capturing
        forward_context.capturing = False

        if num_tokens is None and attn_metadata is not None:
            num_tokens = attn_metadata.num_actual_tokens

        dp_world_size = get_dp_group().world_size
        if dp_world_size > 1 and forward_context.dp_metadata is not None:
            max_tokens_across_dp = forward_context.dp_metadata.max_tokens_across_dp_cpu.item(
            )
        else:
            max_tokens_across_dp = num_tokens

        forward_context.max_tokens_across_dp = max_tokens_across_dp

        if num_tokens is not None:
            if num_actual_tokens is None:
                num_actual_tokens = num_tokens
            tp_world_size = get_tp_group().world_size
            # NOTE: token num which need to pad to when mc2
            forward_context.padded_num_tokens = math.ceil(
                max_tokens_across_dp / tp_world_size) * tp_world_size

            mc2_mask = torch.zeros(forward_context.padded_num_tokens,
                                   dtype=torch.bool,
                                   device=NPUPlatform.device_type)
            mc2_mask[:num_actual_tokens] = True
            forward_context.mc2_mask = mc2_mask

        try:
            yield
        finally:
            pass
