# Ascend NPU环境下的客户端编排测试文档

## 概述

本文档详细说明了 `client_orchestration_ascend.py` 文件的创建过程、代码修改内容以及在Ascend NPU环境下的使用方法。该客户端专门用于测试PD（Prefill-Decode）分离架构下的KV缓存感知调度算法。

## 文件创建和修改说明

### 1. 源文件复制
- **源文件**: `/home/brandonye/CodeSpace/vllm_ascend_main/vllm/tests/v1/kv_connector/nixl_integration/client_orchestration.py`
- **目标文件**: `/home/brandonye/CodeSpace/vllm_ascend_main/vllm-ascend/examples/disaggregated_prefill_v1/client_orchestration_ascend.py`

### 2. 主要修改内容

#### 2.1 文件头部注释修改
```python
# Ascend NPU环境下的DP协调器KV缓存长度统计功能集成测试客户端
# DP Coordinator KV Cache Length Statistics Integration Test Client for Ascend NPU
#
# 适配说明：
# - 直接连接到Ascend NPU上的prefiller和decoder实例
# - 不依赖proxy server，直接通过client调用进行测试
# - 支持Ascend NPU环境的特定配置
```

**修改目的**: 明确标识这是针对Ascend NPU环境的适配版本，说明其独立性和特定用途。

#### 2.2 PD分离架构实现修改（重要修正）
**原始代码**:
```python
decode_client = OpenAI(
    api_key=openai_api_key,
    base_url="http://127.0.0.1:8200/v1",  # Decode实例URL - 直接测试decode实例
    timeout=120.0
)
```

**修改后代码**:
```python
# 适配Ascend NPU环境：实现正确的PD分离架构
# 使用您提供的配置参数：--prefiller-hosts 7.150.11.60 --prefiller-port 20002 --decoder-hosts 7.150.11.60 --decoder-ports 20012

# Prefill客户端 - 负责预填充阶段
prefill_client = OpenAI(
    api_key=openai_api_key,
    base_url="http://7.150.11.60:20002/v1",  # Ascend NPU上的Prefill实例URL
    timeout=120.0
)

# Decode客户端 - 负责解码阶段
decode_client = OpenAI(
    api_key=openai_api_key,
    base_url="http://7.150.11.60:20012/v1",  # Ascend NPU上的Decode实例URL
    timeout=120.0
)
```

**修改目的**:
- **实现正确的PD分离架构流程**：prefill → decode 两阶段处理
- 连接到您的Ascend NPU prefill实例 (7.150.11.60:20002)
- 连接到您的Ascend NPU decode实例 (7.150.11.60:20012)
- 确保符合PD分离架构的设计原则

#### 2.3 测试标题和描述修改
**原始代码**:
```python
print("🧪 开始KV缓存感知调度算法真实负载测试")
print("🎯 测试目标: 验证KV缓存感知调度算法在真实负载下的性能表现")
```

**修改后代码**:
```python
print("🧪 开始Ascend NPU环境下的KV缓存感知调度算法真实负载测试")
print("🎯 测试目标: 验证KV缓存感知调度算法在Ascend NPU真实负载下的性能表现")
print("🚀 Ascend NPU配置: Decoder实例 7.150.11.60:20012")
```

**修改目的**: 
- 明确标识测试环境为Ascend NPU
- 显示具体的连接配置信息
- 便于调试和问题排查

#### 2.4 日志输出增强
在多个关键位置添加了Ascend NPU相关的标识：

```python
print(f"🔗 连接到Ascend NPU Decoder实例: http://7.150.11.60:20012/v1")
print(f"📤 将在{duration_seconds}秒内分散发送{len(requests_to_process)}个请求到Ascend NPU...")
print(f"🚀 Ascend NPU性能: 平均 {len(requests_to_process)/total_duration:.2f} QPS")
```

**修改目的**: 
- 提供清晰的运行状态反馈
- 便于监控Ascend NPU环境下的性能表现
- 帮助识别潜在的网络或配置问题

#### 2.3 KV缓存传输机制实现（关键修正）
**原始函数**:
```python
def process_single_request(client, model, prompt, request_id, max_tokens):
    # 直接发送到decode实例，无KV缓存传输
    response = client.completions.create(...)
```

**修改后函数**:
```python
def process_single_request_pd_separated(model, prompt, request_id, max_tokens):
    # 阶段1: 发送带kv_transfer_params的请求到Prefill实例
    prefill_request_data = {
        "model": model, "prompt": prompt, "max_tokens": 1,
        "kv_transfer_params": {
            "do_remote_decode": True,
            "do_remote_prefill": False,
            "remote_engine_id": None,
            "remote_block_ids": None,
            "remote_host": None,
            "remote_port": None,
            "aborted_request": [],
        }
    }
    prefill_response = await client.post(..., json=prefill_request_data)

    # 提取KV传输参数
    kv_transfer_params = prefill_response.get('kv_transfer_params', {})

    # 阶段2: 将KV传输参数传递给Decode实例
    decode_request_data = {
        "model": model, "prompt": prompt, "max_tokens": max_tokens,
        "kv_transfer_params": kv_transfer_params  # 关键：传递KV缓存信息
    }
    decode_response = await client.post(..., json=decode_request_data)
```

**修改目的**:
- **实现正确的KV缓存传输机制**：严格按照proxy server的实现
- 确保prefill阶段生成的KV缓存能被decode阶段正确获取
- 使用httpx直接发送请求以支持kv_transfer_params传递
- 提供KV传输成功率统计

#### 2.4 性能分析增强
**新增的性能指标**:
```python
print(f"⏱️  平均处理时间: {avg_duration:.2f}秒")
print(f"   🔄 平均Prefill时间: {avg_prefill_duration:.2f}秒")
print(f"   🔄 平均Decode时间: {avg_decode_duration:.2f}秒")
print(f"📊 PD分离架构性能分析:")
print(f"   - Prefill阶段占比: {(avg_prefill_duration/avg_duration)*100:.1f}%")
print(f"   - Decode阶段占比: {(avg_decode_duration/avg_duration)*100:.1f}%")
print(f"🔗 KV缓存传输统计:")
print(f"   - KV传输成功率: {kv_transfer_success_rate:.1f}% ({success_count}/{total_count})")
```

**修改目的**:
- 提供PD分离架构的详细性能分析
- **监控KV缓存传输成功率**：验证PD分离架构的核心功能
- 帮助识别性能瓶颈在哪个阶段
- 验证两阶段负载均衡的效果

#### 2.5 错误处理增强
**修改后的错误处理**:
```python
except Exception as e:
    print(f"❌ Ascend NPU环境下的DP协调器KV缓存长度统计功能集成测试过程中出错: {e}")
    print("请检查以下配置:")
    print("   - Ascend NPU上的prefill实例是否正在运行且可访问 (7.150.11.60:20002)")
    print("   - Ascend NPU上的decode实例是否正在运行且可访问 (7.150.11.60:20012)")
    print("   - PD分离架构是否已正确部署并且服务正常运行")
    print("   - 网络连接是否正常，防火墙设置是否正确")
    print("   - Ascend NPU环境配置是否完整")
```

**修改目的**:
- 提供针对PD分离架构的具体故障排查指导
- 分别检查prefill和decode实例的状态
- 帮助快速定位问题根源

## 运行方法

### 1. 环境准备
确保以下服务正在运行（**PD分离架构要求**）：
- **Ascend NPU上的Prefill实例** (7.150.11.60:20002) - 负责预填充阶段
- **Ascend NPU上的Decode实例** (7.150.11.60:20012) - 负责解码阶段
- 两个实例之间的KV缓存传输连接正常

### 2. 执行测试
```bash
cd /home/brandonye/CodeSpace/vllm_ascend_main/vllm-ascend/examples/disaggregated_prefill_v1/
python client_orchestration_ascend.py
```

### 3. 预期输出
测试将执行以下**PD分离架构流程**：
1. 连接到Ascend NPU上的Prefill实例 (20002) 和Decode实例 (20012)
2. 生成100个长尾分布的测试请求
3. 对每个请求执行两阶段处理：
   - **阶段1**: 发送到Prefill实例进行预填充
   - **阶段2**: 发送到Decode实例进行解码（KV缓存自动传输）
4. 在10秒内分散发送请求（平均10 QPS）
5. 收集和分析PD分离架构的性能数据
6. 监控KV缓存统计信息30秒

## 配置要求

### 网络配置
- 确保客户端可以访问 `7.150.11.60:20002` (Prefill实例)
- 确保客户端可以访问 `7.150.11.60:20012` (Decode实例)
- 检查防火墙设置，确保两个端口都开放
- 验证网络延迟在可接受范围内
- **重要**: 确保Prefill和Decode实例之间的KV缓存传输连接正常

### 服务配置
- **Prefill实例**必须正在运行并响应请求 (7.150.11.60:20002)
- **Decode实例**必须正在运行并响应请求 (7.150.11.60:20012)
- 两个实例应该已经加载了相同的模型
- 确保有足够的资源处理并发请求
- **PD分离架构要求**: KV缓存传输机制正常工作

### 环境依赖
- Python 3.8+
- OpenAI Python客户端库（用于模型信息获取）
- **httpx库**（用于支持kv_transfer_params的HTTP请求）
- 其他标准库（uuid, time, threading, asyncio等）

### 安装依赖
```bash
pip install openai httpx
```

## 监控和日志

### 关键日志文件
- `decode_dp.log`: Decoder引擎的详细日志
- `prefill_dp.log`: Prefiller引擎的日志（如果存在）

### 监控重点
- KV缓存长度统计信息的收集和更新
- 请求处理时间和吞吐量
- 系统资源使用情况
- 错误率和失败原因

## 故障排查

### 常见问题
1. **连接失败**: 检查IP地址和端口是否正确
2. **超时错误**: 增加timeout设置或检查网络延迟
3. **模型加载失败**: 确认服务端模型配置正确
4. **性能问题**: 监控系统资源使用情况

### 调试建议
1. 首先测试单个请求是否能正常处理
2. 逐步增加并发数量
3. 监控服务端日志以识别瓶颈
4. 使用网络工具验证连接性

## 性能基准

### 预期性能指标
- 平均响应时间: < 5秒（取决于请求复杂度）
- 吞吐量: 10+ QPS
- 成功率: > 95%
- KV缓存统计更新延迟: < 1秒

### 性能优化建议
1. 调整并发线程数量
2. 优化网络配置
3. 监控和调整服务端资源
4. 根据实际负载调整测试参数

## 快速开始

### 一键运行命令
```bash
cd /home/brandonye/CodeSpace/vllm_ascend_main/vllm-ascend/examples/disaggregated_prefill_v1/
python client_orchestration_ascend.py
```

### 预期运行流程
1. **连接验证** - 连接到Ascend NPU Decoder实例 (7.150.11.60:20012)
2. **请求生成** - 生成100个长尾分布测试请求
3. **负载测试** - 10秒内发送请求，平均10 QPS
4. **结果分析** - 统计性能数据和KV缓存信息
5. **监控阶段** - 持续监控30秒

### 成功运行的标志
- ✅ 成功连接到Ascend NPU Prefill和Decode实例
- ✅ 请求成功率 > 95%
- ✅ **KV缓存传输成功率 = 100%**（关键指标）
- ✅ 平均响应时间 < 5秒
- ✅ 显示详细的PD分离架构性能分析
- ✅ 每个请求都显示"KV缓存传输: 成功"

## 总结

`client_orchestration_ascend.py` 是专门为Ascend NPU环境定制的测试客户端，主要修改包括：

### 核心适配修改
1. **PD分离架构实现**:
   - 原始: 直接连接decode实例 `127.0.0.1:8200`
   - 修改后: 实现完整的两阶段流程
     - Prefill阶段: `7.150.11.60:20002`
     - Decode阶段: `7.150.11.60:20012`

2. **请求处理流程重构**:
   - 原始: `process_single_request()` - 单阶段处理
   - 修改后: `process_single_request_pd_separated()` - 两阶段处理

3. **性能分析增强**:
   - 分别统计Prefill和Decode阶段耗时
   - 计算各阶段时间占比
   - 提供PD分离架构专用性能指标

4. **环境标识**: 所有输出都明确标注"Ascend NPU"和"PD分离架构"

5. **错误处理**: 针对两个实例的具体故障排查指导

### 保持不变的功能
- 完整的KV缓存感知调度算法测试逻辑
- 长尾分布的真实负载生成
- 详细的统计分析和结果报告
- 实时日志监控功能

### 重要说明
该客户端现在**正确实现了PD分离架构**，确保：
- 每个请求都经过prefill → decode的完整流程
- KV缓存在两阶段间正确传输
- 能够准确测试PD分离架构的性能和稳定性
- 提供针对性的性能分析和问题诊断
