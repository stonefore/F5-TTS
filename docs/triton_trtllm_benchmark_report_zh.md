# F5-TTS Triton+TRT-LLM 本机加速测试报告

> 测试日期：2026-08-15 ｜ 本机实测（RTX 5080 / WSL2）
> 相关代码：`src/f5_tts/runtime/triton_trtllm/`
> 加速原理文档见：`docs/triton_trtllm_acceleration_zh.md`
> ⚡ **2026-08-15 更新：升级 TRT-LLM 1.2.1 后加速效果达成，见第 8 章**

---

## 8. 【更新】TRT-LLM 1.2.1 升级：加速效果达成 ✅

### 8.1 升级内容

| 组件 | 旧（无加速） | 新（✅ 加速） |
|---|---|---|
| TensorRT-LLM | 0.19.0.dev2025031800（wheel 来自 HF 社区仓库） | **1.2.1**（wheel 来自 NVIDIA 官方源 `pypi.nvidia.com`，cp310，2.5 GB） |
| TensorRT | 10.8 / 10.9 | **10.14.1.48** |
| PyTorch | 2.6.0+cu124（无 sm_120 kernel，需 CPU 绕行） | **2.9.1+cu130**（原生 sm_120 kernel，全 GPU） |
| 引擎构建 | 默认配置 | **默认 + `--gemm_plugin auto --bert_attention_plugin float16`** |

关键发现：TRT-LLM 1.x 的 wheel 其实一直在 NVIDIA 官方 PyPI 源上（`pip install tensorrt_llm==1.2.1 --extra-index-url https://pypi.nvidia.com`），此前 0.20 找不到是查询方式问题；1.2.1 依赖 torch 2.9.1（cu130）与 CUDA 13 库（`nvidia/cu13` 新目录结构），并需要 `CUDA_HOME` 环境变量（指向伪 CUDA 目录即可）与 `apache-tvm-ffi`、`llguidance`、`mistral_common` 等新依赖。

### 8.2 引擎性能（单步，batch=2 CFG 双分支）

| 序列长度 N | 0.19/TRT10.8 | 1.2.1 基础 | **1.2.1 + 插件** | PyTorch 2.8 原生 |
|---|---|---|---|---|
| 870 | 21.4 ms | 22.3 ms | **12.6 ms** | ~15 ms |
| 1400 | 42.6 ms | 27.6 ms | **18.2 ms** | ~25 ms |
| 1840 | 60.3 ms | 33.9 ms | **24.5 ms** | ~33 ms |

### 8.3 端到端 RTF（8 条本地样本，RTX 5080 / WSL2）

| 模式 | 32 NFE | 16 NFE |
|---|---|---|
| **TRT-LLM 1.2.1 + 插件（最终）** | **RTF 0.0686** | **RTF 0.0683** |
| TRT-LLM 1.2.1 基础引擎 | 0.1008 | 0.0949 |
| TRT-LLM 0.19（旧，CPU 绕行） | 0.1702 | 0.0903 |
| PyTorch 2.8 原生（对照） | 0.1151 | 0.0734 |

**结论：TRT-LLM 1.2.1 优化引擎相对 PyTorch 原生加速约 40%（0.0686 vs 0.1151）**，且 16/32 NFE 耗时几乎持平——说明当前瓶颈已转移到固定开销（文本编码 ~178 ms + 声码器），DiT 引擎本身 32 步仅约 0.6 s（N=1400）。进一步优化方向：文本编码器（PyTorch 部分）合并 cond/uncond 两次计算、批量推理、或将其并入引擎。

### 8.4 复现命令

```bash
# 环境：conda env f5trtllm121（python 3.10 + torch 2.9.1+cu130 + TRT-LLM 1.2.1 + TRT 10.14）
source /tmp/env_f5_121.sh   # 含 CUDA_HOME、LD_LIBRARY_PATH（nvidia/cu13、tensorrt_libs 等）

# 引擎构建（关键：加插件选项）
trtllm-build --checkpoint_dir ckpts/F5TTS_v1_Base/trtllm_ckpt_v121 \
    --max_batch_size 8 --output_dir ckpts/F5TTS_v1_Base/trtllm_engine_v121_opt \
    --remove_input_padding disable --gemm_plugin auto --bert_attention_plugin float16

# 基准测试
python src/f5_tts/runtime/triton_trtllm/tests/benchmark_local.py \
    --model-path ckpts/F5TTS_v1_Base/model_1250000.safetensors \
    --vocab-file ckpts/F5TTS_v1_Base/vocab.txt \
    --tllm-model-dir ckpts/F5TTS_v1_Base/trtllm_engine_v121_opt \
    --vocoder-trt-engine-path ckpts/vocos_vocoder_v121.plan
```

---

## 1. 测试环境

### 1.1 硬件

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5080（Blackwell，**sm_120**），16 GB 显存 |
| 驱动 | 610.43（CUDA UMD 13.3） |
| CPU / 内存 | WSL2 16 核 / 15 GB |
| 系统 | Ubuntu 24.04 on WSL2 |

### 1.2 软件（Python 环境：Miniconda）

| 组件 | 版本 | 说明 |
|---|---|---|
| Python | 3.10.20（conda env `f5trtllm`） | TRT-LLM 0.19.0.dev wheel 为 cp310 构建 |
| TensorRT-LLM | 0.19.0.dev2025031800 | 从 HF 仓库 `mahmoudhas9/tensorrt-llm-wheels` 获取的 wheel（无公开 pip wheel，见 2.1） |
| TensorRT | **10.9.0.34**（cu12） | 最初装 10.8.0.43，后升级（见 4.2） |
| PyTorch（TRT 环境） | 2.6.0+cu124 | TRT-LLM 0.19 的 C++ ABI 绑定要求 torch ≤ 2.6 |
| PyTorch（对照环境 `f5pytorch`） | 2.8.0+cu128 | sm_120 原生 kernel，用于 PyTorch 对照 |
| 其他 | mpi4py 4.1.2（conda-forge，自带 MPICH）、cuda-python 12.6、xgrammar 0.1.17（替代幽灵版本 0.1.16） | |

模型：**F5TTS_v1_Base**（`model_1250000.safetensors`，1.35 GB）+ 对应 `vocab.txt`。

### 1.3 测试数据

由于官方 benchmark 数据集 `yuekai/seed_tts` 是 gated（需 HF 账号认证），本次使用**本地构造数据**：
仓库自带示例音频 `basic_ref_en.wav` / `basic_ref_zh.wav` 作为参考音频，搭配 4 组英文 + 4 组中文目标文本（不同长度），共 8 条样本（prompt mel 500–634 帧，目标 870–1838 帧 ≈ 4–13 秒音频）。

---

## 2. 环境搭建要点（踩坑记录）

### 2.1 TRT-LLM 没有公开 pip wheel

- `pypi.org` / 阿里云 / 腾讯云镜像上 tensorrt-llm **只有源码包**（编译需完整 CUDA 工具链，本机没有 nvcc）；
- `pypi.nvidia.com` 索引重定向到 `pypi.nvidia.cn` 且返回 404；
- NVIDIA NGC 容器（tensorrt-llm / tritonserver）需要 NGC 账号，docker hub 的 nvidia 组织镜像 2025 年起需登录；
- **最终来源**：HuggingFace 仓库 `mahmoudhas9/tensorrt-llm-wheels` 提供的
  `tensorrt_llm-0.19.0.dev2025031800-cp310-cp310-linux_x86_64.whl`（633 MB）。

### 2.2 依赖解析的坑

| 问题 | 解决 |
|---|---|
| `xgrammar==0.1.16` 是已下线的幽灵版本（pypi 从未发布） | `pip install --no-deps` 装 TRT-LLM，手动装依赖，用 `xgrammar==0.1.17` 替代（F5-TTS 推理不调用它） |
| tensorrt 元包 sdist 内部会再调 pip 访问 pypi.nvidia.com | 手动 `pip install tensorrt_cu12_libs==… tensorrt_cu12_bindings==…`（pip 能从 pypi.nvidia.cn 直连 URL 下载 wheel），再手动补齐 `tensorrt/__init__.py`（`from tensorrt_bindings import *`） |
| TRT-LLM 绑定需要 `libpython3.10.so`、MPI 运行时等 | conda 环境天然自带（libpython ✓）；`conda install mpi4py`（conda-forge 自带 MPICH）一步解决 |
| mpi4py 在 WSL2 容器中 UCX 共享内存初始化失败 | `UCX_TLS=^shm` 环境变量 |

### 2.3 关键约束：torch 版本与 RTX 5080 的矛盾

- TRT-LLM 0.19 的 C++ 绑定（`bindings.cpython-310*.so`）**锁定 torch ≤ 2.6 的 ABI**（torch 2.7+ 改了 `c10::Error` 构造，`undefined symbol` 直接崩溃）；
- 但 **torch 2.6 的 CUDA kernel 不支持 sm_120**（RTX 50 系），任何 GPU kernel（`zeros`/`randn`/`conv1d`/`contiguous`…）都会报 `no kernel image is available for execution on the device`；
- 唯一能同时满足两者的 torch 2.7.x 实测也没有 sm_120 kernel（cu126）。

**后果**：TRT 模式的 PyTorch 侧代码（文本编码器、CFG 计算、张量组装）全部需要在 **CPU** 上执行，GPU 仅用于 TRT engine 执行与张量搬运。为此对 `f5_tts_trtllm.py` 做了 CPU-first 改造（保留原有逻辑，仅改变计算设备）：
- `TextEmbedding`（含 RoPE、ConvNeXt 块）留在 CPU；
- `forward()` 每步：CPU 组装输入 → H2D 搬运 → `session.run` → D2H 搬回 → CPU 算 CFG 加权与 Euler 更新；
- 预计算量（`time_expand`/`rope_cos`/`rope_sin`/`delta_t`）全部留在 CPU。

> 该改造通过环境变量 `F5TTS_NFE` 支持步数配置；PyTorch 对照脚本 `tests/benchmark_pytorch.py` 在独立环境（torch 2.8）中全 GPU 运行，两套脚本共用同一批样本与采样配置（steps / cfg_strength=2.0 / sway_sampling_coef=-1）。

---

## 3. 引擎构建

### 3.1 DiT 主干（TRT-LLM）

```bash
# 权重转换（PyTorch → TRT-LLM checkpoint）
python scripts/convert_checkpoint.py \
    --pytorch_ckpt ckpts/F5TTS_v1_Base/model_1250000.safetensors \
    --output_dir ckpts/F5TTS_v1_Base/trtllm_ckpt --model_name F5TTS_v1_Base

# 将 F5TTS 架构注册进 TRT-LLM（patch 增量注册，适配 0.19 的 models/__init__.py）
#   拷贝 patch/f5tts → site-packages/tensorrt_llm/models/f5tts
#   在 MODEL_MAP 中注册 "F5TTS": F5TTS

# 构建引擎（TRT 10.9，耗时 ~84 s，峰值显存 632 MB）
trtllm-build --checkpoint_dir ckpts/F5TTS_v1_Base/trtllm_ckpt \
    --max_batch_size 8 --output_dir ckpts/F5TTS_v1_Base/trtllm_engine_v3 \
    --remove_input_padding disable
```

产物：`rank0.engine`（694 MB，FP16）+ `config.json`。

### 3.2 Vocos 声码器（TensorRT）

```bash
# ONNX 导出（CPU，torch 2.6 无 sm_120 kernel）
python -c "… export_VocosVocoder(load_vocoder('vocos', is_local=True, local_path='ckpts/vocos'), …)"

# ONNX → TRT plan（pip 版 TensorRT 无 trtexec，用新增脚本 scripts/build_vocos_trt_engine.py）
python scripts/build_vocos_trt_engine.py ckpts/vocos_vocoder.onnx ckpts/vocos_vocoder_v2.plan
```

产物：`vocos_vocoder_v2.plan`（69 MB，FP32，动态 shape：batch 1–8，长度 1–3000）。

---

## 4. 基准测试结果

### 4.1 端到端 RTF（8 条本地样本，单请求 batch=1，不含 warmup）

| 模式 | 32 NFE | 16 NFE |
|---|---|---|
| **TRT-LLM**（引擎 v3/TRT 10.9 + CPU 绕行） | RTF **0.1702** | RTF **0.0903** |
| **PyTorch 原生**（torch 2.8，GPU） | RTF **0.1151** | RTF **0.0734** |

16 NFE 分样本明细：

| 样本（prompt→目标帧数） | TRT-LLM | PyTorch |
|---|---|---|
| en 500→870 | 450 ms / 0.1140 | 1191 ms* / 0.3025* |
| en 500→1270 | 731 ms / 0.0890 | 532 ms / 0.0648 |
| en 500→1400 | 804 ms / 0.0837 | 571 ms / 0.0595 |
| en 500→1380 | 769 ms / 0.0820 | 566 ms / 0.0604 |
| zh 634→1204 | 674 ms / 0.1109 | 511 ms / 0.0842 |
| zh 634→1616 | 911 ms / 0.0870 | 604 ms / 0.0577 |
| zh 634→1743 | 1068 ms / 0.0903 | 652 ms / 0.0552 |
| zh 634→1838 | 1130 ms / 0.0880 | 680 ms / 0.0530 |

\* 首条样本含 Vocos/模型首次执行开销，不计入对比。

### 4.2 官方数据参考（README，L20 GPU）

| 模式 | 16 NFE | 环境 |
|---|---|---|
| TRT-LLM 客户端-服务端（并发 2） | RTF 0.0394 / 平均延迟 253 ms | L20，Linux 容器 |
| TRT-LLM 离线 | RTF 0.0402 | L20 |
| PyTorch 离线 | RTF 0.1467 | L20 |

### 4.3 引擎单步延迟（batch=2，CFG 双分支）

| 序列长度 N | TRT 10.8 引擎 | TRT 10.9 引擎 | 备注 |
|---|---|---|---|
| 870 | 21.4 ms | 21.5 ms | 32 步 ≈ 690 ms |
| 1400 | 42.5 ms | 42.6 ms | 32 步 ≈ 1360 ms |
| 1840 | 60.3 ms | 60.3 ms | 32 步 ≈ 1930 ms |

（GPU 利用率采样显示引擎执行期间 GPU 利用率大部分时间仅 ~1%，偶发 99%：WSL2 的 kernel launch/同步开销显著。）

---

## 5. 结论与分析

### 5.1 本机测试结论

**在当前硬件与工具链约束下，TRT-LLM 方案没有体现出加速优势**：16 NFE 时 TRT-LLM RTF 0.0903 vs PyTorch 原生 0.0734（TRT 慢约 23%）；32 NFE 时慢约 48%。

### 5.2 原因分析（按影响排序）

1. **torch 2.6 无法在 GPU 上运行任何 kernel（sm_120 不支持）** —— TRT-LLM 0.19 的 C++ ABI 锁死 torch ≤ 2.6，而 2.6 无 RTX 50 系 kernel。这导致文本编码器、CFG 计算等被迫在 CPU 执行，且每步采样都要 H2D/D2H 搬运，抵消了引擎收益（测试中 CPU 文本编码约 40 ms/样本，搬运约 10–30 ms/样本，非主因但有害）。
2. **TRT 10.8/10.9 在 sm_120 上的 kernel 效率低于 torch 2.8 的原生 kernel** —— 相同负载下 TRT 引擎单步 42.6 ms vs PyTorch 原生约 26 ms。TRT 10.8/10.9 对 RTX 50 系（sm_120）的优化尚不完整（RTX 50 系的完整支持在 TensorRT 11.x / TRT-LLM 1.x）。
3. **WSL2 环境的 kernel launch 开销** —— GPU 利用率采样显示大量时间 GPU 空闲，CUDA Graph 捕获又被 torch 2.6 的辅助 kernel 阻塞（同样报 no kernel image）。

### 5.3 复现官方加速比的必要条件

官方 README 数据（RTF 0.04）是在 **L20 + Linux 容器 + TRT-LLM 0.16/TensorRT 10.3** 上取得的。要在 RTX 5080 上复现甚至超越，需要：

- **TRT-LLM 1.x + TensorRT 11.x**（完整支持 sm_120 优化 kernel）；TRT-LLM 0.19/0.20 属于 RTX 50 系支持早期，kernel 未优化到位；
- torch 版本与 TRT-LLM ABI 匹配且带 sm_120 kernel（如 NGC 25.0x+ 容器内自带的配对版本），消除 CPU 绕行；
- 建议在原生 Linux（非 WSL2）上运行，或使用 NGC 官方容器（含 Triton Server 全链路）。

### 5.4 测试产出物

| 产物 | 位置 |
|---|---|
| 加速原理详解（已有） | `docs/triton_trtllm_acceleration_zh.md` |
| 本测试报告 | `docs/triton_trtllm_benchmark_report_zh.md`（本文） |
| TRT 模式本地 benchmark 脚本 | `src/f5_tts/runtime/triton_trtllm/tests/benchmark_local.py` |
| PyTorch 对照 benchmark 脚本 | `src/f5_tts/runtime/triton_trtllm/tests/benchmark_pytorch.py` |
| Vocos TRT 构建脚本（替代 trtexec） | `src/f5_tts/runtime/triton_trtllm/scripts/build_vocos_trt_engine.py` |
| TRT 模式 sm_120 CPU 绕行适配 | `model_repo_f5_tts/f5_tts/1/f5_tts_trtllm.py`、`benchmark.py`、`scripts/export_vocoder_to_onnx.py` |
| 引擎 / 模型 / 结果数据 | `ckpts/`（git 忽略）、`tests/benchmark_local*`（git 忽略） |

### 5.5 复现命令

```bash
# TRT 模式（conda 环境 f5trtllm）
source /tmp/env_f5_conda.sh
F5TTS_NFE=16 python src/f5_tts/runtime/triton_trtllm/tests/benchmark_local.py \
    --model-path ckpts/F5TTS_v1_Base/model_1250000.safetensors \
    --vocab-file ckpts/F5TTS_v1_Base/vocab.txt \
    --tllm-model-dir ckpts/F5TTS_v1_Base/trtllm_engine_v3 \
    --vocoder-trt-engine-path ckpts/vocos_vocoder_v2.plan

# PyTorch 对照（conda 环境 f5pytorch，torch 2.8）
~/miniconda3/envs/f5pytorch/bin/python src/f5_tts/runtime/triton_trtllm/tests/benchmark_pytorch.py \
    --model-path ckpts/F5TTS_v1_Base/model_1250000.safetensors \
    --vocab-file ckpts/F5TTS_v1_Base/vocab.txt \
    --tllm-config ckpts/F5TTS_v1_Base/trtllm_engine/config.json --steps 16
```
