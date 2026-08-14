# F5-TTS Triton + TensorRT-LLM 推理加速方案全解析

> 本文档基于 F5-TTS 仓库中 `src/f5_tts/runtime/triton_trtllm/` 目录的完整代码分析，从**加速原理**、**代码结构**、**执行流程**、**关键设计决策**四个维度，对这套推理加速方案做系统性介绍。

---

## 目录

- [1. 总体架构：三件套](#1-总体架构三件套)
- [2. 加速原理总览](#2-加速原理总览)
- [3. 目录结构与文件职责](#3-目录结构与文件职责)
- [4. 端到端工作流（run.sh 各阶段）](#4-端到端工作流runsh-各阶段)
- [5. 核心组件深入分析](#5-核心组件深入分析)
  - [5.1 权重转换：convert_checkpoint.py](#51-权重转换convert_checkpointpy)
  - [5.2 Patch 机制：把 F5TTS 注册进 TensorRT-LLM](#52-patch-机制把-f5tts-注册进-tensorrt-llm)
  - [5.3 引擎封装：f5_tts_trtllm.py](#53-引擎封装f5_tts_trtllmpy)
  - [5.4 Triton 服务模型：model.py](#54-triton-服务模型modelpy)
  - [5.5 Vocos 声码器导出](#55-vocos-声码器导出)
  - [5.6 客户端与基准测试](#56-客户端与基准测试)
- [6. 与原生 PyTorch 推理的差异对照](#6-与原生-pytorch-推理的差异对照)
- [7. 性能数据](#7-性能数据)
- [8. 已知限制与注意事项](#8-已知限制与注意事项)
- [9. 参考](#9-参考)

---

## 1. 总体架构：三件套

F5-TTS 的推理分为两大计算阶段，加上一个服务编排层，这套方案针对每一层都做了专门优化：

```
┌──────────────────────────────────────────────────────────────────────┐
│                    Triton Inference Server (24.12)                    │
│                                                                       │
│  HTTP :8000 / gRPC :8001 / Metrics :8002                              │
│                                                                       │
│  ┌──────────────────────────┐      ┌──────────────────────────────┐  │
│  │  Model: f5_tts           │      │  Model: vocoder              │  │
│  │  backend: python         │      │  backend: tensorrt           │  │
│  │  max_batch_size: 4       │      │  max_batch_size: 4           │  │
│  │  dynamic_batching        │      │  dynamic_batching            │  │
│  │  queue_delay: 1000 µs    │      │  queue_delay: 1 µs           │  │
│  │                          │      │                              │  │
│  │  [预处理]                │      │  vocoder.plan               │  │
│  │   · 重采样/去均值归一化   │      │  (TensorRT Engine, FP32)    │  │
│  │   · Mel 频谱计算          │      │                              │  │
│  │   · rjieba+pypinyin G2P  │      │   mel ──► ConvNeXt ──►        │  │
│  │   · tokenize + pad       │      │         Backbone ──► iSTFT   │  │
│  │                          │      │         ──► waveform         │  │
│  │  [F5TTS DiT 主干]        │      │                              │  │
│  │   · TextEmbedding        │      └──────────────────────────────┘  │
│  │     (PyTorch, 每样本1次) │              ▲  (in-process, DLPack)   │
│  │   · rank*.engine         │              │                          │
│  │     (TRT-LLM, FP16)      │──────────────┘                          │
│  │  [后处理]                │                                         │
│  │   · 裁剪 + RMS 恢复      │                                         │
│  └──────────────────────────┘                                         │
└──────────────────────────────────────────────────────────────────────┘
```

| 组件 | 技术选型 | 加速对象 | 原因 |
|---|---|---|---|
| **DiT 主干**（22 层 Transformer，Flow Matching 的骨干网络） | TensorRT-LLM（`tensorrt-llm==0.16.0`）编译为 `rank0.engine` | 推理热点：每次采样要跑 **32 次**（NFE）前向 | 结构是标准 Transformer（QKV 注意力 + FFN + AdaLN），正好是 TRT-LLM 最擅长的优化对象 |
| **Vocos 声码器**（mel → 波形） | 普通 TensorRT（`trtexec` 构建 `vocoder.plan`） | 每次请求 1 次 | 纯卷积结构，无动态控制流，ONNX → TRT 即可 |
| **服务编排** | Triton Inference Server（python backend + tensorrt backend） | 多请求并发、批处理 | 提供 dynamic batching、模型生命周期管理、HTTP/gRPC 接口 |

> **为什么 DiT 用 TensorRT-LLM 而不是普通 TensorRT？**
> TRT-LLM 是 NVIDIA 面向 Transformer/LLM 的推理框架，相比直接用 `trtexec` 编译 ONNX，它提供了：
> 1. **图级 API**（`functional` 模块）：用 Python 直接构建计算图，编译期做 **kernel 融合**（如 QKV 三个 GEMM 融合、残差+LayerNorm 融合、激活融合）；
> 2. **高度优化的 attention 插件**（`bert_attention` plugin：融合了 QKV 投影后的 attention 计算、支持变长序列与 mask）；
> 3. **多精度与量化支持**（FP16/BF16/FP8，`--fp8_linear` 选项）；
> 4. **多卡并行**（TP/CP 张量并行与上下文并行，代码中通过 `Mapping` 支持）；
> 5. **编译期 shape 优化与显存规划**（dynamic shape 的 min/opt/max 三档优化）。

---

## 2. 加速原理总览

先建立直觉：**F5-TTS 的 DiT 是一个扩散式 Transformer**。在推理时，CFM（Conditional Flow Matching）用 Euler 法求解 ODE，需要把同一个 Transformer 网络**反复执行 NFE 次**（代码中固定为 32 步，配合 EPSS 步进采样与 sway sampling）。因此推理耗时的 90%+ 集中在"DiT 前向 × 32"上。

整套方案的加速思路可以归纳为 **8 个层面**：

| # | 加速手段 | 原理 | 对应代码 |
|---|---|---|---|
| 1 | **离线编译优化** | 把 PyTorch 动态图编译为 TensorRT CUDA engine：算子融合、kernel 自动选择、显存静态规划、消除 Python 解释器与 PyTorch dispatch 开销 | `trtllm-build`（run.sh stage 1） |
| 2 | **低精度推理** | 默认 FP16（可 BF16/FP8），显存减半、计算吞吐翻倍；Attention 的 QKV/O 投影层以 FP32 精度构建以保证数值稳定性 | `convert_checkpoint.py --dtype` |
| 3 | **CFG 双分支 batch 融合** | Classifier-free guidance 的 cond/uncond 两条前向**拼成 2B 的 batch 一次执行**，kernel 启动次数减半，GPU 利用率翻倍 | `f5_tts_trtllm.py` `forward()` / `sample()` |
| 4 | **时间步与 RoPE 预计算** | 32 步的 sinusoidal 时间嵌入 `time_expand`、4096 长度的 `rope_cos/rope_sin`、Euler 步长 `delta_t` 全部在初始化时算好，**32 次迭代内零重复计算**（原生 PyTorch 每步都要现算正弦嵌入） | `f5_tts_trtllm.py.__init__` |
| 5 | **文本编码器移出热点循环** | `TextEmbedding`（Embedding + RoPE + 4 层 ConvNeXtV2）每样本只算一次，留在 PyTorch 中执行，**不进 TRT engine**；engine 的输入直接是拼好的文本嵌入 | `sample()` 中逐条计算 + `pad_sequence` |
| 6 | **采样器数学折叠** | EPSS 步进表、sway sampling（`coef=-1` 的 cos 时间变换）在**初始化时静态折叠**进 `time_expand` 与 `delta_t`，运行期只剩 `x += v·Δt` 的乘加 | `__init__` 中 `time_step = 1 - cos(πt/2)` |
| 7 | **服务端批处理** | Triton `dynamic_batching`：请求先排队 `max_queue_delay_microseconds=1000`，攒够 batch 一起执行；vocoder 配置 `preferred_batch_size` | `config.pbtxt` |
| 8 | **运行时零拷贝** | 引擎输出 buffer 静态分配（`_setup` 一次性分配）；python backend 与 vocoder 之间通过 **DLPack** 交换张量（`from_dlpack/to_dlpack`），无显式拷贝 | `model.py` `forward_vocoder` |

下面逐层展开。

---

## 3. 目录结构与文件职责

```
src/f5_tts/runtime/triton_trtllm/
├── README.md                        # 官方使用说明（含基准数据）
├── Dockerfile.server                # 服务镜像：tritonserver:24.12 + tensorrt-llm==0.16.0
├── docker-compose.yml               # 一键启动（自动下载模型并 build）
├── run.sh                           # 9 阶段流水线（下载→转换→导出→建服→测服→基准）
│
├── model_repo_f5_tts/               # Triton 模型仓库模板（stage 3 会拷贝并填充变量）
│   ├── f5_tts/
│   │   ├── config.pbtxt             # f5_tts 模型配置：python backend、动态批处理、参数占位符
│   │   └── 1/
│   │       ├── model.py             # Triton Python backend：预处理/后处理/调用引擎与 vocoder
│   │       └── f5_tts_trtllm.py     # TRT-LLM 引擎封装：F5TTS 类（加载/采样/CFG/流管理）
│   └── vocoder/
│       ├── config.pbtxt             # vocoder 模型配置：tensorrt backend
│       └── 1/vocoder.plan           # 构建产物（stage 2 生成后拷入）
│
├── patch/                           # TRT-LLM 源码补丁（拷贝覆盖其 models 目录）
│   ├── __init__.py                  # 注册 F5TTS 架构到 MODEL_MAP
│   └── f5tts/
│       ├── model.py                 # TRT-LLM 版 F5TTS 网络定义（functional API）
│       └── modules.py               # TRT-LLM 版 DiTBlock/Attention/AdaLN/ConvNeXt 等算子
│
├── scripts/
│   ├── convert_checkpoint.py        # PyTorch ckpt → TRT-LLM ckpt（权重名映射/TP切分/QK缩放）
│   ├── export_vocoder_to_onnx.py    # Vocos → ONNX（iSTFT 换成可导出的卷积实现）
│   ├── export_vocos_trt.sh          # trtexec：ONNX → TensorRT engine（动态 shape）
│   ├── conv_stft.py                 # 卷积版 STFT/iSTFT（ONNX 可导出）
│   └── fill_template.py             # 用实际路径填充 config.pbtxt 的 ${变量}
│
├── benchmark.py                     # 离线基准：TRT-LLM backend vs PyTorch backend，RTF 统计
├── client_grpc.py                   # gRPC 并发压测客户端（多任务异步发送）
└── client_http.py                   # HTTP 客户端示例
```

---

## 4. 端到端工作流（run.sh 各阶段）

`run.sh <start_stage> <stop_stage> <model>`，模型可选 `F5TTS_v1_Base | F5TTS_Base | F5TTS_v1_Small | F5TTS_Small`：

```
stage 0  下载模型        huggingface-cli download SWivid/F5-TTS → ckpts/
stage 1  [加速核心①]     convert_checkpoint.py → trtllm_ckpt/
                         cp patch/* → tensorrt_llm/models/     （注册 F5TTS 架构）
                         trtllm-build → trtllm_engine/rank0.engine
stage 2  [加速核心②]     export_vocoder_to_onnx.py → vocos_vocoder.onnx
                         export_vocos_trt.sh (trtexec) → vocos_vocoder.plan
stage 3  构建模型仓库     cp model_repo_f5_tts → model_repo/
                         fill_template.py 填充 config.pbtxt 的 ${vocab}/${model}/${trtllm}/${vocoder}
                         cp vocoder.plan → model_repo/vocoder/1/
stage 4  启动服务         tritonserver --model-repository=./model_repo
stage 5  测试 gRPC        client_grpc.py（并发压测 + 输出 RTF/延迟分位数/Triton 统计）
stage 6  测试 HTTP        client_http.py（单条语音合成）
stage 7  离线基准(TRT)    benchmark.py --backend-type trt   （torchrun 单卡）
stage 8  离线基准(PyTorch) benchmark.py --backend-type pytorch（对照实验）
```

几个值得注意的工程细节：

- **stage 1 的 patch 动作**：`cp -r patch/* $python_package_path/tensorrt_llm/models` —— 直接**覆盖** TRT-LLM 安装目录里的 `models/__init__.py`，在 `MODEL_MAP` 中注册 `"F5TTS": F5TTS`。`convert_checkpoint.py` 写入的 `config.json` 中 `architecture: "F5TTS"`，`trtllm-build` 读到该字段后即通过注册表实例化 patch 中的网络定义。这就是"给 TRT-LLM 增加自定义架构"的标准姿势。
- **engine 构建参数**：`--max_batch_size 8 --remove_input_padding disable`。当前默认**关闭** remove-input-padding（RMP），引擎按 `[B, N, D]` 的 3D 输入构建；但 patch 代码里 RMP 路径（2D 输入 `[ΣN, D]`）也完整实现了，可自行开启对比。
- **自定义检查点**：修改 `run.sh` 中的 `ckpt_file` / `vocab_file`；若结构不同需改 `convert_checkpoint.py`。fp32 训练/微调的模型要加 `--dtype float32`。

---

## 5. 核心组件深入分析

### 5.1 权重转换：convert_checkpoint.py

这是"PyTorch 模型 → TRT-LLM 模型"的第一步，核心工作有三件：

**(a) 权重名映射**（`pytorch_to_trtllm_name` 正则表）：

| PyTorch 权重名 | TRT-LLM 权重名 | 原因 |
|---|---|---|
| `time_embed.time_mlp.0.*` | `time_embed.mlp1.*` | `nn.Sequential` 展开为独立模块 |
| `time_embed.time_mlp.2.*` | `time_embed.mlp2.*` | 同上 |
| `input_embed.conv_pos_embed.conv1d.0.*` | `input_embed.conv_pos_embed.conv1d1.*` | 同上 |
| `input_embed.conv_pos_embed.conv1d.2.*` | `input_embed.conv_pos_embed.conv1d2.*` | 同上 |
| `transformer_blocks.N.attn.to_out.0.*` | `transformer_blocks.N.attn.to_out.*` | 去掉 `nn.ModuleList` 下标 |
| `transformer_blocks.N.ff.ff.0.0.*` | `transformer_blocks.N.ff.project_in.*` | 同上 |
| `transformer_blocks.N.ff.ff.2.*` | `transformer_blocks.N.ff.ff.*` | 同上 |

- 只取 `ema_model.transformer.*`（或 `transformer.*`）前缀下的权重，**TextEmbedding 等不进引擎的模块会被排除**（引擎只需要 DiT 主干的权重）。
- 卷积权重 `unsqueeze(-1)` 适配 TRT-LLM `Conv1d` 的权重布局。

**(b) Q/K 缩放折叠**（第 168-184 行）：

```python
scale_factor = math.pow(64, -0.25)   # = 0.35355…
# to_q / to_k 的 weight 和 bias 都乘以 scale_factor
```

原生 PyTorch 路径中注意力带有一层 QK 缩放（x-transformers RoPE/xpos 相关），TRT-LLM 路径的 RoPE 是显式预计算输入，因此把该缩放**静态折叠进 Q/K 权重**，保证引擎输出与原模型数值一致。

**(c) 张量并行切分**（`split_q_tp` / `split_q_bias_tp` / `split_matrix_tp`）：
Q/K/V 按 head 维切分（`dim=1`/`dim=0`），`to_out` 按输出维切分。`--tp_size`/`--cp_size`/`--pp_size` 可配，**pp_size 暂不支持**（`assert args.pp_size == 1`）。

**(d) 模型超参固化**：`--model_name` 选择后自动填充 `hidden_size/depth/num_heads/dim_head/ff_mult/text_dim/pe_attn_head` 等，与 F5-TTS 仓库定义一一对应（Base: 1024/22/16/64；Small: 768/18/12/64）。注意 v0 与 v1 的关键差异：
- **v0**（`F5TTS_Base/Small`）：`text_mask_padding=False`、`pe_attn_head=1`（RoPE 只作用于第 1 个注意力头）；
- **v1**（`F5TTS_v1_Base/Small`）：`text_mask_padding=True`、`pe_attn_head=None`（RoPE 作用于所有头）。

最终产出 `config.json`（含 `architecture: "F5TTS"`、dtype、超参、mapping、可选 FP8 量化配置）+ `rank*.safetensors` 权重。

### 5.2 Patch 机制：把 F5TTS 注册进 TensorRT-LLM

#### 5.2.1 `patch/f5tts/model.py` —— 网络定义

TRT-LLM 的 `PretrainedModel` 子类，用 **functional API**（`tensorrt_llm.functional` 的惰性张量）描述计算图，`trtllm-build` 阶段编译成 engine。

**引擎的输入输出契约**（`prepare_inputs()` + `forward()`）：

| 张量 | shape（非 RMP 模式） | 说明 |
|---|---|---|
| `noise` | `[B, N, 100]` | 当前步的带噪 mel（CFG 下 B=2b） |
| `cond` | `[B, N, 612]` | `[mel(100) ‖ text_embed(512)]`，引擎内与 noise 拼接成 712 维进 `proj` |
| `time` | `[B, 256]` | **已展开的** sinusoidal 时间嵌入（不是标量时间步！） |
| `rope_cos` / `rope_sin` | `[B, N, 64]` | 预计算的 RoPE 余弦/正弦 |
| `input_lengths` | `[B]` (int32) | 每个样本的有效长度，用于生成 mask |
| `denoised`（输出） | `[B, N, 100]` | 预测的速度场（flow） |

动态 shape 三档（`dim_range`）：batch `[2, 2, max_batch_size]`（**最小为 2**，因为 CFG 双分支至少占 2 个 batch 位）、序列长度 `[100, 1500, 3000]`（mel 帧，`prepare_inputs` 中 `max_seq_len=3000`）。编译期按三档做 shape 特化优化，运行时 `Session.set_shapes` 指定实际 shape。

**mask 生成**（非 RMP 时，`forward()` 第 81-95 行）：`position_ids < input_lengths` 广播展开成 `[B, N]` 的 int32 mask，用于 ConvPositionEmbedding 和 attention 的 padding 屏蔽。

**输出标记**：`denoise.mark_output("denoised", self.dtype)` —— 引擎的输出张量名必须与运行期 `f5_tts_trtllm.py` 中的 `expected_tensor_names` 完全一致（有校验逻辑）。

#### 5.2.2 `patch/f5tts/modules.py` —— 算子级重写

与原生 PyTorch 版逐模块对应：

| 模块 | PyTorch 原版 | TRT-LLM patch | 优化点 |
|---|---|---|---|
| `TimestepEmbedding` | Sequential：`SinusPositionEmbedding → Linear → SiLU → Linear` | `mlp1 → SiLU → mlp2`，**正弦嵌入外移** | 正弦嵌入由调用方预计算好直接喂入，引擎省掉 32 步 × 2 次正弦计算 |
| `AdaLayerNormZero` | `norm(x)·(1+scale)+shift`，输出 6 个调制参数 | 同语义；`chunk(emb, 6, dim=1)` | 与 TRT-LLM `LayerNorm`（elementwise_affine=False）组合 |
| `ConvPositionEmbedding` | `nn.Sequential(Conv1d, Mish, Conv1d, Mish)` + masked_fill | `conv1d1 → Mish → conv1d2 → Mish`，mask 用乘法实现 | 消除逐元素 masked_fill，乘法可融合 |
| `Attention` | `nn.Linear` ×4 + SDPA/flash-attn | `ColumnLinear/RowLinear`（支持 TP）+ **`bert_attention` 插件** | QKV 投影可融合；attention 走 NVIDIA 高度优化的融合 kernel |
| `FeedForward` | `Linear → GELU → Linear` | `project_in → GELU → ff` | GELU 与 GEMM 融合 |
| RoPE | x-transformers `RotaryEmbedding` 运行时生成 | `apply_rotary_pos_emb_3dim`：**用预计算好的 cos/sin 输入做 rotate-every-two 旋转** | 位置编码零运行时开销 |

**Attention 的 `bert_attention` 插件路径**（modules.py 第 311-334 行）：当 `plugin_config.bert_attention_plugin` 开启时，QKV 拼接后直接调用 `bert_attention(qkv, input_lengths, heads, head_dim, q_scaling, max_input_length)` —— 这是 TRT-LLM 的融合注意力 kernel（含 mask 处理、softmax 融合、FP32 累加），是相比 PyTorch SDPA 的主要提速点之一。

**精度策略**：QKV/O 投影层显式以 `float32` 声明 dtype（`self.dtype = str_dtype_to_trt("float32")`），其余层跟随模型 `config.dtype`（默认 FP16）——注意力投影用更高精度换取数值稳定，其余 GEMM 用 FP16 吃满 Tensor Core 吞吐。

### 5.3 引擎封装：f5_tts_trtllm.py

这是运行期的核心文件，`F5TTS` 类（注意与 patch 里的 TRT-LLM 模型同名但职责不同）负责加载引擎并实现采样循环。

**(a) 引擎加载与多卡映射**：

```python
rank = tensorrt_llm.mpi_rank()
self.mapping = tensorrt_llm.Mapping(world_size, rank, cp_size, tp_size, pp_size=1, ...)
engine_file = os.path.join(tllm_model_dir, f"rank{rank}.engine")
self.session = Session.from_serialized_engine(engine_buffer)   # 反序列化 engine
```

支持 TP/CP 多卡（每 rank 加载自己的 `rankN.engine`），同时校验引擎 IO 张量名是否与预期一致（`debug_mode` 下允许出现额外张量，便于调试）。

**(b) 初始化期预计算**（`__init__` 后半段，全部一次算完）：

```python
# 1) RoPE：base=10000，head_dim=64，长度 4096，repeat_interleave(2) 匹配 rotate-every-two
self.rope_cos = self.freqs.cos().half()   # [1, 4096, 64]
self.rope_sin = self.freqs.sin().half()

# 2) 时间步：EPSS 32 步 + sway sampling(coef=-1) 折叠
t = 1/32 * torch.tensor(epss[32])                # 均匀 0..1 的 33 个点
time_step = 1 - torch.cos(torch.pi * t / 2)      # = t + (-1)·(cos(πt/2) - 1 + t)，即 sway=-1
delta_t = torch.diff(time_step)                  # Euler 步长 Δt，32 个

# 3) 正弦时间嵌入：与 SinusPositionEmbedding(scale=1000) 完全一致，32 步全部预展开
emb_factor = 1000.0 * exp(-log(10000)/127 · arange(128))
time_expand[:, i, :] = cat([sin(time_step[i]·emb_factor), cos(time_step[i]·emb_factor)])
```

> 对照原生 `cfm.py` 的采样：`t = get_epss_timesteps(32)` 后 `t = t + sway_sampling_coef·(cos(πt/2) − 1 + t)`，再交给 `odeint(euler)` 用 `Δt` 积分。TRT 版把 `sway_sampling_coef=-1` 的变换直接写死为 `1 − cos(πt/2)`，并把 `Δt`、正弦嵌入全部预计算，运行期**只剩查表和乘加**。

**(c) CFG 融合采样循环**（`forward()`，这是加速的核心之一）：

```python
cfg_strength = 2.0
half_batch = batch_size // 2
noise_half = noise[:half_batch]                     # 保留初始噪声

for i in range(self.nfe_steps):                     # 32 次
    self._setup(batch_size, noise.shape[1])         # 静态分配输出 buffer
    current_noise = torch.cat([noise_half, noise_half], dim=0)   # 双分支共享同一噪声
    current_time  = time_expand[:, i]               # 查表
    self.session.set_shapes(current_inputs)         # 指定动态 shape
    ok = self.session.run(self.inputs, self.outputs, self.stream.cuda_stream)

    pred_cond   = self.outputs["denoised"][:half_batch]    # 条件分支
    pred_uncond = self.outputs["denoised"][half_batch:]    # 无条件分支
    guidance = pred_cond + (pred_cond - pred_uncond) * cfg_strength   # CFG 加权
    noise_half = noise_half + guidance * delta_t[i]        # Euler 积分
```

要点：
- **cond/uncond 一次前向**：`sample()` 里把 `[noise; noise]`、`[cat_mel_text; cat_mel_text_drop]`、`[rope; rope]` 等沿 batch 拼接成 2B，一个 engine 调用同时算出两个分支，再拆开做 CFG 加权。相比逐分支执行，**kernel 启动次数减半**，且 GPU 在 batch=2 时利用率更高。
- **`cuda_stream_guard` 装饰器**：Triton Python backend 可能多线程/多流执行，装饰器保证进入引擎前同步外部流、切换到 session 绑定的流，跑完再切回——避免流竞争导致的错误结果或死锁。
- **静态输出 buffer**：`_setup()` 按 `[batch, seq_len, dim]` 一次性 `torch.empty` 分配输出，避免每步动态分配。

**(d) 文本嵌入与输入组装**（`sample()`）：

```python
for i in range(batch):   # 逐条算，避免 batch 内 padding 错位
    text_embedding_i     = self.text_embedding(text[i], est_len[i], drop_text=False)
    text_embedding_drop_i = self.text_embedding(text[i], est_len[i], drop_text=True)
    list.extend([emb_i, emb_drop_i])           # cond/uncond 交错
text_and_drop_embedding = pad_sequence(list, batch_first=True, padding_value=0)
text_embedding      = text_and_drop_embedding[0::2]   # 条件文本嵌入
text_embedding_drop = text_and_drop_embedding[1::2]   # drop 文本嵌入

cat_mel_text = cat([cond_pad_sequence, text_embedding], dim=-1)        # [B,N,612]
cat_mel_text_drop = cat([zeros(B,N,100), text_embedding_drop], dim=-1) # 音频条件置零
```

- `TextEmbedding`（`f5_tts_trtllm.py` 内重新定义的 PyTorch 模块：Embedding + RoPE 加法 + 4 层 ConvNeXtV2 + GRN）直接从原 checkpoint 加载权重（`get_text_embed_dict` 只抽取 `transformer.text_embed.*` 键）。它**不在引擎里**，因为每样本只算一次，而 DiT 要跑 32 次——把低频计算留在 PyTorch、高频计算压进引擎，是这套方案的重要设计取舍。
- `input_lengths = torch.tensor(estimated_reference_target_mel_len, dtype=int32)`：估计的目标 mel 长度（`ref_mel_len × (1 + len(target_text)/len(reference_text))`），用于引擎内 mask 与（可选）padding 去除。
- **`remove_input_padding` 路径**：开启时把所有 `[B,N,D]` 输入用 `remove_tensor_padding()` 压成 `[ΣN_i, D]` 一维拼接，消除 padding 区域的无效计算（尤其省 attention 的 softmax 规模）；输出再按长度切回。当前 `run.sh` 默认 `disable`（`--remove_input_padding disable`），该路径作为可选优化保留。

### 5.4 Triton 服务模型：model.py

`TritonPythonModel` 是 Triton Python backend 的入口，完整覆盖一条 TTS 请求的生命周期：

**`initialize`**：读取 `config.pbtxt` 的 `parameters`（vocab/model/trtllm/vocoder 路径，stage 3 由 `fill_template.py` 填充）；构建 tokenizer（`vocab.txt` → `{char: idx}`）；加载 TRT-LLM 引擎（实例化上面的 `F5TTS`）；初始化 vocos 的 mel 前端（`torchaudio MelSpectrogram`：1024 FFT / 256 hop / 100 mel，power=1）或 BigVGAN 前端。

**`execute`（预处理）**：
1. 从请求取 `reference_wav` / `reference_wav_len` / `reference_text` / `target_text`（DLPack 零拷贝取张量）；
2. RMS 归一化：`if ref_rms < 0.1: wav *= 0.1/ref_rms`（与训练一致）；
3. 重采样到 24kHz（`torchaudio Resample`）；
4. mel 频谱 → 按 `estimated_reference_target_mel_len` 补齐到 batch 最大长度（`max_mel_len=4096` 截断）；
5. 文本 G2P：**rjieba 分词 + pypinyin（Style.TONE3 + 变调）** 转拼音，`list_str_to_idx` 转 token（pad 值 -1）；
6. 调 `self.model.sample(...)` 得到去噪 mel。

**`execute`（后处理 + vocoder 调用）**：

```python
def forward_vocoder(self, mel):
    mel = mel.to(torch.float32).contiguous().cpu()
    input_tensor_0 = pb_utils.Tensor.from_dlpack("mel", to_dlpack(mel))
    inference_request = pb_utils.InferenceRequest(
        model_name="vocoder", requested_output_names=["waveform"], inputs=[input_tensor_0])
    inference_response = inference_request.exec()      # in-process 调用另一个模型
    waveform = torch.utils.dlpack.from_dlpack(...)
```

- 对每条样本裁出 `denoised[ref_mel_len:estimated_mel_len]`（只保留新合成部分）→ 转置为 `[1, 100, T]` → 发给 vocoder；
- 用 `pb_utils.InferenceRequest.exec()` **进程内**调用 vocoder 模型（Triton 的模型集成能力，DLPack 交换避免拷贝）；
- 输出前做 RMS 反向恢复（`audio *= ref_rms/0.1`），保证响度与参考音频一致。

**服务配置要点**（`config.pbtxt`）：
- `f5_tts`：`max_batch_size: 4` + `dynamic_batching { max_queue_delay_microseconds: 1000 }` —— 请求先等最多 1ms 攒批，多个并发请求合成一个 4 以内的 batch 进引擎，这是客户端并发（README 中 concurrency=2 场景）下吞吐的关键。
- `vocoder`：`preferred_batch_size: [1, 2, 4]`，与 f5_tts 的批大小对齐。

### 5.5 Vocos 声码器导出

Vocos（`charactr/vocos-mel-24khz`）由 ConvNeXt backbone + iSTFT head 组成。导出为 TRT engine 有两个难点，代码都做了处理：

1. **iSTFT 不可 ONNX 导出**：`export_vocoder_to_onnx.py` 用 `conv_stft.py`（卷积版 STFT/iSTFT：分帧=Conv1d、FFT=Linear、OLA=ConvTranspose1d，参考 echocatzh/conv-stft）替换 `vocos.head` 的 istft 实现，数学等价但全部是可导出的卷积/矩阵算子：
   ```python
   istft_head_for_export = ISTFTHead(n_fft, hop_length)
   istft_head_for_export.out = self.vocos_vocoder.head.out   # 复用原 head 的线性层
   self.vocos_vocoder.head = istft_head_for_export
   ```
   输出侧：`mag = exp(mag).clip(max=1e2)`，`real/imag = mag·cos/sin(phase)`，再 `stft.inverse` 还原波形。
2. **动态 shape**：ONNX 导出声明 `dynamic_axes`（batch、input_length），`export_vocos_trt.sh` 用 `trtexec` 按三档构建：
   ```
   --minShapes="mel:1x100x1"   --optShapes="mel:1x100x1000"   --maxShapes="mel:8x100x3000"
   ```
   FP32 精度。构建产物 `vocoder.plan` 由 `run.sh` stage 3 拷入模型仓库。

### 5.6 客户端与基准测试

- **`client_grpc.py`**：asyncio + `tritonclient.grpc.aio` 并发发送多个任务流（`--num-tasks`），每个请求含 `reference_wav`（按 1 秒对齐 padding）、`reference_wav_len`、`reference_text`、`target_text` 四个输入；输出 `waveform`。结束时输出 RTF、延迟分位数（P50/P90/P95/P99），并解析 `get_inference_statistics` 得到 Triton 侧的 queue/compute 时间与 batch 分布统计——**这些统计直接反映 dynamic batching 的收益**。
- **`client_http.py`**：单条请求示例，POST 到 `/v2/models/f5_tts/infer`。
- **`benchmark.py`**：离线对比模式（`--backend-type trt | pytorch`）：
  - TRT 模式：直接实例化 `F5TTS`（引擎封装类）跑 `sample()`，vocoder 用 `VocosTensorRT`（`Session.from_serialized_engine` + `infer_shapes` 动态推断输出 shape）；
  - PyTorch 模式：`load_model(DiT, ...)` 走原生 `model.sample(..., steps=32, cfg_strength=2.0, sway_sampling_coef=-1)`，保证与 TRT 路径**采样配置一致**，作为公平对照；
  - 数据来自 HF 数据集 `yuekai/seed_tts`（wenetspeech4tts 等 split），按估计时长降序排列以稳定批内计算量；统计 `RTF`（real-time factor）、DiT 时间、vocoder 时间。

---

## 6. 与原生 PyTorch 推理的差异对照

| 环节 | 原生 PyTorch（`infer/utils_infer.py` + `model/cfm.py`） | TRT-LLM 方案 |
|---|---|---|
| DiT 执行 | eager 模式，PyTorch dispatch | 预编译 engine，kernel 融合，FP16 |
| CFG | `cfg_infer=True` 时也做 batch 拼接（2b），但无编译优化 | 同思路 + 编译优化叠加 |
| 时间步 | 每步现算 `SinusPositionEmbedding(t)`（scale=1000） | 32 步全部预计算，查表 |
| RoPE | `RotaryEmbedding.forward_from_seq_len(seq_len)` 每步生成 | 预计算 4096 长度，切片 + repeat |
| sway sampling | `t + coef·(cos(πt/2) − 1 + t)`，coef=-1 | 折叠进 `time_step = 1−cos(πt/2)` 与 `delta_t` |
| ODE 积分 | `torchdiffeq.odeint(euler)` | 手写 `x += v·Δt` 循环（等价的 Euler） |
| 文本嵌入 | 每 batch 一次（含 cache 机制） | 逐条计算避免 padding 错位 + `pad_sequence` |
| Vocoder | PyTorch Vocos（含 torch iSTFT） | TRT engine（卷积 iSTFT） |
| 服务 | 单进程 CLI/Gradio | Triton 多模型、dynamic batching、HTTP/gRPC |
| 精度 | FP32（`torch.inference_mode`） | 引擎 FP16（QKV/O 投影 FP32），数值经权重缩放对齐 |

数值一致性保障点（转换脚本中）：
- `to_q/to_k` 权重与偏置 × `64^(−0.25)`（QK 缩放折叠）；
- 卷积权重 `unsqueeze(-1)` 适配布局；
- 时间嵌入公式与 `SinusPositionEmbedding(scale=1000)` 逐位一致；
- fp32 训练的模型必须 `--dtype float32` 转换，否则精度损失不可接受（README 特别标注）。

---

## 7. 性能数据

README 官方数据（**单张 L20 GPU，26 组 prompt_audio & target_text 对，16 NFE**）：

| 模式 | 并发/批大小 | 平均延迟 | RTF（实时率） |
|---|---|---|---|
| F5-TTS Base (Vocos) 客户端-服务端 | 并发 2 | **253 ms** | **0.0394** |
| F5-TTS Base (Vocos) 离线 TRT-LLM | batch 1 | — | 0.0402 |
| F5-TTS Base (Vocos) 离线 PyTorch | batch 1 | — | 0.1467 |

- RTF ≈ **0.04** 意味着生成 1 秒音频只需约 40ms GPU 时间；
- TRT-LLM 相对原生 PyTorch 约 **3.6× 加速**（0.1467 → 0.0402）；
- 客户端-服务端并发 2 时平均单请求延迟 253ms，RTF 与离线持平，说明 dynamic batching 的排队开销被批处理收益抵消。

---

## 8. 已知限制与注意事项

1. **单请求 batch=1**：`model.py` 中 `assert wav.shape[0] == 1`（每个请求只接受 1 条参考音频），批量能力靠 Triton 服务端 dynamic batching 合并多个请求实现。
2. **PP 不支持**：`assert args.pp_size == 1`；TP/CP 支持（`Mapping` 已就绪）。
3. **BigVGAN 未实现**：`export_vocoder_to_onnx.py` 与 `benchmark.py` 中均为 `NotImplementedError`。
4. **采样配置固定**：`nfe_steps=32`、`cfg_strength=2.0`、`sway_sampling_coef=-1`（折叠进预计算）；`freq_embed_dim=256`、`max_mel_len=4096`（引擎内 3000）为硬编码。
5. **引擎与模型版本必须匹配**：v1 权重配 `F5TTS_v1_*` 参数、v0 配 `F5TTS_*`，错配会导致 `text_mask_padding`/`pe_attn_head` 语义不一致。
6. **fp32 训练模型**：转换时务必加 `--dtype float32`，否则输出质量劣化。
7. **patch 与 TRT-LLM 版本绑定**：`Dockerfile.server` 固定 `tensorrt-llm==0.16.0` / `tritonserver 24.12`，升级 TRT-LLM 需要同步适配 patch（functional API 有变动风险）。
8. **输入 padding 未去除**：当前引擎 `--remove_input_padding disable`，不同长度的请求在 batch 内按最长补零，存在少量无效计算（代码已支持开启，可自行实验）。

---

## 9. 参考

- 本方案代码：`src/f5_tts/runtime/triton_trtllm/`（README 为官方说明）
- 主仓库 README：`README.md` 第 140 行；`src/f5_tts/infer/README.md` 第 152 行
- 上游参考实现：[F5-TTS-TRTLLM](https://github.com/Bigfishering/f5-tts-trtllm)（Bigfishering）
- NVIDIA 官方 Whisper TRT-LLM 示例（构建流程模板）：<https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/whisper>
- Triton 动态批处理文档：<https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#delayed-batching>
- conv-stft（ONNX 可导出的卷积 STFT 实现）：<https://github.com/echocatzh/conv-stft>
