<div align="center">

<p>
<img src="assets/urbilateria-mark.svg" width="760" alt="Urbilateria Logo 与字标，其中突出显示了 CLI 命令 urb">
</p>

<h3>让前沿 MoE 检查点变得可运行。</h3>

<p>
<a href="README.md">English</a> · 中文
</p>

<p>
一个纯 Rust 研究运行时，用于在内存有限的纯 CPU 机器上理解和运行<br>
GLM-5.2、DeepSeek-V4/V4.1、Kimi-K3、Qwen3.8 与 Hy4等前沿模型。
</p>

<p>
<img src="https://img.shields.io/badge/Rust-1.83%2B-b7410e?style=flat-square&logo=rust&logoColor=white" alt="Rust 1.83+">
<img src="https://img.shields.io/badge/runtime-CPU--only-3d6b5d?style=flat-square" alt="仅 CPU 运行时">
<img src="https://img.shields.io/badge/model_families-5-247ba0?style=flat-square" alt="五个模型家族">
<img src="https://img.shields.io/badge/tests-338_passing-2e7d32?style=flat-square" alt="338 项测试通过">
<a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-6c5ce7?style=flat-square" alt="MIT 许可证"></a>
</p>

<table>
<tr>
<td align="center"><b>2.78T</b><br><sub>最大支持参数</sub></td>
<td align="center"><b>6</b><br><sub>模型适配器</sub></td>
<td align="center"><b>6</b><br><sub>公开生成路径</sub></td>
<td align="center"><b>338</b><br><sub>默认测试通过</sub></td>
<td align="center"><b>0</b><br><sub>所需 GPU</sub></td>
</tr>
</table>

<p>
<a href="#快速开始">快速开始</a> ·
<a href="#工作原理">工作原理</a> ·
<a href="#模型覆盖">模型覆盖</a> ·
<a href="#验证">验证</a> ·
<a href="#cli-参考">CLI</a> ·
<a href="#开发">开发</a>
</p>

</div>

> [!WARNING]
> Urbilateria 是一个实验性的学习与模型取证项目。其标量运行时优先考虑行为的可读性和
> 可运行性，而非生产吞吐量。GLM-5.2、DeepSeek-V4/V4.1、Kimi-K3、Qwen3.8 与 Hy4
> 已可进行大检查点生成，但数值精度、输出质量、内存占用和速度不保证生产性能。
> DeepSeek-V4.1 提供了标量基础文本生成路径，并已通过独立真实权重 BOS 预言机验证；其逐 token 提示词预填充用于正确性研究，速度较慢。

## 为什么选择 Urbilateria？

大多数推理框架运行前沿模型需要令人瞠目结舌的GPU资源。Urbilateria 则让你能够实际体验这些checkpoint，只需要cpu、高速ssd和适量内存。

纯Rust实现，推理路径不依赖任何 Python 框架。本仓库不包含模型权重。

## 快速开始

构建 release 二进制并查看可用命令：

```bash
cargo build --release --locked
./target/release/urb help
```

从元数据开始。以下命令无需读取完整张量 payload：

```bash
# 静态参数与量化透视
./target/release/urb inspect /path/to/model

# RAM、上下文、KV、暂存空间与专家缓存规划
./target/release/urb plan /path/to/model --ram-gib 32 --context 2048

# 精确的运行时张量与内存验证
./target/release/urb preflight /path/to/model --context 2048 --expert-slots 8
```

试用模型原生 tokenizer 与 chat 模版：

```bash
./target/release/urb tokenize /path/to/model "Explain sparse MoE routing" --chat
```

运行 generate 命令：

```bash
./target/release/urb generate /path/to/model \
  --prompt "Explain sparse MoE routing" \
  --ram-gib 32 \
  --max-new-tokens 8 \
  --no-thinking \
  --allow-large-model
```

`generate` 要求显式指定 RAM 上限和 `--allow-large-model`；即使只生成很短的内容，
也可能从存储设备读取许多 GiB 数据。当前公开命令支持 GLM-5.2、DeepSeek-V4、
DeepSeek-V4.1、纯文本 Kimi-K3、纯文本 Qwen3.8 和 Hy4。Qwen3.8 使用该版本始终开启思考的聊天模板，
因此对该模型应省略 `--no-thinking`。Hy4 可在提示词与生成 token 总数不超过 2,048
时精确执行，此时其 top-2,048 DSA 选择包含完整的因果历史。

## 工作原理

```mermaid
flowchart LR
    A[(原生检查点<br/>分片 + 元数据)] --> B[严格配置<br/>与 schema 门禁]
    B --> C[inspect · list · probe]
    B --> D[plan · preflight]
    B --> E[模型原生<br/>tokenizer + chat]
    E --> F[逐层流式<br/>CPU 运行时]
    D --> F
    F <--> G[(有界的逐层<br/>专家缓存)]
    F --> H[共享贪心<br/>token 循环]
    H --> I[流式 UTF-8<br/>与 profile JSON]

    J[KDA · MLA · DSA<br/>DeltaNet · GQA] -. 家族数学 .-> F
    K[INT8 · INT4 · MXFP8<br/>MXFP4 · BF16 · FP8] -. 原生权重 .-> F
```

运行时明确维护工作集，而不是假装检查点已经常驻内存：

| 内存类别 | 其中保存的内容 | Urbilateria 的控制方式 |
| --- | --- | --- |
| **常驻** | 配置、tokenizer、根向量、模型状态 | 针对请求的 RAM 上限进行一次性验证 |
| **逐层流式** | 解码器主干、大型行、词表头分块 | 按执行顺序加载，并在层边界释放 |
| **缓存受限** | 路由后的 MoE 专家 | 使用具有明确槽位预算的确定性逐层 LRU |
| **序列状态** | MLA KV、KDA/DeltaNet 循环状态、卷积、GQA KV | 根据准确的家族几何和请求的上下文计算 |
| **暂存空间** | 激活、量化块、批量提示词工作区 | 在开始读取 payload 前纳入 preflight |

大型 matvec 输出行在一个持久 CPU worker 池上运行。DeepSeek-V4、Kimi-K3、Hy4 和
Qwen3.8 库运行时按层摄取提示词：一个解码器层为整个提示词加载一次，只有最后一个
提示 token 会进入词表大小的 LM head。DeepSeek-V4.1 当前在提示词预填充与 decode
中都使用精确的逐 token 路径。

## 模型覆盖

| 能力 | GLM-5.2 | DeepSeek-V4 | DeepSeek-V4.1 | Kimi-K3 | Qwen3.8 | Hy4 |
| --- | --- | --- | --- | --- | --- | --- |
| **检查点 ABI** | 转换后的 Colibri | 原生 48 分片 | 原生 48 分片；96,085 个张量 | 原生 96 分片 | 原生 213 分片 | 原生 130 分片 |
| **Tokenizer / 聊天** | byte-BPE | 原生文本聊天 | 原生文本聊天 + 数值 effort | TikToken + XTML | 始终思考的 ChatML | 原生 reasoning/no-think |
| **原生权重** | INT8/INT4 | 128×128 MXFP8 + MXFP4 | 32×32 MXFP8 + MXFP4 | BF16 + MXFP4 | block FP8 | ModelOpt MXFP8 |
| **注意力路径** | MLA | local + compressed | CED + CSA2 + Engram + mHC | KDA + MLA | DeltaNet + GQA | iHC + MLA/DSA |
| **公开 CLI 生成** | 实验性 | 实验性 | 实验性，文本基础运行时 | 实验性 | 实验性 | 实验性，≤2,048 tokens |
| **多模态执行** | 否 | 否 | 仅 schema | 仅 schema | 否 | 否 |

### “原生”的含义

**GLM-5.2。** 运行时期望的是转换后的 Colibri 风格检查点，而非官方 FP8 版本。
它支持转换容器中的常驻核心权重与流式专家。

**DeepSeek-V4。** Urbilateria 直接用 Rust 读取该版本的 48 分片布局：配置、tokenizer、
safetensors 头部、E8M0 scale sidecar、MXFP8 矩阵，以及低半字节优先打包的 MXFP4
专家。DSpark 张量会进行 schema 验证，但推测解码被禁用。

**DeepSeek-V4.1。** 独立适配器验证 40 层 CED 图、CSA2 所有权、32×32 MXFP8 ABI、
384 路专家、两个 Engram 表、视觉塔和三个 DSpark 阶段。公开的标量文本生成会执行完整的
逐层流式 CED/CSA2/Engram/mHC 基础路径，包括训练得到的压缩缓存、精确的有界 Engram
读取、top-6 MoE 和流式 LM head。独立的 PyTorch 真实权重 BOS 预言机会验证全部 40 层
的路由和最终 top-16 logits。提示词预填充仍为逐 token；视觉与 DSpark 仍仅进行
schema 验证。

**Kimi-K3。** 文本运行时以流式方式执行 92 个稀疏层中的 896 路原生 MXFP4 路由专家，
使紧凑的 BF16 主干受层边界约束，并保留循环 KDA 与压缩 MLA 状态。XTML 由可信结构
片段组装，同时转义不可信内容。生成在 `<|end_of_msg|>`（163586）停止，而不是在
tokenizer 元数据的 `[EOS]` token（163585）停止。

**Qwen3.8。** 适配器固定到 `model_type="qwen3_5_moe_text"` 以及已发布的
`Qwen/Qwen3.8-2.4T-A95B-FP8` ABI。它实现 92 层基础文本 forward、top-10/512
路由专家、门控共享专家、final norm 与 LM head。MTP 张量会进行 schema 验证，但不属于
基础自回归 forward。官方检查点已经通过完整头部门禁和独立的 92 层加完整 LM head
预言机。微型 FP32/BF16 图 fixture 和有界的真实 E4M3 专家 payload 预言机还固定了
dtype 边界与量化投影数学。

**Hy4。** 适配器固定到 `model_type="hy_v4"` 和 `HYV4ForCausalLM`。它验证完整的
78 层基础模型与原生 MTP payload，无需转换即可读取 ModelOpt MXFP8 权重，并从合并的
256 专家张量中切出单个路由专家。tokenizer/chat、identity-HC 参考数学、
header/probe/preflight 路径与有界内存规划器均已接通。检查点声明支持 1,048,576
个位置，但在 Gated DSA/IndexCache 和激活量化语义通过独立 logits 预言机前，精确执行
仍限制在 2,048。

## 检查点配置

Hugging Face CLI 仅用于下载检查点；它不是推理依赖：

```bash
python3 -m pip install --upgrade huggingface_hub
```

<details>
<summary><b>GLM-5.2 转换检查点</b></summary>

兼容的转换检查点为
[`mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp`](https://huggingface.co/mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp)。

```bash
hf download mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp --dry-run

hf download mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp \
  --local-dir /path/to/glm-5.2-colibri
```

</details>

<details>
<summary><b>Hy4-preview-FP8 原生检查点</b></summary>

```bash
hf download tencent/Hy4-preview-FP8 --local-dir /path/to/hy4-preview-fp8

./target/release/urb inspect /path/to/hy4-preview-fp8
./target/release/urb preflight /path/to/hy4-preview-fp8 --context 2048 --expert-slots 1
```

严格版本门禁要求全部 130 个分片。

</details>

<details>
<summary><b>DeepSeek-V4/V4.1 与 Kimi-K3 原生检查点</b></summary>

原生分片、`config.json`、tokenizer 和索引文件下载完成后，让 Urbilateria 指向模型
发布目录。无需进行 Transformers/Python 转换。

Kimi 的 96 个分片尚未全部到达时即可开始检查传输结果。以点开头的 rsync 临时文件会被
忽略；每个由已完成、精确的 `.safetensors` 文件完整表示的解码器层都会得到验证：

```bash
./target/release/urb preflight /path/to/Kimi-K3 --partial
```

</details>

<details>
<summary><b>Qwen3.8 元数据优先配置</b></summary>

在下载约 2.5 TB 的张量 payload 前，先获取配置、tokenizer 和 manifest 开发所需的
五个小型发布文件：

```bash
hf download Qwen/Qwen3.8-2.4T-A95B-FP8 \
  --revision d2dc35658bcf77e66643428cb52e774cc3b5bd29 \
  --include config.json generation_config.json tokenizer.json tokenizer_config.json model.safetensors.index.json \
  --local-dir /path/to/qwen3.8-metadata
```

全部 213 个分片到达后，执行 preflight 并运行纯文本、始终思考的路径：

```bash
./target/release/urb preflight /path/to/Qwen3.8-2.4T-A95B-FP8 --context 64 --expert-slots 0
./target/release/urb generate /path/to/Qwen3.8-2.4T-A95B-FP8 \
  --prompt "Hello" --ram-gib 4 --max-new-tokens 1 --allow-large-model
```

4 GiB 示例高于 64-token 上下文下实测约 2.04 GiB 的标量运行时峰值，但原生
2.496 TB 检查点仍会让短生成受限于存储速度，并且非常缓慢。

</details>

大型检查点需要充足的本地存储空间和 NVMe 带宽。重新运行相同的 `hf download`
命令会从已完成的工作继续。仅当 Hugging Face 要求认证时才使用 `hf auth login`。

## 验证

Urbilateria 将“文件看起来合理”与“模型生成正确 logits”区分开来。每一层置信度都有
自己的门禁：

```text
1  配置门禁       精确几何、token ID、量化契约
2  头部门禁       张量名称、dtype、形状、字节长度、分片分配
3  微型预言机     对照独立 fixture 的确定性端到端计算图
4  Payload 门禁   有界读取精确解码真实量化字节
5  运行时门禁     在内存契约下执行完整真实层/token
6  输出门禁       独立完整 logits 与已知 token 延续
```

| 家族 | 仓库中呈现的最强已完成证据 | 剩余公开门禁 |
| --- | --- | --- |
| GLM-5.2 | 微型完整模型预言机；真实转换检查点 token 回归 | 吞吐量优化 |
| DeepSeek-V4 | 独立微型预言机；真实层/token 与 tokenizer 回归 | 持续性能工作 |
| DeepSeek-V4.1 | 精确头部、原生 MX payload、完整 Engram/CSA2/mHC 基础 forward 与公开生成、真实 40 层 BOS 路由与 top-16-logit 预言机 | 按层预填充、视觉与 DSpark 执行 |
| Kimi-K3 | 独立全栈 logits 一致性和已知的 `Paris` 延续 | 持续性能和逐版本验证 |
| Qwen3.8 | 独立微型 FP32/BF16 图、真实 FP8 专家 payload 和真实 92 层完整 logits 预言机 | 持续性能和发布版本多 token 验证 |
| Hy4 | 上游 iHC/Gated-MLA 语义；精确 schema/MXFP8/专家 payload 门禁；真实 78 层 token 与公开 CLI 生成 smoke | 独立完整 logits 一致性和超过 2,048 token 的 IndexCache 执行 |

对于 Qwen3.8 发布版 token `Hello`，独立 PyTorch streamer 与 Rust 运行时选择了相同的
argmax 和相同的 top 20 token。在全部 248,320 个 logits 上，余弦相似度为
`0.9999224`，平均绝对误差为 `0.028373`，最大绝对误差为 `0.1875`。由于两个后端
使用不同的 GEMM 归约树，有 23 层在 top-10 路由边界上交换了 920 个专家位置中的 24 个；
门禁将聚合替换数量限制为 32。隔离的真实 FP8 专家门禁仍精确到一个 BF16 ULP。

无模型测试套件速度很快，也不需要检查点：

```bash
cargo test --all-targets --locked
```

当前结果：**338 项通过，0 项失败**；真实检查点测试会被显式忽略，除非提供对应的
模型目录。

<details>
<summary><b>运行有代表性的真实检查点门禁</b></summary>

```bash
URB_DEEPSEEK_V4_DIR=/path/to/DeepSeek-V4-Flash-0731 \
  cargo test --release --locked --test deepseek_v4_real -- --ignored

URB_GLM52_DIR=/path/to/colibri-glm-5.2 \
  cargo test --release --locked --test glm_real -- --ignored

KIMI_K3_MODEL_DIR=/path/to/Kimi-K3 \
KIMI_K3_REFERENCE_LOGITS=/path/to/kimi-k3-in-c/tests/fixtures/golden/ref_logits.json \
  cargo test --release --locked --test kimi_k3_real \
  real_checkpoint_logits_match_the_independent_pytorch_golden -- --ignored --exact --nocapture

KIMI_K3_MODEL_DIR=/path/to/Kimi-K3 \
  cargo test --release --locked --test kimi_k3_real \
  real_checkpoint_completes_france_with_paris -- --ignored --exact --nocapture

HY4_MODEL_DIR=/path/to/hy4-preview-fp8 \
  cargo test --release --locked --test hy4_real -- --ignored --nocapture
```

</details>

## CLI 参考

| 命令 | 用途 | 模型支持 |
| --- | --- | --- |
| `inspect` | 构建静态参数、量化、张量与路由透视 | 全部六个适配器 |
| `plan` | 估算常驻、KV、暂存空间与专家缓存预算 | GLM、DeepSeek、Kimi、Hy4 |
| `preflight` | 不读取 payload，验证精确运行时张量与内存 | 全部六个适配器 |
| `list` | 搜索精确张量名称 | 所有 safetensors 检查点 |
| `probe` | 在不加载检查点的情况下采样一个张量 | 所有 safetensors 检查点 |
| `tokenize` | 编码原始文本或模型原生聊天轮次 | 全部六个适配器 |
| `decode` | 解码逗号分隔的 token ID | 全部六个适配器 |
| `generate` | 运行经过 RAM 规划的贪心生成 | 全部六个适配器 |
| `explain` | 输出 token 路径与张量几何 | 全部六个适配器 |

所有支持 JSON 的分析命令都可以为自动化输出 JSON。运行 `urb help` 查看准确的参数
和默认值。

### 提示词安全

普通用户文本通过模型原生聊天协议渲染。在模型家族允许的情况下，使用
`--no-thinking` 禁用推理。`--raw-prompt` 是一个仅供可信输入使用的逃生舱：
它只接受以模型原生 BOS/协议前缀开头的完整渲染提示词；裸文本会被拒绝，否则可能立即
生成 EOS 或不相关输出。

Kimi-K3 明确划定了信任边界：协议标记只由带类型的结构片段发出，用户、assistant
和工具内容则保持为普通的已转义片段。Qwen3.8 的官方模板始终要求思考，因此
`--no-thinking` 对该家族无效。

## 性能分析

人类可读的 profile 输出到 stderr。具有稳定 schema 的 JSON 只会写入请求的文件，
绝不会混入流式模型输出。计时为 inclusive，因此嵌套阶段会有意重叠。Linux 上的
profile schema v2 还会每 100 ms 采样进程 CPU、RSS/swap、缺页、线程数、系统内存和
进程 I/O，并强制记录结束样本。`storage_read_bytes`/`storage_write_bytes` 是 Linux
存储层对该进程的记账；`read_char_bytes`/`write_char_bytes` 包括 page cache 流量。
两者都不能与阶段的逻辑 checkpoint payload 或整块 SSD 利用率混为一谈。

```bash
./target/release/urb generate /path/to/model \
  --prompt "Hello" --ram-gib 32 --max-new-tokens 1 --no-thinking \
  --allow-large-model --threads 20 --profile \
  --profile-json profile-20.json --profile-trace profile-20.trace.json
```

`--profile-trace` 会开启有界的逐 span 记录并输出 Chrome Trace Event JSON。可以直接用
[Perfetto UI](https://ui.perfetto.dev/) 打开，也可以在本地打开
[`tools/profile_viewer.html`](tools/profile_viewer.html)，把两份 JSON 一起拖入。这个零依赖
查看器支持阶段排名、资源曲线、线程筛选、搜索、缩放/平移和按时间排列的火焰图；未指定
该参数时不会承担逐事件记录开销。
DeepSeek-V4 的 trace span 会在语义明确的边界记录 token 位置/ID、layer ID、expert ID、
批处理开始前的缓存驻留状态、batch 大小，以及稳定的逐层 flow ID。

使用 `--threads 1` 作为串行基线。不指定 `--threads` 时，持久 worker 池使用平台
可用的逻辑并行度。在 SMT 或混合核心 CPU 上，更多线程并不总是更快，因此请在相同的
提示词和缓存状态下比较多个值。

一个无需模型的 release benchmark 会运行相同的 F32 kernel 和 profiler：

```bash
URB_BENCH_THREADS=1 cargo run --release --example profile_matvec --locked
URB_BENCH_THREADS=20 cargo run --release --example profile_matvec --locked
```

它会报告维度、请求和实际 worker 数、延迟百分位数、GMAC/s、checksum 以及完整
profile。`URB_BENCH_ROWS`、`URB_BENCH_COLS` 和 `URB_BENCH_ITERATIONS`
可以覆盖默认值。

<details>
<summary><b>真实检查点性能测试工具</b></summary>

被忽略的 `inference_performance` 测试针对 GLM-5.2、DeepSeek-V4 和 Kimi-K3
测量相同的固定长度贪心工作负载。报告包括引擎 TTFT、观测到的和纯模型 decode
tokens/s、设置阶段、专家缓存遥测、逻辑注意力/KV 增长、规划内存以及 inclusive
profile。

在独立的 release 测试进程中运行每个模型，避免持久 worker 池、内存压力和 page cache
状态相互重叠。本机路径和默认值可以放在被 Git 忽略的 `.env.performance` 文件中：

```bash
./tests/run_inference_performance.sh
```

报告写入 `target/perf/`。只有在固定主机上使用相同检查点、提示词、线程数和专家槽位
设置时，报告之间才具有可比性。

</details>

## 项目结构

| 路径 | 职责 |
| --- | --- |
| `src/analysis/` | 检查点报告、张量分类、探针、trace 与资源规划 |
| `src/storage/` | 严格 safetensors 索引、有界 payload 读取与原生权重 loader |
| `src/math/` | 可读的标量量化、路由、MXFP 与数值参考操作 |
| `src/models/` | 家族私有的配置、schema、提示词、注意力、MoE、权重与运行时代码 |
| `src/runtime/` | 模型无关的运行时契约和确定性专家缓存机制 |
| `src/generation.rs` | 后端无关的自回归 token 循环与停止语义 |
| `src/profiling.rs` | 可选的 inclusive 阶段指标和稳定 JSON 报告 |
| `tests/` | 真实检查点正确性门禁与跨模型性能测试工具 |
| `tools/` | 验证期间使用的独立 Python 预言机生成工具，不参与推理 |

共享边界很小是刻意为之。新的模型家族应当复用存储、生成、性能分析和运行时契约，
而不应强迫一种架构的张量 ABI 或提示词语义适配另一种架构。

## 开发

```bash
cargo fmt --all --check
cargo test --all-targets --locked
cargo clippy --all-targets --locked -- -D warnings
cargo build --release --locked
```

行为变更应包含有针对性的测试。优化 kernel 必须继续与可读的标量参考实现进行核对；
在数值契约明确之前，更快的路径不能视为完成。

## 当前限制

- 目前是带并行标量 matvec kernel 的纯 CPU 正确性运行时；大模型的持续生成仍需要
  SIMD/fused kernel 和更多 I/O 优化。
- 没有服务器 API、Web UI、CUDA 或 Metal 后端。
- GLM 路径支持转换后的 Colibri 容器，而不是任意官方 FP8 版本。
- DeepSeek-V4/V4.1 DSpark 推测解码会进行 schema 验证，但已被禁用；V4.1 视觉塔也仅
  进行 schema 验证，且其提示词预填充仍为逐 token。
- Kimi-K3 仅支持文本生成；MoonViT 和 projector 张量会被验证，但不会执行。
- 与已测试版本不同的 Kimi 检查点必须单独进行 logits 与性能验证，才能继承输出质量声明。
- Qwen3.8 MTP 张量会进行 schema 验证，但推测解码被禁用；标量生成路径适合正确性研究，
  不适合交互式吞吐量。
- Hy4 仅在总 token 数不超过 2,048 时精确生成；更长上下文的 IndexCache 执行和独立
  完整 logits 一致性仍是尚未完成的验证里程碑。
- 有意拒绝任意 Transformers 架构和量化布局。

## 名称由来

**Colibri + Ferris = Urbilateria。**

蜂鸟属于脊索动物和后口动物；螃蟹属于节肢动物和原口动物。*Urbilateria* 通常指这两条
谱系分化之前假想的最后共同两侧对称动物祖先。它不是一种已知的化石物种，其真实形态
仍有争议。

这个名称结合了项目的两个灵感来源：Colibri 对低内存推理的追求，以及 Ferris 所代表的
Rust 生态系统。Urbilateria 在一个独立、面向学习的实现中将这些思想结合起来。

## 参与贡献

欢迎提交 issue、实验和范围集中的 pull request。请保持参考行为可读，为行为变更加入
测试，并记录任何新的检查点或数值假设。

## 致谢

- [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2-FP8)：提供模型架构
- [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)：提供发布配置和独立推理参考
- [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)：提供原生 CED/CSA2/Engram/mHC 模型与检查点
- [Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3)：提供发布模型、tokenizer 和架构规范
- [Qwen3.8-2.4T-A95B-FP8](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B-FP8)：提供固定到发布版本的混合注意力 MoE 模型
- [Hy4-preview-FP8](https://huggingface.co/tencent/Hy4-preview-FP8)：提供固定到发布版本的 iHC/Gated-DSA MoE 模型
- [kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c)：提供独立的 Apache-2.0 行为参考，用于交叉核对标量公式和检查点约定
- [Colibri](https://github.com/JustVugg/colibri)：启发低内存专家流式加载
- [Rabbit](https://github.com/ferrumox/rabbit)：启发面向 Rust 学习的推理路径

Urbilateria 是独立实现，不包含上述项目的代码或模型权重。

## 许可证

Urbilateria 使用 [MIT 许可证](LICENSE)。
