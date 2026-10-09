# 上下文预算：实测（不是公式推导）

测量脚本：`scripts/probe_vram_vs_context.py`（GPU 节点，`bash scripts/run_vram_probe_on_gpu.sh`）
数据：`outputs/logs/vram_vs_context.log`  ·  环境：RTX 2080 Ti 10.57 GiB，sm_75

## 1. 结论先行

| 卡 | 权重后可用 | **N 上限（token）** | 历史 pair / 消息 | 单次 generate |
|---|---|---|---|---|
| 10.57 GiB（现状） | 2.63 GiB | **3,870** | 10 / 20 | 2.8 s |
| 16 GiB | 8.06 GiB | 7,010 | 27 / 55 | 4.8 s |
| 24 GiB | 16.06 GiB | 9,974 | 44 / 88 | 7.8 s |
| **32 GiB** | 24.06 GiB | **12,240** | **56 / 113** | **10.8 s** |
| 48 GiB | 40.06 GiB | 15,826 | 76 / 153 | 16.7 s |

（"历史 pair" = 1 次工具调用 + 1 条工具结果 = `MAX_HISTORY_MESSAGES` 计数的 2 条消息。
固定开销 2089 token = plain 系统提示 + 11 个工具 schema + 首条用户消息。）

## 2. 实测的定律（两参数最小二乘，5 点最大偏差 0.6%）

```
峰值显存增量(N) = 256 MiB + 170.7 B × N²
单次 generate(N) = 1.88 s  + 59.3e-9 s × N²
```

实测点：

| N (token) | 峰值增量 | 拟合 | 偏差 |
|---|---|---|---|
| 2,449 | 1,225 MiB | 1,232 MiB | +0.6% |
| 2,809 | 1,543 MiB | 1,540 MiB | −0.2% |
| 3,349 | 2,089 MiB | 2,081 MiB | −0.4% |
| 3,709 | 2,498 MiB | 2,495 MiB | −0.1% |
| 4,069 | 2,944 MiB | 2,951 MiB | +0.2% |

10.57 GiB 下 4,249 token 实测 OOM，与拟合给出的 3,870 同量级（拟合略保守）。
自检全部通过：token 数严格单调、峰值严格单调、**增长确实是二次的**（B/tok² 在上半段只差 5%）。

## 3. 重要更正：`llm.py` 打印的 `attn_matrix≈` 低估了 5.3 倍

```
llm.py:  matrix_mib = n_tok² × 16 heads × 2 B   ← 假设 fp16 且只算一个矩阵
实测:    170.7 B/tok²  =  32 B/tok² × 5.3
```

`attn_matrix≈176MiB`（ctx=2400）读起来像"还有 3 GiB 富余"，实际当时已经用掉 1.2 GiB。
这是把**推导量当成测量量**的典型后果：它是按公式算出来的预测，从来不是实测值。

成因（**假设**，尚未独立验证）：sm_75 上 SDPA 的 flash / memory-efficient 后端都需要
sm_80+，落到 math 回退路径后会**以 fp32 实体化** score 与 softmax 输出，再加一次转 fp16。
16 heads × 4 B（fp32）× 2~3 份 ≈ 128~192 B/tok²，与实测 170.7 吻合。
注：24 个线性注意力层是 chunked Gated DeltaNet（`torch_chunk_gated_delta_rule`，
chunk=64），**是 O(N) 不是 N²**，所以不是它的锅。

如果这条假设成立，那么换一个不落 fp32 的注意力实现可以把 B 从 170.7 降到 ~32，
**上下文上限提高约 5 倍**（32 GiB 上从 12k 到 ~28k）。这是"想继续放宽时该动的地方"，
但在被测出来之前只能算线索。

## 4. 另一个实测更正：`image_max_side` 根本没有省 token

```
默认 preprocessor_config.json: shortest_edge = 65536  (= 256×256)

输入  64px -> grid [1,16,16] -> 64 image tokens
输入 160px -> grid [1,16,16] -> 64 tokens      ← 被上采样回 256
输入 224px -> grid [1,16,16] -> 64 tokens
输入 256px -> grid [1,16,16] -> 64 tokens
输入 384px -> grid [1,24,24] -> 144 tokens
输入 512px -> grid [1,32,32] -> 256 tokens
```

`llm.py` 里"160px 比 256px 便宜 2.6 倍"这句话**没有实测支持**：任何 ≤256px 的图都被
processor 按 `shortest_edge` 上采样回 256×256，token 数一样是 64。

真正能省的接法（已实测）：

```python
processor(images=..., size={"shortest_edge": 4096, "longest_edge": 262144})
160px -> grid [1,10,10] -> 25 tokens   # 2.56x，和注释里声称的一致，但要有这行才生效
```

已加进 `QwenAgentModel(image_min_pixels=...)`。

代价很小：一帧 64→25 token，省 39 token；就算留 8 帧也只省 312 token。
**图片不是主要开销，N² 项才是**，所以不要把注意力放在这里。

## 5. 边际成本（直接换算"能多留几步历史"）

| 项 | 实测 |
|---|---|
| 1 个历史 pair（= 2 条消息：1 次调用 + 1 条结果） | **180 token** |
| 固定开销（系统提示 + 11 工具 schema + 首条用户消息） | 2089 token |
| 1 帧图（当前接法） | 66 token = 64 图 token + 2 分隔符 |
| 1 帧图（`min_pixels=4096`） | 27 token |

当前 `MAX_HISTORY_MESSAGES=4` 只值 360 token —— 占 2.4k 窗口的 15%。
**不是没地方放历史，是从来没给过历史。**

## 6. 建议的放宽阶梯（32 GiB）

不要一次拉到上限：12,240 是**显存**上限，而每轮 prefill 的 N² 算力是真实成本，
且在 Turing 上 N² 项会随 N² 涨（拟合给出 12k 时单次 generate ≈ 10.8 s，现状 2.4k 是 2.8 s）。

| 档 | 配置 | 约合 token | 显存占用 | 说明 |
|---|---|---|---|---|
| 保守 | `MAX_HISTORY_MESSAGES=16`、`HISTORY_IMAGE_KEEP=2` | ~5.1k | 3.0 GiB | 覆盖"下降→挤压→掉落"这种跨 3~4 步的因果 |
| 推荐 | `MAX_HISTORY_MESSAGES=32`、`HISTORY_IMAGE_KEEP=4` | ~7.4k | 4.2 GiB | 整个 episode 到中段基本都在窗口里 |
| 激进 | `MAX_HISTORY_MESSAGES=64`、`HISTORY_IMAGE_KEEP=6` | ~11.8k | 8.0 GiB | 接近上限，留 2/3 显存余量给波动 |

推荐档只用到 32 GiB 的约 4.2/24 GiB，离上限还很远 —— 这与"尽量用满"相反，是故意的：
episode 里 token 数会抖动（工具结果长度不一），而 OOM 是**一次就毁掉整场**的失败。

## 7. 更根本的做法

只加历史是把 `MAX_HISTORY_MESSAGES` 从 4 调大，历史仍是线性堆积，
而 cost 是 N²。DESIGN.md M2 里写的"两级压缩"在 32 GiB 上才真正可行：
- 软压缩（已有）：图像淘汰；
- 硬压缩（待做）：把旧历史总结成一段文字，**token 数不随步数线性增长**。

P0f 已经证明"往工具结果里加提示"收益为 0。跨阶段因果缺失是"窗口太短"的症状，
而治法是**压缩**（信息密度），不是**加长**（信息量）——否则 60 轮的 episode 迟早会
撞回同一个天花板，只是在天花板更高的地方撞。
