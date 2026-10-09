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

成因：**已实测确定，见 [`arch_levers.md`](arch_levers.md)**。一句话：两个融合核
（flash / memory-efficient）都被 **GQA 的头数不匹配**（q 16 头 / kv 4 头）挡掉，
SDPA 只能落到 math 路径，而 math 路径会实体化**三份** N×N
（fp32 scores + fp32 softmax 输出 + fp16 转换）。
实测：N=4096 / 16 头，math 峰值 2560 MiB，内存高效核只要 96 MiB。
2560 MiB @ 4096 ≈ 160 B/tok²，与这里的 170.7 B/tok² 对得上。

（本节先前写的是"sm_75 上 flash 与 memory-efficient 后端都需要 sm_80+"。
**那句话是错的**：memory-efficient 在 sm_75 上可用，我当时的测试用例用了全 16 头，
所以它能跑 —— 测试用例没复现真实形状。已更正。）

注：24 个线性注意力层是 chunked Gated DeltaNet（`torch_chunk_gated_delta_rule`，
chunk=64），**是 O(N) 不是 N²**，所以不是它的锅 —— 这条是查源码得到的。

**在当前卡上无法用一个 wrapper 绕开**：实测把 K/V 扩成 16 头 + 去掉 `enable_gqa`
之后，显存曲线与不加补丁**逐位相同**，关掉 math 后端时仍报 `No available kernel`。
所以下面第 6 节的阶梯是**在当前约束下**的建议；想真正抬高上限，需要换注意力实现 /
推理引擎，或换卡（见 `arch_levers.md`）。

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

不要一次拉到上限：12,238 是**显存**上限，而每轮 prefill 的 N² 算力是真实成本。

换算用的实测系数（全部来自上面）：固定开销 2089、1 个 pair 180、1 帧图 66
（`min_pixels=4096` 时是 27）。`MAX_HISTORY_MESSAGES` 数的是**消息**，1 pair = 2 条。

| 档 | 配置 | N (token) | 显存增量 | 总显存 | 32 GiB 剩余 | gen_s |
|---|---|---|---|---|---|---|
| 现状 | `MSG=4`、`IMG_KEEP=1` | 2,269 | 1.08 GiB | 8.42 GiB | 23.6 GiB | 2.2 s |
| 保守 | `MSG=16`、`IMG_KEEP=2` | 3,661 | 2.38 GiB | 9.72 GiB | 22.3 GiB | 2.7 s |
| **推荐** | `MSG=32`、`IMG_KEEP=4` | **5,233** | 4.60 GiB | 11.94 GiB | 20.1 GiB | 3.5 s |
| 激进 | `MSG=64`、`IMG_KEEP=6` | 8,245 | 11.06 GiB | 18.40 GiB | 13.6 GiB | 5.9 s |
| 上限 | 56 pair ≈ 107 条消息 | 12,238 | 24.06 GiB | 31.4 GiB | ~0.6 GiB | 10.8 s |

推荐档只用到 32 GiB 的 20 GiB 余量里的 4.6 GiB —— 这与"尽量用满"相反，是故意的：
episode 里 token 数会抖动（工具结果长度差异很大），而 OOM 是**一次就毁掉整场**的失败，
并且 12k 档的 gen_s 已经接近 11 s，60 轮就是 11 分钟纯 prefill。

顺带一个比例上的事实：现状 `MAX_HISTORY_MESSAGES=4` 只值 **360 token**，
占 2.4k 窗口的 15%。**不是没地方放历史，是从来没给过历史。**

## 7. 更根本的做法

只加历史是把 `MAX_HISTORY_MESSAGES` 从 4 调大，历史仍是线性堆积，
而 cost 是 N²。DESIGN.md M2 里写的"两级压缩"在 32 GiB 上才真正可行：
- 软压缩（已有）：图像淘汰；
- 硬压缩（待做）：把旧历史总结成一段文字，**token 数不随步数线性增长**。

P0f 已经证明"往工具结果里加提示"收益为 0。跨阶段因果缺失是"窗口太短"的症状，
而治法是**压缩**（信息密度），不是**加长**（信息量）——否则 60 轮的 episode 迟早会
撞回同一个天花板，只是在天花板更高的地方撞。
