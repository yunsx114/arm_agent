# 架构层面的优化空间：实测了什么、推翻了什么

探针：`scripts/probe_arch_levers.py`、`scripts/probe_vram_vs_context.py`（`GQA_EXPAND=1`）
日志：`outputs/logs/arch_levers.log`、`vram_gqa.log`、`vram_gqa2.log`
环境：RTX 2080 Ti 10.57 GiB，sm_75，torch 2.8.0+cu128，transformers 5.12.1

本文只写**测到的**。凡是没有测到的（比如 5090 上的行为）单独标为"预期"。

---

## 1. 结论：170.7 B/tok² 的真正成因（推翻了我上一轮的说法）

上一轮我写的是"sm_75 上 SDPA 没有融合核，所以落到 math 路径"。**这句话是错的**，
实测：

```
can_use_FLASH_ATTENTION     : False   ("Flash attention only supports sm80..sm121, attempting sm 7.5")
can_use_EFFICIENT_ATTENTION : True
```

**内存高效核（memory-efficient）在 sm_75 上是可用的**。我当时的测试用例用的是
q/k/v 全 16 头，所以它能跑；于是我得出了"sm_75 有融合核"——但真实模型不是这个形状。

真正的阻断因素是 **GQA**：`num_attention_heads=16` / `num_key_value_heads=4`。
两个融合核对 dense 输入都要求 q/k/v 头数相同。transformers 的
`sdpa_attention.py` 已经为此传了 `enable_gqa=True`，而实测 `enable_gqa=True`
**本身就会让融合核拒绝**（我的内核测试里 `enable_gqa=True` + EFFICIENT 直接
`No available kernel`）。

于是 SDPA 只能走 math 路径，而 math 路径会实体化**三份** N×N：

| 后端 | N=4096, 16 头, fp16 输入的峰值增量 | 说明 |
|---|---|---|
| MATH | **2560 MiB** | = fp32 scores 1024 + fp32 softmax 输出 1024 + fp16 转换 512，逐项吻合 |
| EFFICIENT（无 mask、等头数） | **96 MiB** | 就是输入本身，没有 N×N |
| FLASH | 不可用 | sm_75 < sm80 |

2560 MiB @ 4096 token ≈ **160 B/tok²**，与独立测出来的显存定律 170.7 B/tok² 对得上。
**整条因果链闭合了。**

## 2. 尝试过、但没成的修法（重要：不要重复走）

"既然只是头数不匹配，那把 K/V 扩成 16 头不就行了？" —— 试了，**没用**：

```
GQA_EXPAND=1（K/V 扩成 16 头 + 去掉 enable_gqa）
  实测曲线与不加补丁时**逐位相同**：2629 tok -> 1400 MiB，4069 tok -> 2976 MiB
  在关掉 math 后端时仍然 RuntimeError: No available kernel
诊断：补丁确实被调用了（sdpa calls 1, GQA calls 1），但融合核依然拒绝
```

所以阻断因素**不只是头数**，还包括真实调用签名里的东西（attn_mask / is_causal / scale）。
在这张卡上无法用一个 wrapper 绕开 —— 这和"加上补丁就能拿 5 倍上下文"是两回事，
我一度以为前者成立，实测否掉了。

结论：**在 sm_75 上，这个 N² 开销是不可绕过项**，要动就得换注意力实现或换推理引擎。

## 3. 7.34 GiB 的权重里有一半是没被量化的两个矩阵

```
allocated after load : 7.35 GiB
by dtype:  bfloat16 3.800 GiB   uint8 3.432 GiB

model.language_model.layers          3.224 GiB
model.language_model.embed_tokens    1.895 GiB   <-- bf16，未量化
lm_head.weight                       1.895 GiB   <-- bf16，未量化（tie_word_embeddings=False）
model.visual.*                       0.219 GiB
```

- `vocab_size=248320`，embed 与 lm_head 各 248320×4096 参数 ≈ **2.03 GiB / 个（bf16）**。
- bitsandbytes 的 4bit 默认不量化这两个，所以"4bit 模型"实际是
  `7.35 GiB × 8 / 9e9 ≈ 7.0 bit/参数`。
- 可省：把这两个也量化到 8bit → 各 0.95 GiB（**省 1.9 GiB**）；4bit → 各 0.47 GiB（**省 2.8 GiB**）。
  按第 1 节的定律，省 2.8 GiB ≈ 上限从 3,870 抬到 `sqrt(2.63+2.8 GiB / 170.7B) ≈ 5,280` token。
  **这是在当前卡上真实可拿的收益**，且与注意力内核无关。

## 4. 5090（32 GB, sm_120）能改什么

必须区分"确定的"和"预期的"：

**确定（由上面第 1 节的实测消息直接推出）**
- `Flash attention supports [sm80, sm121]` → **sm_120 在范围内**，flash 可用。
- flash 内核原生支持 GQA，不需要 `enable_gqa` 的 hack，也**完全不会实体化 N×N**。
- 于是 170.7 B/tok² 应该退回到接近理论的 **32~40 B/tok²**。
- **预期上限**：`32 GiB - 7.34 GiB ≈ 24.7 GiB`，配 0.6 GiB 余量 → `sqrt(24.1GiB/40B) ≈ 25,000 token`。
  对比当前 3,870，约 **6.5 倍**。（这是外推，不是实测；60 B/tok² 时约 20,000。）

**确定**
- 32 GB 装得下 **bf16 全精度**（~20 GiB 权重含 lm_head），可以完全不用 4bit。
  这消掉了量化误差 —— 对一个要输出厘米级坐标的模型，这是值得单独测的问题，不是纯性能问题。

**需要实测才能定的**
- sm_120 是消费级 Blackwell，**很多预编译内核还没覆盖**。Python 侧的
  `torch.cuda.get_arch_list()` 必须包含 `sm_120`，否则会在加载时才炸。
- bitsandbytes 在 sm_120 上的 4bit 是否有快速路径。

## 5. 与显卡无关的优化（这些在当前卡上就能做）

按预期收益排序，全部需要实测验证：

1. **换推理引擎**：vLLM / SGLang 的 paged attention + chunked prefill
   从根上避免 N×N 实体化，同时自动做 prefix caching。
   这是唯一能同时解掉"上限"和"每轮重算"两件事的改法。
   代价：`harness.py` 里的 `model.chat()` 要换成引擎的 client（工具调用协议要重接）。
2. **prefix / KV 复用**：系统提示 + 11 个工具 schema（2089 token）**每轮都重新 prefill**。
   在 agent 循环里历史是单调增长的，理论上整个旧前缀都可复用，每轮只需算新增的 ~180 token。
   注意：当前 `compact_history` 会**丢弃**旧消息，一丢就让前缀失效 ——
   即"压缩历史"和"复用前缀"在实现上是对立的，必须先决定要哪个。
3. **量化 embed/lm_head**（第 3 节），省 1.9~2.8 GiB。改动小、无副作用。
4. **真正缩小图片**：`min_pixels=4096` 让一帧从 66 token 降到 27（见 `context_budget.md`）。
   但一帧只省 39 token，收益小。
5. **换更小的模型**：这是设计选择不是优化，但 9B 在 4bit 下要 7.35 GiB，
   而任务只是"看画面 + 选工具"，可能不值得。

## 6. 一句话总结

当前 10.57 GiB 上的天花板（~3,870 token）**主要不是显存不够，而是 GQA + 无 flash 导致
SDPA 落到 math 路径、把 N×N 实体化成三份**。
- 当前卡上能拿的：量化 embed/lm_head（省 2.8 GiB，上限 ≈ +36%）。
- 换 5090 能拿的：flash + GQA 原生 → B 从 ~170 降到 ~32~40，上限 ≈ 6 倍（外推）。
- 不用换卡也能拿的：换推理引擎 / 复用前缀 —— 但需要改 `harness.py` 与推理栈的接口。
