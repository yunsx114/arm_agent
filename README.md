# arm_agent

让纯多模态 LLM（Qwen3.5-9B，4bit）通过工具调用控制机械臂，完成 LIBERO 仿真中的具身任务。

```
┌─ 登录节点 ────────────────┐        ┌─ gpu026 (RTX 2080 Ti) ───────┐
│  libero conda env (py3.9) │        │  qwen35 conda env (py3.11)   │
│  LIBERO + MuJoCo 2.3.7    │        │  Qwen3.5-9B 4bit + harness   │
│  SimServer ───────────────┼── NFS ─┤  SimClient → LLM tool loop   │
│  (渲染 90% 开销已优化)     │  IPC   │                              │
└───────────────────────────┘        └──────────────────────────────┘
```

**为什么是两个节点**（实测，不能改）：gpu026 的 CPU 指令集被裁剪（无 SSE4.2），`import mujoco` 直接 SIGILL；两节点间 TCP 不通（gpu026 在孤立子网），但 `/lab` NFS 共享，读/写延迟 ~1ms。

## 快速开始

```bash
# 1) 仿真环境（一次性）
bash scripts/setup_sim_env.sh            # 创建 libero env（robosuite 1.4.0 + mujoco 2.3.7）

# 2) 无 GPU 全链路自检（约 2 分钟，推荐先跑）
bash scripts/simenv.sh scripts/smoke_libero_e1.py            # 仿真链路
bash scripts/simenv.sh scripts/probe_action_semantics.py     # 动作语义（10 项断言）
bash scripts/simenv.sh scripts/probe_ipc_roundtrip.py        # NFS IPC 协议
bash scripts/simenv.sh scripts/probe_harness_dryrun.py       # 完整闭环（脚本策略替 LLM）

# 3) 真实 LLM episode（双节点自动编排）
bash scripts/run_m1_on_gpu.sh 1          # 1 个 episode（约 10-30 分钟）
bash scripts/simenv.sh scripts/analyze_episode.py            # 复盘最新 transcript
```

### 录制视频（用于 debug 与展示）

```bash
RECORD=1 bash scripts/run_m1_on_gpu.sh 1        # 每个 episode 生成一个 mp4
RECORD=1 MAXTURNS=6 bash scripts/run_m1_on_gpu.sh 1   # 短冒烟
```

画面布局：左侧上=agentview、下=wrist 双相机；右侧为本轮模型输出文字、解析出的工具调用、工具执行结果（含实测数字）与状态。
**成功 / 失败 / OOM / 崩溃都会录制**：录制器每轮先把合成帧写成 PNG（硬崩溃也不丢），finalize 时才编码 mp4，成功编码后清理 PNG。
若进程被强杀导致没有 mp4，可用帧目录补编码：

```bash
bash scripts/agentenv.sh scripts/encode_frames.py outputs/videos/<run>/frames
```

中文字体（文泉驿正黑，从阿里镜像获取）已放在 `assets/fonts/`。

## 结构

```
src/arm_agent/
├── contracts.py           # 动作/观测/IPC 的共享常量（唯一真源）
├── cli.py                 # sim-server | ipc-ping | run
├── sim/                   # ← libero env 运行（登录节点）
│   ├── libero_env.py      #   LIBERO 封装：reset/step/观测打包/渲染开关
│   ├── action_adapter.py  #   语义动作 → OSC 闭环伺服（本项目的事故易发区）
│   ├── server.py          #   NFS 文件投递服务端
│   └── client.py          #   客户端（agent 侧用）
└── agent/                 # ← qwen35 env 运行（GPU 节点）
    ├── llm.py             #   Qwen3.5 4bit 加载 + 生成（E5 三必修项已内建）
    ├── parser.py          #   XML <tool_call> 解析
    ├── prompts.py         #   system prompt + 8 工具 schema
    └── harness.py         #   主循环：观察→决策→执行→数字回灌→验证
```

## 实测约束（写代码前必读）

| 约束 | 实测值 | 写在哪 |
|---|---|---|
| OSC 开环 delta 只兑现 ~12% | 2.5cm 请求→0.3cm | `action_adapter.py` 闭环伺服 |
| 反馈坐标须 ≥0.1mm 精度 | 1mm 量化→极限环 | `harness.py` 4 位小数 |
| robosuite 须 `ignore_done=True` | horizon 后拒 step | `libero_env.py` |
| 渲染 = 单步 90% | 245ms→24ms renderless | `libero_env.set_render` |
| 4bit 下生成须显式 eos_token_id | 248046≠config 的 248044 | `llm.py` |
| 256px 图像预算：keep=2 | 15 轮 OOM 实测 | `llm.py` / `harness.py` |

完整设计文档（含 Phase 0 六项实验、决策记录、路线图）：[`DESIGN.md`](DESIGN.md)
