# 工具面重设计：把 harness 收缩成「可自动化 + 场景无关」的原语

> 起因（2026-10-11，用户）：模型的能力**不一定在数字理解上**——每轮喂一堆浮点坐标、
> 相对位移、夹爪状态，模型不一定判断得清"现在是什么状态、下一步该做什么"；**画面更重要**。
> 所以 harness 必须负责的是**完全可自动化、且跨场景可泛化**的语义原语（例如"夹住"，
> 并由代码自行判断要不要继续加压），最后**只给模型概念上的反馈**；同时**减少对仿真精确量
> （物体坐标 / 深度 / 高度）的依赖**——现在的 `approach` 一次做了太多事，而且强依赖仿真能给出精确信息。

参考实现：[`FBddcz/embodied-jev`](https://github.com/FBddcz/embodied-jev)（Jev 分层 XYZ / 视觉航点）。

---

## 1. 从 Jev 抄什么

| # | 它的做法（出处） | 对我们的意义 |
|---|---|---|
| 1 | 接触/夹持由**代码判概念**：`finger_contacts`（MuJoCo 接触表，`dist ≤ 0.5mm`）→ `held = 双指都接触 and closed`；`grasp_secured = 接触持续 ≥ 0.16s`；另有 `support_contact` / `forbidden_contact`（`physics.py::contacts`）。原文：*"Closing the gripper does not prove a grasp"*（`incremental.py`） | 与我们 `_grasped` 假阳性的教训**独立收敛到同一结论**：夹持是"接触"这一事实，不是指宽阈值 |
| 2 | `target.kind = pixel \| relative \| world`（`libero_policy.py::resolve_plan`）：模型选**一个可见像素**，代码用 depth + 标定反投影成世界点，再加模型给的 `offset_m` | clearance 由**模型**给，不是 harness 常数；物体坐标不用仿真给 |
| 3 | `annotated_image`：只画 **64px 网格 + 实测 TCP 投影十字准线**，不做任何语义标注 | 让模型能用像素说话，同时不泄露"哪个是目标" |
| 4 | **方向由模型定、幅度由代码定**（`hierarchical.py::assemble_motion`："Scale magnitudes, never replace a model-selected direction"）；动作菜单固定 2/10/40 mm 三档 | 把"算数"从模型身上拿掉，但**不拿走决定权** |
| 5 | **拒绝要带理由，绝不悄悄换动作**：*"程序不会悄悄换一个动作"*；`Candidate.admitted=False` + `rejection=原因` 进下一轮反馈 | 我们现在只有**格式错**的重试通道，物理上不可执行的动作没有通道 |
| 6 | **观测源显式可切换**：`privileged`(真值) / `rgbd`(颜色检测+深度) / `vision`(纯像素)，元数据带 `source`；并有守卫测试 `SensorWorld.__getattr__` → 感知层**读任何仿真真值字段即抛错** | 把"真值泄漏"从暗处搬到明处（我们现在正相反，见 §4） |

**不抄**：它的 `subgoal` 菜单 / `servo_choices` 属于**代码侧阶段选择**，是一条拐杖；我们保留
"任务骨架写在提示词里"（等价、透明、模型可自己改）。

---

## 2. 划线标准

> 代码负责**物理上可自动化 + 跨场景通用**的部分，只回**概念**；
> 模型负责看、选方向、选目标、判断该不该放手。

| 归属 | 内容 | 标准 |
|---|---|---|
| 代码（不可让渡） | 逆运动学 / 关节伺服、夹爪闭合到接触并按需加压、接触判定、动作合法性、工作空间边界、单位换算、标定反投影 | 换物体、换容器、换场景都一样 |
| 模型（必须留） | 抓什么、放哪里、横移方向、抬多高、现在该不该松手、失败后重试还是换策略 | 这就是能力本身 |
| 需要显式选（开关，而不是"顺便替它定了"） | 幅度（代码分档 or 模型给毫米）、clearance（模型给 or 代码常数）、对准（像素闭环 or 世界坐标单次） | 每一条都要能切 |

---

## 3. 目标工具面

| 现在 | 目标 | 代码内部做 | 只回给模型（概念） |
|---|---|---|---|
| `approach(name, height_offset_cm)` | **`goto_pixel(camera, u, v, offset_mm[3])`** | 像素→深度→世界→IK/伺服；越界/深度无效→拒绝并给理由 | `arrived / short / blocked / invalid_depth` |
| `gripper(open\|close)` + 指宽浮点 | **`grasp()` / `release()`** | 张开→闭合到接触→**按需继续加压**→保持窗口→滑脱判定；`release` 内部含落定窗口 | `grasped / empty / slipped`、`released / still_carrying` |
| `move(direction, distance_cm)` | `move(delta_mm)` 或 `move(axis, size∈{small,medium,large})` | 单轴闭环、阻塞检测、边界校验 | `moved_mm`、`blocked` |
| `rotate(axis, angle_deg)` | 同上，幅度分档 | 姿态伺服 | 实测角度变化 |
| `observe`（已注销） | **`look(camera)`** + **`state()`** | 渲染/取图；从接触表算概念 | `holding / contacts / support / forbidden` |
| `verify(claim)` | 不变（环境真值） | 环境 `check_success()` | 通过 / 未通过 |

**整张表没有任何工具需要知道"物体叫什么"** —— 名字只进人类可读日志，不用于取坐标。

**反馈分层**（这条是本次设计的关键取舍）：

* 模型看到的文本 = **概念**（枚举 + 少量定性描述）；
* 转写里的 `structured_result.measurement` = **数字**（指宽 mm、接触侧、步数、时间）。
  ⇒ 模型不为数字分心，而**仪器不丢分辨率**（`scripts/audit_release_poses.py` 那类离线审计依赖它）。

---

## 4. 现状的两个问题（本次要修）

1. **GT 从后门进来**：v2 提示词写着"本配置不提供物体的坐标数字（没有定位工具）"，但
   `server.py:150-157` 的 `approach`/`align` 用 `pack()["objects"]`（robosuite 真值 `<name>_pos`）
   当目标，结果文本里还带坐标。`observe` 被注销了，真值却没走。
2. **每个动作都在向模型抛浮点**：`format_state` 的指宽三分类 + `爪上：<名字>（离爪心 X.Xcm）`、
   `format_outcome` 的 `夹空了 / 指间有物体` 等判断，都是 harness 替模型做的**结论**。

---

## 5. 落地阶梯

| 档 | 改什么 | 为什么在这个位置 | 判定指标（只进日志/转写） |
|---|---|---|---|
| **A**（本次） | `gripper(open\|close)` → `grasp()/release()`；`_grasped` 改由**接触**判定；模型文本只给概念 | 低风险：工具数量不变、观测层不动，**直接验证核心假设"只给概念够不够"** | 抓取成功率、`empty`/`slipped` 计数、re-grasp 次数、turn 数、`release` 后 `verify` 通过率 |
| **B** | `approach(name)` → `goto_pixel` + 三档观测模式(`truth`/`rgbd`/`pixels`) + 守卫测试 | 同时拆掉"真值依赖"和"名字耦合"；需要先做误差探针 | 对准误差（离线用真值算）、到达步数、`invalid_depth` 率、成功率 |
| **C** | 删形容词层（阈值分类、"只走了 X cm"、"已到达/未到达"） | A/B 之后才安全：先有概念替代，再删旧的 | 同上；主要看是否出现"看不懂数字"型停滞 |
| **D** | `move` 幅度分档 + 短动作块（每调用最多 N tick 后强制重观察） | 顺带对齐官方 OSC 接口形状；直接对接 A3 那条"每条命令要 150–200ms 才兑现"的已知问题 | 每步实际/请求位移、到达步数 |

纪律（沿用 `docs/P0_stripdown.md`）：**一次一档、固定 SEED 与 init 集合、n≥16**，
同时报 `env` 与 `declared` 两个成功数（8 回合分辨不了 5% 差异，实测过反转）。

---

## 6. 档 A 的具体设计（本次实现）

### 6.1 概念词表（冻结）

```text
grasp()   → status ∈ { grasped, empty, slipped, rejected }
            grasped  = 两指同时对**同一个**物体有接触，且保持 ≥ HOLD_TICKS
            empty    = 闭合到底（指宽 ≤ EMPTY）且无任何接触
            slipped  = 曾双向接触，保持窗口内接触丢失
release() → status ∈ { released, still_carrying }
            released = 张开到位且落定窗口后**无任何指-物接触**
```

可选的 `measurement`（只进转写）：`contact_sides`、`width_mm`、`hold_ticks`、`settle_ticks`。

### 6.2 接触怎么算（场景无关）

```text
两侧手指 body：robot0_leftfinger / robot0_rightfinger（按名字前缀找，不写死）
物体侧：接触表里与该手指成对的**任意 body**（排除另一侧手指、手臂自身、以及基础/台面 geom）
held = 存在 body b，使 (leftfinger,b) 与 (rightfinger,b) 都在接触表里
```

⇒ 不需要物体名字、不需要物体坐标、不需要指宽阈值，跨场景通用。

### 6.3 内部行为

* `grasp()`：① 先张开到位（消掉"仿真是半闭启动、只捏一下"的假阳性）→ ② 逐步闭合，
  每步查接触，双向接触即停 → ③ **继续命令闭合 RAMP 步**（我们已量到不加压会中途滑脱）
  → ④ 保持窗口（命令 hold，查接触是否持续）→ ⑤ 出概念。
* `release()`：① 张开到位 → ② **落定窗口 N tick**（这是"放手后立刻 `verify` 必然失败"的成因：
  两指还没让开，物体还没落定）→ ③ 查接触 → ④ 出概念。

### 6.4 同时被改掉的东西

* `harness._grasped`：从 `"holding" in note`（字符串）改为 adapter 的 **`held` 概念（接触）**。
* `harness._held_body`：新增，持物身份也来自接触表（`object_body`），`_gate_container` 优先用它；
  旧的"离爪心最近"（真值距离）只作为 legacy 回退。
* `format_state`：改为 `机械爪位置 …；爪上：有物体（两指接触）/ 空`，
  去掉指宽三分类、以及 `爪上：<物体名>（离爪心 X.Xcm）`（名字来自真值坐标，属泄漏）。
* `format_outcome`：`grasp/release` 只回概念句，**不含任何数字**；数字进 `structured_result.measurement`。
* `_structured_outcome`：把概念 `status` 与 `measurement` 并入转写（原先只有 eef/steps，概念会被丢掉）。
* `tool_protocol`：`grasp/release` 登记为 v2 工具；**`gripper` 加入 `UNREGISTERED_TOOLS`**（v2 下模型
  不能回退到旧面）；旧面仍是 v1 的 A/B 路径。
* v2 提示词 + `skills/scene_tools.md`：删掉指宽配方与"先 open"的配方，改为 grasp/release 语义。
* 无 GPU 回归：`scripts/probe_harness_dryrun.py` 增加"概念腿"（脚本化 v2 序列，5 项断言）。

### 6.5 实测（2026-10-11，CPU 仿真，无需 GPU）

`bash scripts/simenv.sh scripts/probe_grasp_concept.py`：

| 场景 | 概念输出 | 说明 |
|---|---|---|
| reset（指宽 20.67mm，接触 0） | 无 | **旧指宽规则会报"半闭（指间有物体）"= 假阳性**；接触判定正确 |
| 空中 `grasp()` | `empty` | |
| 对准后 `grasp()` | `grasped`，双向接触，`object_body=alphabet_soup_1_main` | squeeze 10 tick + hold 10 tick |
| 高空 `release()` | `released`，`cleared_after_ticks=1` | |
| 偏轴 / 闭合指下降后再 grasp | `slipped`（单侧接触）/ `empty` | 旧规则在这两处分别报"闭合（空）"与"半闭" |

**关键一条**：完整流程（release 张开 → approach 对准 → grasp → move +z ×2 → approach(篮,25) → release）
之后，`check_success()` **在 release() 返回后立刻就是 True**（此前 8/8 回合"放手后立刻 verify 必失败"）。
注意它的机制：`contain_region` 的 z 上限是 0.137，所以罐子**还在下落**时环境就可能判成功——
用的是 LIBERO 自己的判据，仪器侧仍应记录"落定后"的状态（探针里记录 0.0554）。

**同时暴露的设计缺口（已解决）**：`grasp()` 内部虽然会先张开，但**下降发生在它之前**，
所以"空手张开"必须有入口。决定：`release()` 兼作"张开/复位"，在工具描述与提示词里写明。
若模型出现用错的情况，下一轮可改名为 `open_pads()`。

**回归**：`python scripts/probe_tool_wiring.py` 全 PASS（8 个 v2 工具都有 dispatch 分支）；
`bash scripts/simenv.sh scripts/probe_harness_dryrun.py` → `PLUMBING OK`（含概念腿 5/5）。

---

## 7. 待办 / 风险

* **档 A 之外**：`approach` 的替代（档 B）需要先做"像素→世界"误差探针（agentview 在物体处约
  2.70 mm/px，而抓取需要 ~2 mm 对齐 ⇒ 必须"先对准再下降"两段式，或做图像空间伺服）。
* **`grasp()` 的"要不要加紧"**：加压后仍需**滑脱检测**（接触集变化 = 滑脱），这条跨场景通用。
* **概念反馈会掩盖细节**（滑到指间外侧、被口沿卡住）：用 `slip` 事件 + 腕部图像补，**不再加浮点**。
* **拒绝通道**：动作不可执行时要回"拒绝 + 理由 + 你现在能做什么"，不退化成静默重试（§1 第 5 条）。

## 7. 档 B 已完成（2026-10-12）

### 7.1 先测误差预算（`scripts/probe_pixel_grounding.py`，CPU，纯几何）

判据用**256 个网格点打平面**（平面必须回来还是平面），而不是拿少量物体比对——
第一版就是后者，结果手腕相机只看到自己的爪子（单样本），把行序判反了两次。

| 判据 | 结果 |
|---|---|
| 深度缓冲行序 | **row0=bottom（OpenGL）**：桌面平面 MAD **4.7mm（第三人称）/ 0.1mm（腕部）**；反向 526mm / 33.7mm |
| 数学正确性 | 往返 u/v 误差 **0.000px**；平面散布 0.1–4.7mm ⇒ 反投影与标定自洽 |
| 第三人称精度 | **2.96–4.04 mm/px**；±1px=3.6mm、±2px=7.4mm；网格 4px=2.8mm、8px=9.5mm、16px=32mm |
| 腕部（抓取高度，罐子在正下方） | **1.10 mm/px**；±1px=1.1mm；网格 4px=2.0mm、8px=5.0mm |
| 表面 vs 中心 | 可见表面比物体中心近 **+20…+90mm**（篮子 +90mm）⇒ **clearance 必须由模型给** |

⇒ 结论：第三人称只能做粗对准（~1cm），**抓取所需的 ~2mm 必须靠腕部近距离**（或图像闭环）。
所以 `goto_pixel` 做成"像素 + 模型给的 offset"的一次到位原语，**由模型分两段用**。

### 7.2 实现了什么

| 层 | 内容 |
|---|---|
| 新模块 `sim/pixel_grounding.py` | `Camera`（**世界系** `data.cam_xpos/cam_xmat`；`model.cam_pos` 对挂在手部的相机是局部坐标）、`depth_metres`、`sample_depth`（3×3 中值 + 量程校验）、`backproject`、`ground_pixel`（**不可用就带理由拒绝**）、`annotate_frame`（网格 + 夹爪准线，无任何语义标注） |
| `sim/server.py` | 新动作 `goto_pixel`；`observation_mode`（`truth`/`pixels`）**在服务端强制**：pixels 下按名字的 `descend`/`align` 直接抛错并给替代；两种传输的图像路径都加标注 |
| `sim/action_adapter.py` | `classify_descent`（arrived/short/blocked 的**唯一**判词来源，避免两处各自解释 note） |
| 工具面 | `goto_pixel` 登记；`MODE_ONLY_TOOLS` 让 pixels 模式**不广告也不接受** `approach`，拒绝时提示改用 `goto_pixel` |
| `harness.py` | `goto_pixel` 的 dispatch；`parse_vec3`（模型给的向量常是字符串 `"[0,0,120]"`，逐字符迭代会发出 14 个数被拒）；`format_outcome` 只回概念 + mm/px；**`model_view`**：pixels 模式下把 `measurement.object_body` 从**给模型的那份**里剥掉（转写保留） |
| CLI/脚本 | `--observation-mode`、`--pixel-grid`（服务端与 run 两侧都必须一致）；`run_single_node.sh` 支持 `OBS_MODE`/`PIXEL_GRID` |

### 7.3 守卫与实测（`probe_harness_dryrun.py` 的"像素腿"，无 LLM，~40s）

7 项断言全 PASS，且 `success=True source=env turns=9`：
工具面里没有 `approach` / `goto_pixel` 多次调用且给概念 / **工具结果与状态行里没有物体名** /
**没有物体坐标** / `approach` 被拒且提示替代 / 任务由像素完成 / **画面确实带网格标注**
（按收到的像素里网格颜色计数验证，不是"代码里画了"）。

这套断言就是**行为版的 sensor-rights 守卫**（Jev 的 `SensorWorld` 是静态版：读禁字段即抛错）。
它当场抓到一个真泄漏：档 A 的接触判定把 `object_body="alphabet_soup_1_main"` 写进了每个工具结果的结构化
JSON 里，而 pixels 模式的全部意义就是"没有物体身份"。

### 7.4 版图的粗糙处（已知）

* pixels 模式下 skill 文件仍以 `approach(<名字>)` 叙述，靠提示词附录里一句"一律读作
  `goto_pixel`"过渡。**下一步**：给 pixels 模式写一份独立 skill（把 `approach` 的段落换成像素流程）。
* `rgbd` 档（颜色/深度检测出物体坐标）**没有做**：它需要一个知道物体颜色/大小的**场景专用检测器**
  ——Jev 自己也写着"换了颜色或形状，需要调整检测器"。现在只做 `truth`/`pixels` 两档。
* 第三人称的 3.7mm/px 决定了"一次到位"不可行；`goto_pixel` 不做自动再观察，由模型自己分两段。
* 尚未用真模型跑过（需要 GPU）。预期的新失败形态：模型读网格读不准、或忘了给 clearance。

## 8. 档 B v2：`goto_pixel` → `aim` + `target`（2026-10-12）

### 8.1 为什么改（实测，不是推测）

| 分辨率 | 模型瞄的像素 vs 罐子真实像素 | 离罐子 | 行为 |
|---|---|---|---|
| 256 | asked (96,128)/(112,160) vs can (65.5,123.3) | **>121mm** | 37 轮里只调 1 次像素，其余 36 轮是盲 `move` |
| 512 | asked (96,288) ×4（四次几乎逐像素相同）vs can (131,247) | **101mm** | 5 次像素 + 19 次 `move`，同样转为盲走 |

⇒ 提高分辨率把"瞄错物体"变成"瞄对物体但偏 10cm"，但**偏置是系统性的**，所以"多试几次会收敛"
不成立；而抓取容差只有 ±2.2cm（实测 2.2cm 仍能夹住、4.6cm 夹空）。差 5 倍。

⇒ 结论：这个模型做**绝对像素命名**就是 ~50px/10cm 级精度。补分辨率补不上。要补的是
**把绝对估计换成相对判断**——这正是 Jev 分层原则里的"方向由模型定、幅度由代码定"。

### 8.2 设计（两个工具，一静一动）

| 工具 | 动作用 | 相机 | 返回 |
|---|---|---|---|
| `aim(camera, u, v)` | **干跑**：把红圈画在指定像素上，不动手、不走物理 tick | agent **或** wrist | 同一张带网格的图 + `marked`/`rejected` + mm/px |
| `target(camera, u, v, offset_mm, max_steps)` | 移到该像素（反投影 + 模型给的 clearance） | 只能 agent | `arrived`/`short`/`blocked`/`rejected` |

* 红圈画在**模型给的那个像素**上，不画在反投影再投影回来的位置（往返实测 0.000px，
  画回去只会掩盖接地误差）。同时图上有**青色十字=夹爪真实投影**，所以"我想去哪"和"我在哪"
  可以在同一张图里比。
* 标记几何：**6px 空心圆环 + 1px 黑描边 + 中心点**（`AIM_MARK_RADIUS_PX`）。第一版提案的
  3×3 实心点在 256px 下只有罐子（~16px 宽）的 1/5，看不出差距；而**要看的不是标记本身，
  是标记与目标的差距**。半径是 CLI/环境变量（`AIM_MARK_PX`），因为它是个实验变量。
* **短动作块**：`max_steps`（1~8，默认 8）限制一次调用最多几个 xy/z 修正步，对应 Jev 的
  `max_motor_steps`。旧实现一次调用最多 80 步（~40cm 行程）——两次真实跑各有 36/19 轮盲走。
  块用尽会回 `truncated`，措辞与 `blocked` 分开（一个是"停下来看看"，一个是"过不去"）。
* **拒绝分成两类**，一刀切开：**物理上不可用**（越界/深度无效）→ 软拒绝，带理由 + 把画面一起
  退回（同一轮就能换点）；**工具用错**（`target` 给 wrist）→ 抛错并给替代写法（腕部只能 `aim`）。
* pixels 模式下**从给模型的那份结果里剥掉世界坐标**（`surface_xyz`/`target_xyz`/`offset_m`/
  `depth_m`/`mark`），转写里全留。`mm_per_px` 故意留给模型：它需要一把尺子把格子换成厘米。

### 8.3 守卫与实测

`probe_harness_dryrun.py` 像素腿（无 LLM）：**13 项全 PASS**，`success=True source=env turns=17`，
其中新增的：`aim` 只标注不动手（用 `eef_start==eef_end` 判）、`aim` 可用腕部而 `target` 不行、
越界像素被拒且有理由、**短动作块有上限（每次 ≤ 8 个修正步）**、不给模型世界坐标键、
**红圈真的出现在返回的画面上**（按 `AIM_MARK_COLOUR` 像素计数，不是"代码里画了"）。

肉眼复核（`scripts/probe_aim_marker.py`，故意瞄偏 30px）：红圈画在 requested 像素上，
落点是旁边的瓶子（仪器独立报告离 `alphabet_soup_1` 16mm、离罐子像素 30px），两张图都
能看出差距——即"仪器计数"与"像素"两处独立证实。

### 8.4 已知粗糙处

* `aim` 的代价是**一个模型回合**（每回合 ~2–6s），所以"调到满意再动手"确实更贵；
  这是拿回合换精度的交易，而旧路子（盲目移动）已经证明换不到。
* 尚未测过"模型能不能看出红圈与目标的差距"——这正是下一个岔路口：若 Δ 不收敛，就不该再
  加像素粒度，而应换成**是非题/候选菜单**（Jev 每轴 `negative|hold|positive`）。
* `zoom=True`（以该像素为中心放大 4× 的裁剪）没有做，留作下一个实验变量。
* skill 文件（`skills/scene_tools.md`）已同步改名，但它是**共享**的，truth 模式也会读到
  `aim`/`target` 的段落（那里这两个工具不存在，靠 tools 列表不广告来兑）。

## 9. 参考

* 计划与教训：[`docs/P0_stripdown.md`](P0_stripdown.md)（该剥/不该剥的边界）
* 场景与几何实测：[`skills/scene_tools.md`](../skills/scene_tools.md)
* 放手位姿审计：[`scripts/audit_release_poses.py`](../scripts/audit_release_poses.py)
* aim 标记/不动手探针：[`scripts/probe_aim_marker.py`](../scripts/probe_aim_marker.py)
* Jev 出处：`src/embodied_jev/physics.py::contacts`、`libero_policy.py::resolve_plan`/`annotated_image`、
  `hierarchical.py::assemble_motion`、`perception.py::VisualObserver`/`ImageObserver`
