# 纯仿真实验的复现入口

这台机器已经装好环境和数据。在项目根目录可以直接复查冻结程序。在新检出目录复现本轮实验时，先在项目根目录用 `unzip -n reports/simulator_evidence.zip` 展开冻结程序和逐次成绩，再按下文安装 LIBERO。实验结果见[中文报告](../reports/simulator_findings_zh.md)。

```bash
LIBERO_PY=/data/yuchaoq/cs6244/libero-env/bin/python
"$LIBERO_PY" sim/libero_long.py score --policy runs/long_codex_01/policy_calibrated.py --split test --limit 1
```

## 多步操作任务

任务、模型资产、BDDL 目标和每项任务的 50 个初始状态来自 [LIBERO 官方仓库](https://github.com/Lifelong-Robot-Learning/LIBERO/tree/8f1084e3132a39270c3a13ebe37270a43ece2a01)。使用 LIBERO-Long 的全部 10 项任务。[项目划分](../sim/data/libero_long.json)记录每个源文件的 SHA-256。划分属于本项目，成绩适用于给策略提供物体名称和位置的仿真设定。

| 分组 | 任务与初始状态 | 用途 |
| --- | --- | --- |
| 调试 | 6 项任务，每项索引 0–19 | 代理看结果、修改程序 |
| 验证 | 同 6 项，每项索引 20–29 | 控制器选版本，代理看不到这批试跑轨迹 |
| 熟悉任务的新摆放 | 同 6 项，每项索引 30–49 | 程序冻结后计分 |
| 新任务 | 另 4 项，每项全部 50 个初始状态 | 测同一份程序执行新指令的能力 |

调试任务包括开炉放壶、碗入柜并关上、两种不同的双食品装篮、双色杯分别放盘，以及把书收入收纳盒的后格。新任务包括杯子放进微波炉再关门、两把摩卡壶放炉上、奶酪和黄油一起装篮，以及杯子上盘同时布丁放在盘子右边。最后一组既包含新物品组合，也包含开发阶段没用过的电器。任务的实际成功条件由上游判定器读取，有些指令隐含多个条件，例如两把壶上炉还要求炉子开启。

在项目根目录执行以下命令。需要 Python 3.11、Git，以及已登录的 Codex CLI 或 Claude Code CLI。安装脚本给 LIBERO 建一个独立 Python 环境，以便原抽屉项目继续使用其自身版本的 MuJoCo。可用环境变量 `LIBERO_VENV` 指定磁盘空间充足的位置，`PYTHON` 指定 Python 3.11 可执行文件。

```bash
bash sim/setup_libero.sh
LIBERO_PY="${LIBERO_VENV:-third_party/libero-env}/bin/python"
OPENBLAS_NUM_THREADS=1 "$LIBERO_PY" sim/multitask.py develop --agent codex --run runs/long_codex_new
OPENBLAS_NUM_THREADS=1 "$LIBERO_PY" sim/multitask.py test --run runs/long_codex_new
OPENBLAS_NUM_THREADS=1 "$LIBERO_PY" sim/multitask.py develop --agent claude --run runs/long_claude_new
OPENBLAS_NUM_THREADS=1 "$LIBERO_PY" sim/multitask.py test --run runs/long_claude_new
```

每次开发默认生成 3 个候选。每项任务每个候选默认使用 2 个调试状态和 2 个验证状态，以控制计算量。可显式调高 `--dev` 和 `--val`。第一版程序拿到任务指令、可用接口和开发任务的物体目录。后续版本还会看到各开发任务的成功数、某次失败的子目标、物体提升高度或程序异常。下一轮从开发集上最好的程序开始改。最终版本由另一批验证状态决定，先比完整任务成功次数，打平再比各任务达成的子目标比例，仍相同则保留较早版本。每次开发都保存提议代码、调试结果、验证结果和冻结程序摘要。

测试命令在同一批初始状态上评估第一版、最后一版及被选中的版本，默认使用 320 个测试状态。程序在运行时只收到自然语言指令和模拟感知接口，不能调用裁判接口。测试使用上游任务的成功条件，所有移动、抓放和按压均通过物理控制器执行。这里的 500 步指机器人控制步数。开发程序由本机运行，适用于受信任的实验输入。

如需单独复查某份冻结策略，可以跳过 CLI，直接运行下面的命令。

```bash
"$LIBERO_PY" sim/libero_long.py score --policy runs/long_codex_new/policy.py --split unseen --limit 2 --out /tmp/libero_unseen.json
```

## 用抽屉任务拆解涨分原因

在原来的 `robot` 环境执行。对每个代理取历史多布局条件的前三次开发，每次取第一次试跑时的程序。每份相同的起始代码复制到四组，组合“固定或变化的开发噪声”和“显示或隐藏噪声标签”。每组使用相同布局、相同噪声种子、一次反馈、一次完整程序修改。固定噪声为 5 mm，变化噪声每 8 个开发场景涵盖 2.5–10 mm。24 份修订程序全部生成后，再在同一批 64 个未见仿真场景（种子 605）上对照 6 份原始程序。历史运行文件和输入场景位于当前项目的本地实验目录。

```bash
OPENBLAS_NUM_THREADS=1 python sim/noise_ablation.py develop --workers 4
OPENBLAS_NUM_THREADS=1 python sim/noise_ablation.py score --workers 24
```

这个实验回答“一次修复时，哪种信息有帮助”。历史五轮开发还包含多轮选择与错误累积，不能直接用一次修复给整个历史涨幅分摊功劳。

## 原抽屉程序的独立验证选版

给现有 `sim/harness.py` 的开发命令增加 `--select-checkpoint`，会额外输出 `selected_policy.py` 和 `selection.json`。选择器在 16 个新的多布局开发验证场景上试跑所有不同候选，噪声涵盖 2.5、5、7.5 和 10 mm。得分相同时保留先前版本。历史对照可用下面的命令重放。它固定两家代理各 8 次历史运行的代码提议，比较原交付版本和验证选版，最终在旧实验的第二批 100 个测试场景上计分。

```bash
OPENBLAS_NUM_THREADS=1 python sim/replay_selection.py --workers 24
```

多任务小样本验证有时分不出好坏。可用下面的命令在全部 60 个验证状态上给同一批已经生成的候选重新排名，然后冻结并独立测试。这个过程不会再向代理提问，也不会把测试反馈送回开发过程。

```bash
"$LIBERO_PY" sim/multitask.py calibrate --run runs/long_codex_new
"$LIBERO_PY" sim/multitask.py test --calibrated --run runs/long_codex_new
```

为定位抽屉策略究竟受哪项感知影响，还可以对历史最终程序做干预。下面的命令在同一批 32 个新仿真配置上分别保留原噪声、把抽屉开度/位置和把手位置读准、或把杯子位置和角度读准。其余感知和物理参数沿用同一集配置，固定策略过程中不再调用模型。

```bash
OPENBLAS_NUM_THREADS=1 python sim/sensor_ablation.py --workers 24
```

## 新指令到来时，只给代理一次改写机会

另外留出 LIBERO-90 的三项任务，分别为开炉放煎锅、叠两只碗再放入托盘、把书放进收纳盒前格。各有 50 个官方初始状态，文件和校验值见[扩展清单](../sim/data/libero_transfer.json)。两种运行方式都允许程序在执行时读取指令和物体位置。一种直接复用多任务程序；另一种在收到指令时，把原程序、英文指令和物体/区域名称给同一个 CLI，允许它为该任务重新编写一次程序。没有示范、初始坐标、裁判、测试试跑和测试反馈。三份改写都冻结之后，才在相同的 150 个初始状态上成对打分。

```bash
"$LIBERO_PY" sim/task_adapter.py compile --agent codex --base runs/long_codex_new/policy_calibrated.py --out runs/transfer_codex_new
OPENBLAS_NUM_THREADS=1 "$LIBERO_PY" sim/task_adapter.py test --out runs/transfer_codex_new
```

Claude 使用同样的命令，把代理名、输入程序和输出目录改成对应的 Claude 运行。大批量 LIBERO 测试可以设置 `LIBERO_WORKERS`（例如 8）来调节多进程并行数；单独的新指令测试使用 `--workers`。

为比较看过物理反馈后再改一次是否有用，`refine` 使用每个扩展任务的初态 0–2 调试，初态 5–7 在原程序、按指令改写和一次针对性修复这三个版本中选一个。调试反馈含失败的子目标、物体与目标的最终位置和夹爪尝试。调用 `test-repair` 只比较各任务的初态 10–49，三个版本各有相同的 120 次测试，冻结后不再修改。这个补充实验是在看过前面整批汇总后设计的，属于回顾性比较。模型本身没有看到测试反馈。

```bash
"$LIBERO_PY" sim/task_adapter.py refine --out runs/transfer_codex_new
"$LIBERO_PY" sim/task_adapter.py test-repair --out runs/transfer_codex_new
```

报告中两份有代表性的关抽屉程序，也可以用 `python sim/sensor_ablation.py --closing-examples` 单独复查。它只改指定传感器或指定高度常量，结果保存在诊断目录，不会覆盖原程序。

原抽屉批处理也支持 `batch.py run --select-checkpoint` 和 `batch.py score --selected`。后者校验冻结文件的摘要，把成绩写入单独的 `<tag>_selected` 目录。同一批程序可保留原交付版和验证选版两份成绩。

## 复现两把壶、布丁和五项新任务

本轮给每条新指令一次 CLI 调用。模型只收到英文指令和可感知的物体、区域名称，输出有顺序的“搬哪个物体、放到哪里”。同一份计划在该任务的全部初态复用。放置执行器会给共享目的地分配落点，确认拿起来后再搬运，根据物体与夹爪的实际偏移决定在哪里松手。包含开关门等其他操作的指令可以调用该模型上一轮冻结的整段程序。

[实验清单](../sim/data/libero_composition.json)固定了任务与上游文件校验值。两项开发任务各用初态 0–5 调试、6–9 验证，10–49 评估修复。旧题里另外两项各用全部 50 个初态复查。五项新题每项 50 个初态，在执行器和提示词冻结后才为各题生成计划，计划也在计分前冻结。

在项目根目录先解压上轮的 `reports/simulator_evidence.zip`，并运行本页的 LIBERO 安装步骤。只想重放本轮成绩时，解压本轮的 `reports/composition_evidence.zip`，直接运行下列 `test` 和 `freeform-test` 命令即可。要重新调用两个模型，从头做实验时使用新目录，如下所示。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
p="${LIBERO_VENV:-third_party/libero-env}/bin/python"
r=runs/composition_new
"$p" sim/libero_long.py prepare
"$p" sim/composition.py plan-debug --out "$r"
"$p" sim/composition.py debug --out "$r"
"$p" sim/composition.py validation --out "$r"
"$p" sim/composition.py freeze --out "$r"
"$p" sim/composition.py plan-test --out "$r"
"$p" sim/composition.py test --out "$r" --workers 8
for agent in codex claude; do
    "$p" sim/composition.py freeform-plan --out "$r" --agent "$agent"
    "$p" sim/composition.py freeform-test --out "$r" --agent "$agent" --workers 8
done
"$p" sim/composition_release_probe.py --out "$r"
```

重放本轮原始计划时将 `r` 设为 `runs/composition_01`。测试文件里的 `success` 是 LIBERO 官方条件首次满足时的结果。另一个字段 `holds_after_release` 表示当时已成功，而且统一松爪、抬手、等待共 40 个控制步后，全部目标仍同时成立。额外 40 步只用于核验，策略的预算仍是 500 步。

叠碗任务存在上游文字与判定条件冲突。完成两项 `freeform-test` 后，可运行 `"$p" sim/composition_language_audit.py --out "$r"` 核对同一程序结束时是否满足文字指令。完整结果见[本轮中文报告](../reports/composition_findings_zh.md)。

## 红杯落点与双抽屉

本轮入口是 `sim/drawer_harness.py`。它沿用上一轮的放置动作，增加了区分“物体旁边”和“容器内部”的目标选择、按柜子朝向操作抽屉、以及检查两步动作先后顺序的记录。红杯修复题采用用户实际给出的指令，原始模拟器文字也保存在输入文件中。迁移测试始终使用上游自带的英文。

[任务清单](../sim/data/libero_drawers.json)包含两项修复和六项新题。使用本页的 LIBERO 环境；若在干净检出目录运行，先解压 `reports/simulator_evidence.zip` 取回两个模型的旧程序。要重放旧计划对照，再解压 `reports/composition_evidence.zip`。下列命令从头生成一轮新计划，调用的模型、可感知名称、提示词和每次模拟的目标状态都会保存。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
p="${LIBERO_VENV:-third_party/libero-env}/bin/python"
r=runs/drawers_new
"$p" sim/drawer_harness.py prepare --out "$r"
for m in plan old_plan free free_skills; do
    "$p" sim/drawer_harness.py compile --group repair --method "$m" --out "$r"
done
"$p" sim/drawer_harness.py debug --out "$r"
for m in plan old_plan rim_off tilt_off; do
    "$p" sim/drawer_harness.py validation --method "$m" --out "$r"
done
"$p" sim/drawer_harness.py freeze --out "$r"
for m in plan old_plan previous free free_skills; do
    "$p" sim/drawer_harness.py repair-test --method "$m" --out "$r"
done
for m in plan free free_skills; do
    "$p" sim/drawer_harness.py compile --group fresh --method "$m" --out "$r"
done
for m in plan free free_skills; do
    "$p" sim/drawer_harness.py transfer --method "$m" --out "$r"
done
```

`plan` 要求模型给出有顺序的动作。两种自由改写都以该模型原有程序为起点：`free` 提供原来的底层接口，`free_skills` 另提供与新版计划相同的抓放、相对位置和抽屉接口。各组各模型对每项任务调用 CLI 一次，同一份程序复用全部初态。`previous` 重放上一轮保存的两份零分计划，`old_plan` 是让上一轮提示词重新读取本次给定的指令。`rim_off` 和 `tilt_off` 只用于比较关闭新抓点和新腕姿后的变化。

整条指令完全达成才记成功。记录里的 `first` 是各子目标首次满足的控制步；`settled` 要求成功后统一松爪、抬手、等待 40 步，结果仍成立。要复查本轮的冻结程序，解压 `reports/drawer_evidence.zip`，把 `r` 改为 `runs/grounded_drawers_01`，运行 `prepare`、`repair-test` 和 `transfer` 即可，无需再请求模型。

若 CLI 已返回完整代码，但 `episode.check_policy_source` 拒绝该程序，为了保留每题一次调用，可以执行 `"$p" sim/record_rejected_proposal.py 运行目录/fresh/方法/模型/任务`，再恢复该组 `compile`。此工具重新核验原回复，保存原程序和拒绝原因，并生成零动作的报错程序用于计分。本轮 Claude 在底层接口自由改写组有一条这样的回复；详见[本轮报告](../reports/drawer_findings_zh.md)。

## 按物体选择抓法

`sim/grasp_harness.py` 同时运行两种开发方式。两家 CLI 各收到同一接口、同一模型自己的起始程序、同一句任务英文和物体目录。一个提示要求编写顺序计划并为每件物体选抓法，另一个允许自由改写。两组都能直接选择 `rim`（夹碗沿）、`thin_edge`（沿薄物体的厚度夹住）或 `standard`（原有的普通抓放），也都能调用原来的动作。薄物体抓法使用模拟器可见的物体碰撞盒姿态和目标区域尺寸；两组共享实现。

[固定的任务清单](../sim/data/libero_grasps.json)记录了任务定义和起始状态的校验值。先按本页的安装说明准备 LIBERO 和旧程序。每题各模型各组只调用一次 CLI。程序生成失败也消耗这次机会。下面的 `r` 取新目录名时会创建新实验。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
p="${LIBERO_VENV:-third_party/libero-env}/bin/python"
r=runs/object_grasps_new
"$p" sim/grasp_harness.py prepare --out "$r"
"$p" sim/grasp_harness.py compile --group repair --out "$r"
"$p" sim/grasp_harness.py debug --out "$r"
"$p" sim/grasp_harness.py validation --out "$r"
"$p" sim/grasp_harness.py freeze --out "$r"
"$p" sim/grasp_harness.py repair-test --out "$r"
"$p" sim/grasp_harness.py compile --group fresh --out "$r"
"$p" sim/grasp_harness.py transfer --out "$r"
```

输出 JSON 按“方法/模型”分组。`fixed` 用同一份顺序计划，把显式选抓法的调用交给原来的抓放动作。`old_rim_offset`、`no_pick_yaw`、`no_slot_yaw` 分别只改变碗沿抓点、拿书时的腕部对准、书进格前的旋转。另用 `legacy_pick_motion` 替换提起前后的动作顺序和时长，用 `low_edge` 单独降低薄物体抓点。这些对照限于调试和验证集。各次运行记录官方最终目标、首次达成步骤、抓法和提起高度，以及统一松爪抬手等待 40 步后的结果。要复查本轮而不用再调用模型，在项目根目录解压 `reports/grasp_evidence.zip`，设 `r=runs/object_grasps_01`，只运行 `prepare` 和相应计分阶段。

本轮成绩和原因核验见[按物体选抓法的实测报告](../reports/grasp_findings_zh.md)。
