# CS6244 · 让代码智能体在仿真里做家务

给 Codex 或 Claude 一句家务指令，例如“把后方的黑碗放到盘子上”，它会写出一份可重复使用的 Python 策略。我们用 LIBERO 和 MuJoCo 执行这份策略，观察它能否找对物体、做完动作，并在机械臂松手后保持结果。项目目前全部在模拟器中运行。

我们研究的是一个具体问题：**给同一个代码智能体什么样的工具和反馈，它才能在没调过的任务上少犯错？** 本轮比较只看动作回执、动作前核对对象，以及核对后还能查询对象支持的动作并重试。每组沿用一份已有策略，再调用一次模型生成代码。测试时由冻结的程序控制机械臂，重试也无需再调用模型。

## 系统如何工作

![Codex 或 Claude 生成策略，harness 验证动作意图并处理歧义，共享技能在模拟器中执行，独立评测器检查松手后的结果](docs/harness-architecture.svg)

图左是生成策略的阶段，图右是执行一条新指令的过程。[透明背景 SVG 原图](docs/harness-architecture.svg)

1. **写清楚想做什么。** 策略把指令写成结构化动作（IR），声明要用的技能、操作哪个物体、放到哪个目标，以及这些角色应当满足哪些条件。
2. **动手前核对。** Harness 对照模拟器公开的实体名称、类别、区域归属和实时坐标检查提名。遇到并列时，新版回到原句，区分描述物体的词和描述目的地的词，再用标签或到参照点的距离核对。程序还可以查询每个候选能执行哪些技能。例如，固定柜顶和连接滑轨的抽屉都在上方，但只有后者能用抽屉技能拉开。若没有提出动作或提议被拦下，harness 会给出原因，允许程序最多再提两次。
3. **执行和计分。** 共享技能库负责抓取、放置、开关抽屉和短距离推压。策略能看到位移、抓起高度等动作回执。独立评测器读取任务目标，检查官方条件，并在松手、抬手、等待共 40 个控制步后再判一次。本项目报告的主指标是这次复查后的**稳定成功**。

## 最近一次测试

新版比同一份策略在旧 harness 上多完成了 146 个回合（15 个任务编号，每题 50 个摆放，每个模型三份策略）。其中 **47 次**能用同状态重放证实是从错对象改到对对象带来的新增成功；98 次是补上了原本缺失的动作，另外 1 次恢复旧提名也能完成。单看排除了同义改写和熟悉子任务的七题，两种模型的新增成功都是 **0**。完整报告给出了三种方法的分数，以及为什么开抽屉和关炉会出现不同的结果。[动作能力与重试实验](reports/affordance_findings_zh.md)

## 快速复查

从本仓库根目录运行。新机器需要 Git 和 Python 3.11。安装脚本会下载固定版本的 LIBERO 和仿真依赖。已有项目环境时，直接将 `LIBERO_PY` 指向其 Python 即可。

```bash
bash sim/setup_libero.sh
LIBERO_PY="${LIBERO_VENV:-third_party/libero-env}/bin/python"
export MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$LIBERO_PY" harness/reselection/test_recovery.py
```

想看一次真实的仿真执行，可以展开冻结策略，在每道题的一个官方初始摆放上运行评分器。这条命令使用早期 LIBERO-Long 策略，便于先确认物理环境可用。

```bash
DEMO_DIR="$(mktemp -d)"
"$LIBERO_PY" -m zipfile -e reports/simulator_evidence.zip "$DEMO_DIR"
"$LIBERO_PY" sim/libero_long.py score \
  --policy "$DEMO_DIR/runs/long_codex_01/policy_calibrated.py" --split test --limit 1
```

上一轮的动作前重选报告解释了哪些摆放得救、哪些仍被误拦，以及如何复核三组成绩。重新生成策略才需要登录两种 CLI，查看证据和重放已有策略均可离线运行（先备好环境和数据）。[实验报告](reports/reselection_findings_zh.md) · [冻结程序与逐次结果](reports/reselection_evidence.zip)

## 从哪里读起

| 想了解什么 | 入口 |
| --- | --- |
| 课程要求、项目起点，以及纯仿真的实验范围 | [课程要求](docs/project_requirement.md) · [项目背景](docs/project_intro.md) · [仿真实验指南](docs/simulator_experiments.md) |
| 机器人能看到哪些数据、调用哪些技能 | [观测与动作回执](harness/interaction/tools_api.py) · [共享技能库](harness/preflight/skill_library.py) |
| 动作如何表示、歧义怎样重新核对 | [动作 IR 示例](harness/preflight/README.md) · [重选实现](harness/reselection/recovery.py) |
| 如何排除旧题、固定新摆放、公平比较 | [抽题规则](harness/reselection/sample.py) · [配对评测程序](harness/reselection/run.py) |
| 此前在哪些地方涨过分、在哪里失败 | [抓法实验](reports/grasp_findings_zh.md) · [动作回执实验](reports/interaction_findings_zh.md) · [动作前检查实验](reports/preflight_findings_zh.md) |

模拟器向策略公开有名称的场景实体和位置。不同任务可能共享一个房间、同一类碗或同一套抓取技能；新题通过指令和初始状态去重。因此这里的成绩衡量的是有这些观测时的仿真迁移能力。[官方 LIBERO 任务和资产](https://github.com/Lifelong-Robot-Learning/LIBERO/tree/8f1084e3132a39270c3a13ebe37270a43ece2a01)
