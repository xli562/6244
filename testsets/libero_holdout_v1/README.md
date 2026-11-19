# LIBERO 纯测试集 v1

这套题用于在锁定项目的抓取版 harness 后做一次独立测量。选题单位是完整的官方套件。Goal、Object 和 Spatial 各十题，每题有 50 个官方起始摆放，共 1,500 个任务与摆放组合。所有题目都用于测试。调试集和验证集为空。[固定版本的上游项目](https://github.com/Lifelong-Robot-Learning/LIBERO/tree/8f1084e3132a39270c3a13ebe37270a43ece2a01)

- `official_tasks.zip` 包含 30 份 BDDL 任务定义、30 份官方初态文件、任务清单和上游许可证。数据可离线检查。物理资产和模拟器沿用项目原有 LIBERO 安装。
- `manifest.json` 列出上游版本、每个文件的 SHA-256、全部试次编号、历史任务排除清单和出分前确定的指标。三个套件各占总分的三分之一。
- `collect.py` 从固定版本的网址逐份下载定义和初态，对照本地同一 Git 提交验证字节，检查与历史任务及初态的交集。同时把初态转换为同一数值类型后逐个比较。
- `run.py` 将独立的测试清单交给现有 harness 的程序生成和迁移测试入口。它对照上轮的系统指纹，并核验代码、旧程序和测试清单。每个模型对每题分别生成一份顺序计划和一份自由改写。两种方式都可选相同抓法。程序生成一次后复用于全部 50 个摆放。旧抓法对照重放同一份顺序计划。
- `summarize.py` 核验 120 次调用、程序指纹以及每组的 1,500 个任务与摆放组合，输出逐题 CSV 和汇总 JSON。官方成功率是主指标。首次成功后再松手、抬起夹爪并等待，总计 40 个控制步后重新判分，得到稳定成功率。

## 复算

在项目根目录使用已有 LIBERO Python 环境运行。下列最后一条命令只读取已经生成的程序与分数，不消耗模型调用。完整模拟需要项目上轮冻结文件、两家模型的旧程序和对应的 CLI 环境。

```sh
python testsets/libero_holdout_v1/collect.py
python testsets/libero_holdout_v1/run.py prepare
python testsets/libero_holdout_v1/run.py compile --workers 12
python testsets/libero_holdout_v1/run.py test --workers 48
python testsets/libero_holdout_v1/summarize.py
```

项目的测试输出在 `runs/libero_holdout_v1/`。首次得到的分数会保留。已生成的程序可由原 harness 读取复用，测试入口会阻止覆盖已有成绩。公开结果后，如果根据这些题目修改系统，这套题就应改作回归测试使用。
