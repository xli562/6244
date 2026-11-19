# 动作前检查

在项目的 LIBERO 状态模拟环境里，程序用 `api.act(step)` 选择技能，并提名要操作的物体和目标。带检查的程序还要写下两个角色各自必须满足的条件。以下是虚构场景中的接口示例。

```python
step = {
    "op": "place", "item": proposed_piece, "target": proposed_slot,
    "grasps": ["thin_edge", "standard"],
    "expect": {
        "item": {"kind": "item", "terms": ["piece"], "near": visible_marker},
        "target": {"kind": "region", "terms": ["slot"], "parent": holder},
    },
}
evidence = api.preview(step)  # 可选，只读，不消耗控制步
receipt = api.act(step)      # 执行前会根据当时的观测重新计算
```

每个角色按类别、描述词及可选的区域归属筛选。需要消歧时，可以选距可见参照点最近的候选，或在指定坐标轴上取最小值、中位数、最大值。相差不超过 2 毫米视作并列。只有一个匹配时执行器才会采用它。没有匹配或有多个匹配时，会把候选和原因返回程序，机械臂还没动。哪些词、哪个参照点及哪个坐标系符合用户指令，仍由模型生成的策略判断。[选择器实现](binding.py)

三组共用同一技能库。放置可指定最多三种抓法，当上一种没有测到足够的提起且仍有步数时，才会尝试下一种。另有打开抽屉、关闭抽屉和沿给定世界坐标方向做短距离推压。动作后的观测沿用前一版。实验对照的控制接口限于这些共享技能，通用程序在部署时无需模型调用。[技能实现](skill_library.py)

在项目根目录使用已配置的 LIBERO Python 环境依次运行：

```sh
python harness/preflight/experiment.py compile --workers 12
python harness/preflight/experiment.py select
python harness/preflight/experiment.py test --workers 48
python harness/preflight/audit.py
```

编译阶段只给模型旧练习题的指令和实体名称，每模型、每种写法独立调用三次，每次生成一份通用程序。编译完成后才按固定规则从官方 LIBERO-90 抽取测试任务。模型写的检查版同时按“启用检查”和“关闭检查”重放。独立生成的仅动作回执组使用相同的感知输入、技能、模型调用和控制步上限。主指标是官方判定成功后松手、抬手和等待共 40 步，结果仍然成立的比例。

输出在 `runs/preflight_01/`。每组的初始提名、执行时的绑定、测量值、阻止或替换记录、动作轨迹与最终得分保存在 `results.json`。未调用执行接口的程序可能没有绑定记录。已完成的成绩禁止覆盖。

首批生成完成时，Claude 六份候选均未能交给模拟器。测试任务仍未抽取。补充批次给两个模型的两组提示都追加同一段兼容性说明，让模型用普通字符串操作并控制程序长度；每组仍独立生成三次。两批分别报告。完整复现时，先运行首批 compile，再运行下列第一条命令，然后才运行 select 与首批 test。最后运行补充批次的 test 与 audit。

```sh
python harness/preflight/second_batch.py compile --workers 12
python harness/preflight/second_batch.py test --workers 48
python harness/preflight/second_batch.py audit
```
