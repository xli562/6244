# 跨任务交互实验

一份 Python 策略接收当前英文指令，在运行时查找物体、组合动作、检查动作效果。模型开发期间可以收到旧任务的物理记录。测试期间复用同一份冻结程序，每个起始摆放最多 500 个控制步。运行时使用 LIBERO 提供的物体和区域状态。

## 动作和回执

策略仍可调用项目原有的动作接口。新增的 `api.catalog()` 返回实体类别与区域归属，`api.remaining()` 返回剩余步数。`api.attempt(step)` 接受一个动作字典，执行完成后返回可观测的物理变化。例如已经从场景目录选好了物体和目标时，可以写：

```python
receipt = api.attempt({"op": "place", "item": item, "target": target, "grasp": grasp})
lift = receipt["max_lift"]
if lift is not None and lift < 0.025 and api.remaining() >= retry_budget:
    api.attempt({"op": "place", "item": item, "target": target, "grasp": alternate})
```

这段代码只是接口示例。具体的重试条件和策略由 agent 自己写。动作支持放置、打开或关闭抽屉、移动夹爪和开合夹爪。移动或开合时可传入 `watch` 指定观察对象。回执包含动作前后坐标、位移、用掉的步数，以及能测到的提起高度和水平距离。水平距离很小仍可能放得不稳。终止步会被评测器记录，已经终止的程序不会继续重试。技能实现位于 [tools_api.py](tools_api.py)。

## 对照与复核

- 使用一个模型在八道旧任务上写通用程序。每个模型独立生成三份初稿。每份初稿在相同的 16 个练习摆放上运行，随后派生两种修改。两种修改拥有相同的初稿、提示、动作和模型调用预算。
- 自查组依据初稿做第二次代码审查。反馈组在相同提示后额外收到初稿执行时的观测记录。观测包含动作前后变化、异常、步数和抓放记录；开发提示不包含测试指令或官方目标谓词。两组都可以在部署时根据动作回执进行闭环操作。
- 所有初稿和修改稿均冻结并参加测试。主指标取三个独立生成结果的平均稳定成功率。每个共有 500 个相同的公开摆放。旧版参照在同一批摆放上使用既有逐题自由改写入口，每个模型看每道题后另写一份程序。
- [清单](manifest.json) 固定了历史排除范围、状态文件校验值、练习与测试的编号、模型生成重复次数及指标。新题从固定版 LIBERO-90 中按房间和场景抽取。已在上一轮用作纯测试的三个完整套件没有用于练习。[上游数据](https://github.com/Lifelong-Robot-Learning/LIBERO/tree/8f1084e3132a39270c3a13ebe37270a43ece2a01/libero/libero/bddl_files/libero_90)

使用项目 LIBERO 环境，从项目根目录运行：

```sh
python harness/interaction/run.py prepare
python harness/interaction/run.py develop --workers 12
python harness/interaction/run.py prior --workers 8
python harness/interaction/run.py test --workers 48
python harness/interaction/summarize.py
```

程序、提示、每次 CLI 调用、模型收到的练习记录、逐次结果和摘要保存在 `runs/interaction_01/`。已完成的程序按提示复用，正式成绩写入后不会被实验入口覆盖。
