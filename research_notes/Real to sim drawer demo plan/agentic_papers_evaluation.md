# Agentic-Robotics Reading List and Evaluation Protocol for Real-to-Sim + Scene Variation (cup-in-drawer)

Researched 2026-10-16. All numbers below come from the primary paper or project page cited next to them. Where a fetch tool summarized a page and a figure was only shown in a chart, these notes say so rather than guessing.

## Q1. What does each paper on the reading list do, and how does it relate to the project?

### Takeaway
The list follows a clear line. ProgPrompt and Code as Policies (2022) showed that an LLM can write a program or plan in one shot. ApBot and SetItUp (Hsu group) use structured intermediate models such as a state machine or a grounding graph, plus a verification step. The 2026 NVIDIA/Berkeley "harness" papers close the loop. ASPIRE repairs programs and builds a skill library from execution traces. GaP rehearses graph policies in Isaac Sim over sampled task variations. ENPIRE runs the loop on real robots. VoLo orchestrates tools while the robot is running. GaP and ASPIRE are the closest templates for this project: both iterate on a program in simulation over sampled variations and then test on held-out instances.

### Cited Findings

**ProgPrompt (Singh et al., arXiv 2209.11302; ICRA 2023)**
- **Authors and venue:** Ishika Singh, Valts Blukis, Arsalan Mousavian, Ankit Goyal, Danfei Xu, Jonathan Tremblay, Dieter Fox, Jesse Thomason, Animesh Garg — [arXiv](https://arxiv.org/abs/2209.11302). The project page lists it as ICRA 2023, with an extended version in *Autonomous Robots* 2023 — [progprompt.github.io](https://progprompt.github.io).
- **Problem:** Generate situated robot task plans with LLMs. The prompt is written as a Python-like program: available actions are imported functions, scene objects are a list, and a few example programs are included. This lets the LLM produce plans that match the environment and the robot's capabilities — [arXiv](https://arxiv.org/abs/2209.11302).
- **Feedback:** Generated plans contain assertions that "check for action pre-conditions, and trigger recovery actions" — [progprompt.github.io](https://progprompt.github.io).
- **Metrics:**
  - SR: the fraction of executions that achieve all goal conditions.
  - Exec (executability): the fraction of actions that can be executed.
  - GCR (goal-condition recall): the share of goal conditions achieved.
  - Source: [ar5iv](https://ar5iv.labs.arxiv.org/html/2209.11302).
- **Results on VirtualHome (70 household tasks):**
  - ProgPrompt with Codex: SR 0.40, Exec 0.90, GCR 0.72.
  - ProgPrompt with GPT-3: SR 0.34, Exec 0.84, GCR 0.65.
  - LangPrompt: SR 0.00, Exec 0.36, GCR 0.42.
  - Huang et al. baseline: SR 0.00, Exec 0.45, GCR 0.21.
  - Source: [ar5iv](https://ar5iv.labs.arxiv.org/html/2209.11302).
- **Ablations:**
  - Removing assertion feedback lowered SR from 0.34 to 0.28 and GCR from 0.65 to 0.56.
  - Removing comments also hurt performance.
  - Source: [ar5iv](https://ar5iv.labs.arxiv.org/html/2209.11302).
- **Real robot:** A Franka Panda ran 4 tabletop tasks, each with and without distractors. It almost always succeeded, failing only on the sort task. Assertion feedback was not used on the robot — [ar5iv](https://ar5iv.labs.arxiv.org/html/2209.11302).
- **Code:** https://github.com/NVlabs/progprompt-vh — [progprompt.github.io](https://progprompt.github.io).

**Code as Policies (Liang et al., arXiv 2209.07753)**
- **Authors:** Jacky Liang, Wenlong Huang, Fei Xia, Peng Xu, Karol Hausman, Brian Ichter, Pete Florence, Andy Zeng — [arXiv](https://arxiv.org/abs/2209.07753).
- **Method:** Code-writing LLMs generate Python policy code from language commands. The code can use loops and conditionals, call perception APIs and control primitives, and do spatial-geometric reasoning. Hierarchical code generation recursively defines functions that are still undefined — [arXiv](https://arxiv.org/abs/2209.07753).
- **Reported number:** Hierarchical code generation solves 39.8% of HumanEval — [arXiv](https://arxiv.org/abs/2209.07753).
- **Domains:** Tabletop manipulation (blocks/bowls; fruits, bottles, and plates), whiteboard drawing, and mobile navigation/manipulation. The generated code can include perception "feedback loops," such as "wait until you see…" — [code-as-policies.github.io](https://code-as-policies.github.io).
- **Code:** google-research/code_as_policies, plus a Colab notebook — [code-as-policies.github.io](https://code-as-policies.github.io).
- **Venue:** The project page shows only the arXiv citation. ICRA 2023 is the commonly cited venue but was not confirmed here.

**ApBot — "Robot Operation of Home Appliances by Reading User Manuals" (arXiv 2505.20424)**
- **Authors and dates:** Jian Zhang, Hanbo Zhang, Anxing Xiao, David Hsu. v1 was posted 26 May 2025 and v2 on 23 Jul 2025 — [arXiv](https://arxiv.org/abs/2505.20424).
- **Building the appliance model:** A VLM builds a symbolic state-machine appliance model from the manual, with states, actions, transitions, and goal states. Syntax checkers validate it, with up to 3 regeneration attempts — [arXiv HTML](https://arxiv.org/html/2505.20424v2).
- **Macro actions:** Macro actions compute button-press counts from the transition model — [arXiv HTML](https://arxiv.org/html/2505.20424v2).
- **Grounding:** SAM, OWL-ViT2, and EasyOCR propose panel elements, and a VLM assigns actions to them — [arXiv HTML](https://arxiv.org/html/2505.20424v2).
- **Closed-loop update:** After each macro action, the robot estimates the state. On a mismatch, it explores the variable's range, updates the transition model, and replans — [arXiv HTML](https://arxiv.org/html/2505.20424v2).
- **Simulation setup:** 30 appliances of 6 types, 300 instructions, and symbolic simulators. Success means finishing within 25 steps — [arXiv HTML](https://arxiv.org/html/2505.20424v2).
- **Simulation success rates:** These are unweighted means over appliance types, computed by the fetch tool from Table 4 — [arXiv HTML](https://arxiv.org/html/2505.20424v2).

  | Method | Success |
  |---|---|
  | ApBot | ≈0.93 |
  | w/o model | ≈0.75 |
  | LLM with grounded actions | ≈0.60 |
  | w/o button policy | ≈0.52 |
  | LLM with image | ≈0.40 |
  | w/o closed-loop update | ≈0.36 |

- **Statistics:** χ² tests were run on all method pairs — [arXiv HTML](https://arxiv.org/html/2505.20424v2).
- **Real world:** A Kinova Gen3 operated a blender, an induction cooker, and a water dispenser. ApBot averaged 0.90 success versus 0.83 for LLM with image; the authors say both perform similarly on these simpler appliances — [arXiv HTML](https://arxiv.org/html/2505.20424v2).
- **Failure analysis:** Of 18 failures, about 83% were modeling errors and about 17% were grounding errors — [arXiv HTML](https://arxiv.org/html/2505.20424v2).
- **Code:** No code or project link was found on the arXiv page — [arXiv](https://arxiv.org/abs/2505.20424).

**SetItUp — "Set It Up": Functional Object Arrangement with Compositional Generative Models (arXiv 2508.02068)**
- **Authors and venue:** Yiqing Xu, Jiayuan Mao, Linfeng Li, Yilun Du, Tomás Lozano-Pérez, Leslie Pack Kaelbling, David Hsu. This is the IJRR journal version of an RSS 2024 paper. v1 was posted 4 Aug 2025 — [arXiv](https://arxiv.org/abs/2508.02068).
- **Representation:** An intermediate grounding graph of abstract spatial relations splits the problem into two parts — [arXiv HTML](https://arxiv.org/html/2508.02068v2):
  - Semantic inference: LLM roles (sketch generator, coder, verifier against 5 examples) induce a Python program that outputs the graph.
  - Geometric grounding: a library of 48 relations, each with a diffusion model trained on synthetic data. These are composed at inference with an annealed ULA sampler.
- **Evaluation design:** 3 task families (dining, bookshelf, bedroom), each with 20 scenes: 5 for training and 15 for evaluation with unseen instructions and objects — [arXiv HTML](https://arxiv.org/html/2508.02068v2).
- **Metrics:** Physical feasibility (% of objects collision-free and inside their container), functionality (% of functional relations satisfied), and human ratings (15 raters) — [arXiv HTML](https://arxiv.org/html/2508.02068v2).
- **Results (dining, physical feasibility / functionality):**
  - SetItUp: 94.0 / 89.1%.
  - Direct LLM with spec: 82.1 / 67.7%.
  - End-to-end diffusion: 22.7 / 21.8%.
  - Single-shot program without coder/verifier: 53.3 / 70.8%.
  - Source: [arXiv HTML](https://arxiv.org/html/2508.02068v2).
- **Other experiments:** Dual-arm robot experiments add reachability constraints, and the induced program is reused for a buffet scene with over 30 objects — [arXiv HTML](https://arxiv.org/html/2508.02068v2).
- **Code:** None was found on the arXiv page — [arXiv](https://arxiv.org/abs/2508.02068).

**ENPIRE — Agentic Robot Policy Self-Improvement in the Real World (NVIDIA GEAR)**
- **Authors and venue:** Wenli Xiao, Jia Xie, Tonghe Zhang, Haotian Lin, Letian Fu, …, Jimmy Wu, Guanzhi Wang, S. Shankar Sastry, Ken Goldberg, Linxi Fan, Yuke Zhu, Guanya Shi (NVIDIA/CMU/UC Berkeley). Accepted to CoRL 2026 — [project page](https://research.nvidia.com/labs/gear/enpire/).
- **Links:** arXiv 2606.19980; code at github.com/NVlabs/ENPIRE — [project page](https://research.nvidia.com/labs/gear/enpire/).
- **Problem:** Coding agents automate algorithm search in purely digital settings. Real-world robotics needs a repeatable loop: reset the scene, run a policy, verify the outcome, refine — [project page](https://research.nvidia.com/labs/gear/enpire/).
- **Harness modules** — [project page](https://research.nvidia.com/labs/gear/enpire/):
  - Environment (EN): reset, safety, verification, and logging APIs.
  - Policy Improvement (PI): writes and revises policy code from rewards, videos, traces, and failures.
  - Rollout (R): budgeted trials, possibly on several robots in parallel.
  - Evolution (E): compares branches, reuses successful recipes, and drops hypotheses that fail.
  - Supporting tools: SAM 3, cuRobo, and a YAM arm. Agents may use heuristics, behavior cloning, and offline or online RL.
- **Agents compared:** Codex (GPT-5.5), Claude Code (Opus 4.7), and Kimi Code (K2.6), on AutoEnvBench — [project page](https://research.nvidia.com/labs/gear/enpire/).
- **Results:**
  - Tasks: Push-T, pin insertion, GPU insertion, zip-tie tying, and zip-tie cutting.
  - Headline: 99% "pass@8."
  - pass@8 here means up to 8 in-context retries per subtask within one long rollout, conditioned on earlier failures. It is not best-of-8 independent samples.
  - Automatic success detection uses multi-view perception fused into a binary reward, and resets are automatic and randomized.
  - The idea-tree ablation reports, for example, +12.5 pp from re-evaluation and +10.8 pp from BC regularization.
  - RoboCasa simulation tasks include Open Drawer and Counter to Cabinet, but no numbers are given on the page.
  - Source: [project page](https://research.nvidia.com/labs/gear/enpire/).
- **Stated limitations:** Robots and GPUs sit idle while agents think, and larger fleets cost more tokens — [project page](https://research.nvidia.com/labs/gear/enpire/).

**ASPIRE — Agentic Skills Discovery for Robotics (NVIDIA GEAR)**
- **Authors and links:** Runyu Lu, Yubo Wu, Ethan Kou, Haotian Lin (equal contribution), …, Ajay Mandlekar, Guanya Shi, Ken Goldberg, …, Jan Kautz, Yuke Zhu, Linxi Fan, Guanzhi Wang (NVIDIA, UMich, UIUC, UC Berkeley, CMU). arXiv 2607.00272; code at github.com/NVlabs/ASPIRE — [project page](https://research.nvidia.com/labs/gear/aspire/).
- **Method:** A self-improving continual-learning system for code-as-policy programs. It has three parts — [project page](https://research.nvidia.com/labs/gear/aspire/):
  1. A closed-loop execution engine that logs the inputs, outputs, and visual evidence of every perception, planning, grasp, and control call. The agent uses these logs to localize failures and re-runs to validate repairs.
  2. A growing skill library that distills validated fixes into reusable skills, retrieved as in-context guidance.
  3. Evolutionary search over programs, with parallel refinement.
- **Evaluation protocol (most relevant to this project):**
  - The agent learns on a small debug split and is scored on disjoint held-out seeds.
  - Seeds per task: LIBERO-PRO, 160 tasks under position and instruction perturbations, 50 seeds per task. Robosuite, 100 seeds per task. BEHAVIOR-1K, 25 seeds per task.
  - Main baseline: CaP-Agent0, which regenerates a program for each seed with test-time retries.
  - A model comparison holds the environment and API fixed.
  - Zero-shot transfer: skills learned on LIBERO-90 are tested on LIBERO-Long.
  - Source: [project page](https://research.nvidia.com/labs/gear/aspire/).
- **Real robot:** Tasks include cap unscrewing, cup stacking, table bussing, and a Rubik's Cube. Skills found in simulation cut the tokens needed to reach a first real success and raise held-out success. Exact numbers appear only in charts — [project page](https://research.nvidia.com/labs/gear/aspire/).
- **Stated limitations:** Needs robust success detection and safe resets, depends on a frozen frontier LLM, is bounded by a predefined primitive API, has an incomplete skill-library lifecycle, and is compute-heavy — [project page](https://research.nvidia.com/labs/gear/aspire/).
- **Inconsistency on the page:** The limitations name Claude Opus 4.6 as the frozen LLM, while the model comparison lists newer models such as Opus 5.5 — [project page](https://research.nvidia.com/labs/gear/aspire/).

**GaP — A Graph-as-Policy Multi-Agent Self-Learning Harness for Variational Automation Tasks (arXiv 2607.05369; CoRL 2026)**
- **Authors:** Kaiyuan Chen, Shuangyu Xie, Letian Fu, Justin Yu, …, Guanzhi Wang, Adithyavairavan Murali, Balakumar Sundaralingam, S. Shankar Sastry, …, Yuke Zhu, Linxi Fan, Ken Goldberg (UC Berkeley, CMU, Bosch, NVIDIA) — [arXiv HTML](https://arxiv.org/html/2607.05369v1).
- **Policy structure:** The policy is a directed computation graph — [arXiv HTML](https://arxiv.org/html/2607.05369v1):
  - Nodes are typed atomic skills drawn from a library of 51 skills, including SAM2/3, Grounding DINO, OWL-ViT, Molmo, Contact-GraspNet, GraspGen, M2T2, and cuRobo.
  - Data edges carry outputs between nodes, and control edges hold conditional branches.
- **Agents:** An Orchestration Agent splits the task into segments. Skill Agents build a subgraph for each segment, and a graph validator checks the wiring — [arXiv HTML](https://arxiv.org/html/2607.05369v1).
- **Self-learning:** Graphs are rehearsed in parallel over N task instances sampled from a belief space in Isaac Sim. Failures are diagnosed from state differences before and after each node, and the graph is revised until performance plateaus — [arXiv HTML](https://arxiv.org/html/2607.05369v1).
- **Execution:** The final graph runs on an interpreter without the agents. The LLM/VLM is Gemini-3.1-Flash-Lite — [arXiv HTML](https://arxiv.org/html/2607.05369v1).
- **Variation design:** This closely matches the project's variant family — [arXiv HTML](https://arxiv.org/html/2607.05369v1):
  - X-Y uniform sampling over a 20×20 cm zone, basket swap, object permutation, and a "mixed_all" setting combining them.
  - Minkowski sums keep sampled objects from colliding.
  - Wash Crates varies yaw by ±15° and position by ±2.5 cm.
  - The paper does not clearly describe a separate held-out train/test split.
- **Simulation results:**

  | Variant | GaP | π0.5 | CaP-X single agent |
  |---|---|---|---|
  | mixed_all | 0.97 | 0.20 | 0.10 |
  | X-Y 20×20 | 0.95 | 0.78 | 0.07 |
  | Pack varied | 0.98 | 0.18 | 0.01 |

  Source: [arXiv HTML](https://arxiv.org/html/2607.05369v1).
- **Real-world results:**
  - Grocery orders: 25/25 versus 8/25 for TiPToP.
  - Packing: 28/30 versus 10/30.
  - Popcorn: 18/20 versus 0/20. The initial graph reached about 33% in simulation, and self-learning raised it to about 94%.
  - USB-C insertion: 121/130.
  - Source: [arXiv HTML](https://arxiv.org/html/2607.05369v1).
- **Ablations:** Both graphless generation (raw Python) and a single consolidated agent dropped to 0 success, mainly from interface mismatches and failed static verification — [arXiv HTML](https://arxiv.org/html/2607.05369v1).
- **Code:** The paper says code and data "will be posted" at graph-robots.github.io/gap — [arXiv HTML](https://arxiv.org/html/2607.05369v1); [project page](https://graph-robots.github.io/gap/).

**VoLo — A Physical Orchestrator for Open-Vocabulary Long-Horizon Manipulation (arXiv 2606.07723)**
- **Authors:** Siyi Chen (first author and project lead) et al., NVIDIA and the University of Michigan — [arXiv HTML](https://arxiv.org/html/2606.07723v1).
- **Method:** A VLM (Claude Opus 4.6) acts as an orchestrator over interruptible tools — [arXiv HTML](https://arxiv.org/html/2606.07723v1):
  - Tools include a π0.5 VLA, SAM3, Molmo2, and GraspGen with IK.
  - The VLA and primitives run at 15 Hz, and the VLM monitors at 0.2 Hz.
  - At each check the agent chooses continue, next_subgoal, or recovery. Recovery can replan, rewrite the VLA's instruction, or run grasp/place primitives.
- **RoboVoLo benchmark:** Built on RoboLab and Isaac Lab, with 126 tasks in 4 suites and 3 fixed-seed episodes per task — [arXiv HTML](https://arxiv.org/html/2606.07723v1).
- **Simulation results:**
  - Overall: full system 41.80%, Only VLA 34.97%, No VLA 17.76%, π0.5 alone 12.57%.
  - Significance: a paired randomization test gives p<0.05 against all baselines except Only VLA (p=0.0598).
  - Source: [arXiv HTML](https://arxiv.org/html/2606.07723v1).
- **Real robot:** Franka FR3, 14 tasks × 3 trials, reported with 95% Wilson CIs — [arXiv HTML](https://arxiv.org/html/2606.07723v1).

  | Method | Success [95% CI] |
  |---|---|
  | Full | 42.9% [29.1, 57.8] |
  | No VLA | 45.2% [31.2, 60.1] |
  | Only VLA | 40.5% [27.0, 55.5] |
  | π0.5 | 14.3% [6.7, 27.8] |

  The authors say the ablation variants are not statistically separable on the real robot.
- **Failure taxonomy:** World failures are wrong object, wrong target, or stuck. VLM failures are planning, monitoring, or tool use — [arXiv HTML](https://arxiv.org/html/2606.07723v1).
- **Code:** GitHub NVlabs/VoLoAgent; project page at chicychen.github.io/VoLo — [GitHub](https://github.com/NVlabs/VoLoAgent); [project page](https://chicychen.github.io/VoLo/).

**Related, found during the search (not on the list): DynaHarness (arXiv 2609.40306, NTU, 30 Sep 2026)**
- **Method:** Assigns each failed episode to one of 13 ordered diagnostic layers. A paired regression gate on development seeds decides whether a revision is admitted — [arXiv HTML](https://arxiv.org/html/2609.40306).
- **Evaluation:** Tuned on development seeds 21–40, then tested on 800 newly sampled states and 261 untouched official states — [arXiv HTML](https://arxiv.org/html/2609.40306).
  - Development: 74.25% versus 16.2% for bare π0.5.
  - New states: 75.2% versus 17.5%.
  - Official untouched states: 77.0% versus 16.5%.
- **Real world:** Includes "drawer placement" at 70% over 10 trials — [arXiv HTML](https://arxiv.org/html/2609.40306).

### Inferences
- **What each paper offers the project:**
  - ProgPrompt and Code as Policies: the program representation and the primitive API.
  - ApBot: an explicit world-model and state-estimate loop that updates on mismatch, plus ablations that remove the closed-loop update.
  - SetItUp: a verifier stage and functional/physical metrics. Its insight that a structured intermediate representation beats direct LLM output applies directly to placing a cup in a drawer, a spatial-relation problem.
  - ASPIRE, GaP, ENPIRE, VoLo: the loop of execute, diagnose, revise, then validate on held-out seeds.
- **Closest analogue:** GaP is the best match for the project's hypothesis. It samples task instances from a belief space of pose and object variations in Isaac Sim, self-improves on them, and transfers to the real world. The project differs by holding out real configurations explicitly and comparing against a single-scene baseline, which GaP does not clearly do.
- **Protocol to copy:** ASPIRE's split between debug seeds and disjoint evaluation seeds (with CaP-Agent0 as baseline), and DynaHarness's dev seeds versus newly sampled and untouched states, are the cleanest templates for the "develop on variants, test on unseen" claim.

### Gaps
- Exact numeric results for ENPIRE and ASPIRE appear only in charts on the project pages and could not be verified as text. The arXiv PDFs (2606.19980, 2607.00272) were not fetched.
- No code repository was found for ApBot or SetItUp. GaP code was "to be posted" when the paper was written.
- The Code as Policies venue (ICRA 2023) was not confirmed by the pages fetched.

## Q2. How do LLM coding agents use simulation feedback to iterate on robot programs or reward functions?

### Takeaway
The common pattern has four steps:
1. Generate executable code (a policy, reward, or task) with the environment's code or API as context.
2. Run it in simulation, often in parallel.
3. Summarize structured feedback: compile and runtime errors, success predicates, reward-component statistics, per-node traces, and failure labels.
4. Revise, with selection or evolution across candidates. The 2026 systems add skill libraries and gate revisions on held-out seeds.

### Cited Findings
- **Eureka (Ma et al., ICLR 2024):**
  - The LLM receives unmodified environment source code and a task description, then runs evolutionary search over reward code. Candidates are scored by GPU-accelerated RL in Isaac Gym.
  - "Reward reflection" summarizes RL training statistics back to the LLM, which then edits hyperparameters, changes the form of reward components, or adds new ones.
  - Results: beats expert human rewards on 83% of 29 tasks, with a 52% average normalized improvement.
  - Source: [project page](https://eureka-research.github.io/); [arXiv](https://arxiv.org/abs/2310.12931).
- **GenSim (Wang et al.):**
  - Generated task code passes a staged filter: syntax check, runtime check, an oracle demonstration that must reach full reward, and an LLM critic run 3 times with agreement required. Optional human review takes under 10 seconds per task.
  - Failures are not added to the library, and an "error book" of common failures is fed back into later prompts.
  - Results: grows the task library from 10 to over 100 tasks, and sim-to-real success rises from 39.2% (CLIPort) to 62.5%.
  - Source: [arXiv HTML](https://arxiv.org/html/2310.01361).
- **RoboGen:** Uses a self-guided propose-generate-learn cycle in which foundation models generate tasks, scenes, and training supervision — [HF paper page](https://huggingface.co/papers/2311.01455).
- **Text2Reward:** Generates dense reward code grounded in a compact Pythonic description of the environment. The code is refined from human feedback on failure modes — [arXiv HTML](https://arxiv.org/html/2309.11489v3).
- **ProgPrompt:** In-plan assertions provide execution-time precondition feedback. Removing them lowers SR from 0.34 to 0.28 — [ar5iv](https://ar5iv.labs.arxiv.org/html/2209.11302).
- **CaP-X (Fu, Yu, Goldberg, Fan et al., ICML 2026):** Contains CaP-Gym, CaP-Bench, CaP-Agent0, and CaP-RL — [PMLR](https://proceedings.mlr.press/v306/fu26u.html).
  - Across 12 models, success falls when the human-designed high-level abstractions are removed.
  - Test-time compute recovers robustness: multi-turn interaction, execution feedback, visual differencing, automatic skill synthesis, and ensembling.
- **ASPIRE:** Logs every perception, planning, grasp, and control call. The agent localizes failures from these traces, repairs the program, validates by re-execution, and distills fixes into a skill library — [project page](https://research.nvidia.com/labs/gear/aspire/).
- **GaP:** Diagnoses failures from state differences before and after each node, across parallel Isaac Sim rehearsals of sampled instances, and iterates until performance plateaus — [arXiv HTML](https://arxiv.org/html/2607.05369v1).
  - The authors warn that single coding agents can hallucinate skills or invent trivial success metrics — [arXiv PDF](https://arxiv.org/pdf/2607.05369).
- **ENPIRE:** Revises policy code from rewards, videos, traces, and failure cases, and uses an evolution module to compare branches. Its biggest single gain in the idea tree, +12.5 pp, came from "re-evaluation" — [project page](https://research.nvidia.com/labs/gear/enpire/).
- **DynaHarness:** Uses a 13-layer failure attribution and a paired regression gate on development seeds before admitting a revision — [arXiv HTML](https://arxiv.org/html/2609.40306).

### Inferences
- **Feedback for the cup-in-drawer loop:** The loop should return structured feedback, not just pass/fail:
  - stage reached (approach, grasp, lift, transport, place in drawer, release)
  - per-primitive inputs and outputs
  - collisions and contacts
  - final cup pose relative to the drawer interior
  - a short failure label for each episode
- **Guard against reward hacking:** The success predicate must be fixed by the humans, not written by the agent. Examples: the cup's center of mass lies inside the drawer volume, the cup is upright, and the gripper is released. This addresses GaP's warning about invented success metrics.
- **Regression gate:** Admit a code revision only if it does not regress on the full development set of variants, as in DynaHarness and ASPIRE. Otherwise agents overfit to the last failure.

### Gaps
- I did not verify whether RoboGen feeds simulation results back into automatic code revision, as opposed to only filtering. AutoRT was not researched in this pass.

## Q3. How do real-to-sim, digital-cousin, and domain-randomization papers measure sim-to-real transfer?

### Takeaway
The standard design has four parts:
1. Compare an exact-twin baseline against twin plus randomization against targeted variation (cousins).
2. Evaluate on a sim twin, on held-out sim setups of increasing distance, and on real hardware.
3. Use 10–50 real trials per condition, reporting success rate (sometimes with bootstrapped standard deviation or Wilson CIs).
4. Stress-test under graded perturbations: pose, distractors or clutter, and physical disturbances.

ACDC is the closest result to the project's hypothesis. The twin-only policy reached 100% in simulation but 25% on the real robot, while twin plus cousins reached 95% on the real robot.

### Cited Findings
- **ACDC (Digital Cousins, arXiv 2410.07408):**
  - Baselines: Twin, Twin+DR (more pose and scale randomization), Twin+Cousin, Cousin only, and "all feasible assets" (naive randomization).
  - Tasks: door opening, drawer opening, and putting away a bowl (open drawer → place bowl → close drawer).
  - Training data: about 10,000 scripted demonstrations (2,000 for the bowl task), kept by rejection sampling. Evaluation instances are excluded from training.
  - Sim-to-sim: 100 rollouts per policy, on the twin plus 3 held-out setups at increasing DINOv2 embedding distance.
  - Sim-to-real is zero-shot, with 50 simulation trials and 20 real trials on a real IKEA cabinet:

    | Policy | Sim | Real |
    |---|---|---|
    | Twin | 100% | 25% |
    | Twin+DR | 70% | 55% |
    | Twin+Cousin | 92% | 95% |
    | Cousin | 94% | 90% |

  - Ablations: number of cousins (2, 4, 8), the visual matching method, and reconstruction pipelines.
  - Code: digital-cousins.github.io.
  - Source: [arXiv HTML](https://arxiv.org/html/2410.07408).
- **RialTo (arXiv 2403.03949):**
  - Tasks include an open drawer and cup in trash.
  - Three graded test levels: (1) randomized object and robot poses, (2) cluttered visual distractors, (3) physical disturbances during the rollout, such as closing the drawer or moving the object.
  - Reports the average over at least 10 rollouts with bootstrapped standard deviation.
  - RialTo versus BC averaged over tasks: 91 vs 25% (pose), 77 vs 11% (distractors), 75 vs 5% (disturbances).
  - On the real target drawer, training on the reconstructed target scene reached 90%, versus 10% when training on 4 untargeted Objaverse drawers.
  - Mug on shelf with distractors: 70% with distractor training, 30% without.
  - Source: [arXiv HTML](https://arxiv.org/html/2403.03949).
- **URDFormer (RSS 2024):**
  - 5 real cabinets with 5 trials per task. Tasks include opening and closing, plus placing and retrieving objects in drawers.
  - Baselines: OWL-ViT plus planning at 0/50, untargeted DR at 9/50, and twin-like URDFormer-ICP at 24/45.
  - URDFormer with targeted randomization reached 39/50 (about 78%), roughly 40% better than DR on average. DR works on simple cabinets but fails on complex ones.
  - Targeted randomization swaps parts and textures while keeping the real object's size and base.
  - Source: [arXiv HTML](https://arxiv.org/html/2405.11656v3).
- **SplatSim:** Gaussian-splat rendering for zero-shot RGB sim-to-real. 86.25% average zero-shot real success, versus 97.5% for policies trained on real data — [arXiv](https://arxiv.org/abs/2409.10161).
- **GaP:** Real trials per benchmark were 20–30, plus 130 for insertion. Simulation used 100 instances per table cell — [arXiv HTML](https://arxiv.org/html/2607.05369v1).
- **VoLo:** 14 tasks × 3 real trials, with 95% Wilson CIs and paired randomization tests in simulation — [arXiv HTML](https://arxiv.org/html/2606.07723v1).
- **ApBot:** χ² tests across all method pairs — [arXiv HTML](https://arxiv.org/html/2505.20424v2).
- **Statistics references:**
  - Clopper-Pearson (exact) interval example: 90% success over 70 rollouts gives a 95% CI of about 80.5–95.9% — [NVIDIA blog](https://developer.nvidia.com/blog/how-to-evaluate-general-purpose-robot-policies-for-real-world-deployment/).
  - STEP (Snyder et al., RSS 2025): a sequential test for comparing two policies that cuts evaluation trials by up to 32% while keeping correctness and power — [arXiv](https://arxiv.org/abs/2503.10966).
  - SureSim (arXiv 2510.04354): combines large-scale simulation with a few paired real trials via prediction-powered inference, giving CIs that correct for simulator bias and saving 20–25% of hardware effort — [arXiv](https://arxiv.org/abs/2510.04354).
- **Wilson 95% intervals at typical trial counts** (computed for these notes with the standard Wilson formula, z=1.96):

  | Successes | 95% Wilson CI |
  |---|---|
  | 10/10 | 72.2–100% |
  | 9/10 | 59.6–98.2% |
  | 7/10 | 39.7–89.2% |
  | 18/20 | 69.9–97.2% |
  | 15/20 | 53.1–88.8% |
  | 10/20 | 29.9–70.1% |
  | 27/30 | 74.4–96.5% |
  | 45/50 | 78.6–95.7% |
  | 40/50 | 67.0–88.8% |
  | 90/100 | 82.6–94.5% |
  | 180/200 | 85.1–93.4% |

  The 7/10 case matches the 39.7–89.2% reported in [particula.tech](https://particula.tech/blog/how-many-trials-evaluate-robot-policy).

### Inferences — proposed evaluation protocol for the cup-in-drawer project

**Conditions (the agent loop, LLM, primitive API, and iteration/token budget stay the same; only the development scene distribution changes):**
- **B0, one-shot:** Code-as-Policies-style single generation with no simulation feedback.
- **B1, digital-twin only:** The agent iterates using only the original reconstructed scene. This is the main baseline, matching ACDC "Twin" and RialTo's single target scene.
- **B2, twin plus naive domain randomization:** Uniform random noise on all axes, not targeted. Matches ACDC Twin+DR and URDFormer's DR.
- **M, variant family (proposed):** Structured variants along cup pose, drawer opening, and clutter, as in GaP's belief-space sampling and ACDC's cousins.
- **Optional, M with skill memory:** In the style of ASPIRE.

**Development and test split:**
- Generate the variant family from parameterized axes:
  - cup x-y offset, for example within 20×20 cm (as in GaP), and yaw
  - drawer opening, for example 30–100% of travel
  - clutter: number, identity, and placement of distractors in the path or the drawer
- Development set: K variants, for example 20–50 seeds.
- Held-out test set: disjoint seeds, plus out-of-range extrapolation cells such as a smaller drawer opening than any seen in development, new clutter objects, or a combined shift. Following ACDC, order the held-out cells by distance from the original scene.

**Metrics:**
- Primary: binary success. The cup rests inside the drawer interior volume, stays upright, the gripper is released, and nothing has fallen.
- Partial progress: the stage reached (grasped, lifted, over the drawer, inside), in the spirit of ProgPrompt's GCR.
- Safety: collisions (count of contacts between robot and drawer or clutter, objects knocked over or displaced more than a threshold) and execution errors.
- Efficiency: execution time.
- Development cost: agent iterations, tokens, and wall-clock time to reach the success threshold. ENPIRE and ASPIRE report cost and utilization in this way.
- Generalization gap: development success minus held-out success. Report it per condition; the twin-only gap should be large, as in ACDC (100% to 25%).

**Ablations (which variation axes matter):**
- Leave one axis out: the family without cup-pose, without drawer-opening, or without clutter variation.
- Single-axis families.
- Family size: K = 1, 4, 8, 16, …, matching ACDC's cousin count.
- Feedback richness: pass/fail only versus stage and failure labels versus full traces.
- With and without a regression gate on the development set.
- RialTo-style stress test at evaluation time: physically disturb the scene by nudging the cup or partly closing the drawer.

**Trials and statistics:**
- Simulation: at least 100 held-out episodes per condition and per cell, as in ACDC (100 rollouts) and ASPIRE (50–100 seeds per task). That keeps 95% CI half-widths at about ±6–10 pp.
- Repeat the agent development run with 3 or more random seeds or LLM samples. Agent loops vary from run to run, and ACDC uses 6 runs.
- Real robot, if available: 10–20 trials per condition on matched held-out configurations, with the same physical configurations for every condition. Report k/n with Wilson 95% CIs.
- Comparisons: a paired test where the same configurations are used (McNemar's test, or a paired randomization test as in VoLo); χ² or Fisher's exact test otherwise (ApBot uses χ²).
- Consider STEP for sequential stopping, and SureSim-style paired sim and real trials, to tighten estimates of real-world success.
- Expect to separate only large effects: at 20 trials each, 18/20 versus 10/20 (70–97% versus 30–70%) is distinguishable, but 18/20 versus 15/20 is not.

### Gaps
- ACDC does not describe its real held-out configurations beyond the single IKEA cabinet. RialTo's per-task randomization ranges were in a section that was not fetched.
- SplatSim's task list and trial counts were not extracted, only the abstract.
- No paper found evaluates a coding agent's program, rather than a learned policy, as developed on a single twin versus a variant family with real held-out tests. The project's exact comparison appears to be new; ACDC and RialTo test this for learned policies only.

## Q4. How should a simulation-only proxy evaluation be designed when real-robot access is limited?

### Takeaway
Use a sim-to-sim protocol in which a perturbed target scene stands in for "unseen real." ACDC already does this: it evaluates on the twin plus 3 held-out setups of increasing distance. ASPIRE and DynaHarness use a debug/dev split and disjoint evaluation seeds. Make the proxy target differ from the development scenes in ways that mimic reconstruction error, not just new samples of the variation parameters.

### Cited Findings
- **ACDC:** Sim-to-sim evaluation uses 100 rollouts on the exact twin plus 3 held-out setups at increasing DINOv2 distance, explicitly as an out-of-distribution proxy. Evaluation instances are excluded from training — [arXiv HTML](https://arxiv.org/html/2410.07408).
- **ASPIRE:** Learns on a small debug split and is scored on disjoint held-out seeds (25–100 per task) — [project page](https://research.nvidia.com/labs/gear/aspire/).
- **DynaHarness:** Uses three tiers — development seeds, 800 newly sampled states after freezing, and 261 untouched official states — and reports all three (74.25 / 75.2 / 77.0%) — [arXiv HTML](https://arxiv.org/html/2609.40306).
- **LIBERO-PRO:** Used by ASPIRE, DynaHarness, and CaP-X as a perturbation benchmark (position and instruction perturbations, object or basket swaps). It shows that policies near ceiling on nominal LIBERO collapse under these perturbations — [GaP arXiv HTML](https://arxiv.org/html/2607.05369v1); [DynaHarness arXiv HTML](https://arxiv.org/html/2609.40306).
  - In GaP's table, π0.5 scores 0.96 on LIBERO object but 0.20 on mixed_all.
- **Imperfect-simulator correction:** SureSim shows that a few paired real trials can correct a simulator's bias to give valid CIs on real performance — [arXiv](https://arxiv.org/abs/2510.04354).
- **GaP real-world gains:** GaP's simulation self-learning raised Make Popcorn from about 33% to about 94% in simulation, and the result held at 18/20 on the real robot — [arXiv HTML](https://arxiv.org/html/2607.05369v1).

### Inferences
- **Proxy design:**
  - **Source scene S0:** The reconstructed kitchen, used as given.
  - **Development family F:** Variants of S0 along the 3 axes.
  - **Proxy real targets T1…Tn:** Held-out scenes that also apply "reality gap" perturbations never used in development:
    - mesh/pose noise of a few centimeters and a few degrees on the drawer and cabinet, to simulate reconstruction error
    - different drawer friction, damping, and handle geometry
    - different cup mesh and mass, and different table height
    - camera extrinsics noise, and different lighting or textures if perception is in the loop
    - unseen clutter objects
  - The perturbation parameters for T are fixed and kept secret from the agent loop, so no target leaks into development.
- **What to report:**
  - Success on T for the twin-only policy (B1) versus the variant-family policy (M), with Wilson CIs, along with the full metric set from Q3.
  - The sim-to-sim gap per condition: success on S0 or the development set minus success on T.
  - The predicted pattern from ACDC and URDFormer: B1 is high on S0 and drops on T, while M is slightly lower on S0 and much higher on T.
- **Pre-registration:** Fix the held-out target set and success predicate before running any agent iterations. Log every agent revision so the authors can show that test scenes were never seen.
- **Small real check:** If any real access exists, run a few real trials (for example 10 per condition on 2–3 configurations) as a sanity check of the proxy ranking. SureSim's paired-trial approach is the principled way to combine them.
- **Demo order:** First, S0 plus a small variant family plus 3–5 held-out perturbed targets, with the B1 versus M comparison. Then add the axis ablations.

### Gaps
- I found no published standard for how much perturbation a proxy target needs to predict the real reality gap. Choosing perturbation magnitudes, for example from measured reconstruction error, remains a project design decision.
- I did not find work that validates sim-to-sim proxy rankings against real rankings for code-as-policy agents specifically. SureSim addresses learned policies.
