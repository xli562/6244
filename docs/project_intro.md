# Project Intro: Real-to-Simulation and Scene Variation

**Next action for the team:** agree on the midpoint-check status (Mon 19 Oct) using the milestone table in section 5.

Source: [Project Reference for Agentic Robotics](https://docs.google.com/document/d/1-E_lPR6wKXm5POH6zqdR3vsvybdHcCFnVygHtACUlHI/edit), Project 1. Course-level requirements are in [project_requirement.md](project_requirement.md).

## 1. One-line goal

Turn one real task scene into **many interactive simulation scenes**, so that robot behaviors developed in them work better on new real-world setups.

| | |
| --- | --- |
| **Input** | A captured real scene (e.g., a kitchen with a drawer and a cup) |
| **Output** | A family of interactive simulation environments: the reconstructed scene plus task-preserving variants |
| **Deliverable** | A pipeline that runs input → output |
| **Success** | Solutions developed in the variants beat solutions developed in the original scene only, on **unseen real-world configurations** |

## 2. Where this fits

The reference doc splits agentic robotics into two modes. We are in **offline mode** (agent-driven policy development):

1. User gives an instruction
2. Agent generates or updates a program or policy
3. Program runs **in simulation** ← our project builds these environments
4. Outcomes are evaluated and failures diagnosed
5. Feedback goes back to the agent; repeat

Our pipeline supplies step 3. Without varied, interactive scenes, the agent can only overfit to one setup.

## 3. Test case: place a cup in a drawer

Start from a captured kitchen scene. The reconstruction must:

1. Make the **drawer move** (articulated joint)
2. Make the **cup interact with surfaces** (contact, collision, gravity)
3. **Preserve the clearances** that decide whether placement succeeds

Generate variants along three axes:

- Cup position
- Drawer opening
- Surrounding clutter

## 4. Research questions

1. **Reconstruction:** How do we recover task-relevant geometry, physical properties, object interactions, and articulation?
2. **Variation:** How do we generate diverse variants while keeping the task the same?
3. **Fidelity:** Which parts of simulation fidelity matter most for transfer to the real world?

## 5. Draft milestones

Draft only. The reference doc does not set milestones; edit to match what the team has done. The course asks for 3–6 milestones of 1–2 weeks each.

| # | Milestone | Done when | Estimate |
| --- | --- | --- | --- |
| M1 | Capture and reconstruct the static scene | Scene geometry loads in the simulator | 1–2 weeks |
| M2 | Make the scene interactive | Drawer slides, cup rests and collides correctly | 1–2 weeks |
| M3 | Variant generator | N variants generated, each still solvable (clearances checked) | 1 week |
| M4 | Develop solutions | One solution from original scene only, one from the variant family | 1–2 weeks |
| M5 | Real-world evaluation | Both solutions tested on unseen real configurations; success rates compared | 1 week |

## 6. Environment ready now

- Conda env `robot` (Python 3.11) with **MuJoCo 3.16.0**
- Headless GPU rendering works (`MUJOCO_GL=egl` is set in the env)
- Check: `conda activate robot && python -c "import mujoco; print(mujoco.__version__)"`

## 7. Reading list

From the reference doc.

1. [SetItUp](https://arxiv.org/abs/2508.02068)
2. [ENPIRE](https://research.nvidia.com/labs/gear/enpire/) (NVIDIA GEAR)
3. [ASPIRE](https://research.nvidia.com/labs/gear/aspire/) (NVIDIA GEAR)
4. [ProgPrompt](https://arxiv.org/abs/2209.11302)
5. [ApBot](https://arxiv.org/pdf/2505.20424)

## 8. Open questions for the team

1. Capture method: which sensor and reconstruction tool?
2. How do we get physical properties (mass, friction, joint limits) — estimate, measure, or assume?
3. How many variants, and how do we check each one is still solvable?
4. What "solution" do we compare: a generated program, a learned policy, or both?
5. Which real robot and how many unseen configurations for the final test?
