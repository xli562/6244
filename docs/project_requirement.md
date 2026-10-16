# Project Requirements: Agentic Robot Programming

CS6244 — Agentic Robot Intelligence: Foundation Models & System Architectures (NUS, 2026/27)

Source: [course information doc](https://docs.google.com/document/d/1zKAHWqZ3xgrANrOwaIPpFragDVZCldRmc_jiDroq3XM/edit), section *Projects → Agentic Robot Programming*. Everything under sections 1–5 is taken from that doc; section 6 lists what the team still has to decide.

## 1. Course context

| Item | Detail |
| --- | --- |
| Instructor | David HSU (dcsdavid@nus.edu.sg) |
| Project mentor | Cunjun YU (cunjun.yu@gmail.com) |
| Weight | Research project is ~60% of the course grade (team work) |
| Duration | About 8 weeks |
| Consultation | Monday 12:00–1:00 pm, or by appointment |

The course states that the main objective of the project is to **learn**, including from failures. Teams are not expected to advance the state of the art, though that would be welcome. Students from the most successful teams will be invited to apply for internships with **Nvidia Research Singapore**.

## 2. Project description

Coding agents have shown strong capabilities in building complex software systems, which has raised interest in using them for robot programming. The project asks:

- What is the role of coding agents in robot programming?
- How does that role differ from their role in general computer programming?

### Issues to explore

1. **User intent understanding and task specification.** Given a user's natural-language instruction or prompt, how do we specify the task so that it meets the user's intent?
2. **Structural prior for code generation.** What kinds of programs should be generated for robots? What constraints should be imposed on them?
3. **Verification and agentic feedback.** How do we verify that the generated code is correct? What feedback can improve the generation and close the agentic loop?

## 3. What the team must do

- **Choose a specific robot task.**
- **Develop an agentic harness that automates the robot system development** for that task.

### Suggested starting points

Prior work in which LLMs generate the program representation, but with the surrounding process done manually:

- ProgPrompt: Generating situated robot task plans using large language models — Singh et al., 2023 ([arXiv:2209.11302](https://arxiv.org/abs/2209.11302))
- ApBot: Robot Operation of Home Appliances by Reading User Manuals — 2025 ([arXiv:2505.20424](https://arxiv.org/abs/2505.20424))

The goal is to automate what these works do by hand.

### Related course readings

- Week 4 (modular systems): *Do As I Can, Not As I Say* (SayCan); *ProgPrompt*; *TidyBot*
- Week 8 (reasoning): *SayPlan*; *Video Language Planning*; *Inner Monologue*
- Week 9: *Correcting Robot Plans with Natural Language Feedback*
- Week 11: *LLMs Can't Plan, But Can Help Planning in LLM-Modulo Frameworks*
- Week 12 (agentic programming for robots): *Code as Policies*; *GaP: A Graph-as-Policy Multi-Agent Self-Learning Harness for Variational Automation Tasks*; *VoLo: A Physical Orchestrator for Open-Vocabulary Long-Horizon Manipulation*

## 4. Project proposal requirements

- Length: 1–2 pages, single-spaced, Times 11 pt.
- Scope: a project of about 8 weeks.
- Required structure:
  - **Objectives and Motivation**
    - Define the problem as clearly as possible, including its **input and output**.
    - Explain why the project is interesting or exciting (applications, advancing the state of the art, answering long-standing questions, etc.).
    - State the potential challenges.
  - **Plan**
    - Describe the main proposed solution idea, even if preliminary.
    - Lay out the main steps, for example **3–6 milestones of 1–2 weeks each**.
  - **Deliverables**
    - Describe the expected outcome as clearly as possible.
    - State **how the system's performance will be evaluated**.
- Submission: a single PDF on Google Drive with **comment access for all**, linked in the course [submission sheet](https://docs.google.com/spreadsheets/d/1mQNIFm9x7Lkn09LWumO08IkV-UXhNLdjJtni2tGiKM0/edit?gid=1472013509#gid=1472013509) under the tab "Project Submission".

## 5. Timeline

| Week | Date | Milestone |
| --- | --- | --- |
| 4 | 31 Aug | Team formation starts (complete by 11 Sep) |
| 6 | 14 Sep | Team formation completed |
| 7 | 28 Sep | **Project proposal due** |
| 10 | 19 Oct | **Project midpoint check** |
| 12 | 2 Nov | Lecture: agentic programming for robots |
| 13 | 9 Nov | Public holiday |

The course doc does not yet state the final deliverable format or due date. Confirm both with the instructor or mentor.

## 6. Open decisions for the team

Not specified by the course; to be filled in by the team and agreed with the mentor.

- [ ] Target robot task and its input/output definition
- [ ] Task specification format (how a natural-language instruction becomes a checkable spec)
- [ ] Program representation and constraints (the structural prior)
- [ ] Verification method and feedback signals for the agentic loop (e.g., simulation rollouts, tests, execution traces)
- [ ] Execution platform (simulator and/or real robot)
- [ ] Evaluation metrics and baselines (e.g., a manually prompted ProgPrompt/ApBot-style pipeline)
- [ ] Milestone plan (3–6 milestones of 1–2 weeks each)
- [ ] Final deliverable format and due date (to confirm)
