# Agent instructions

<!-- pm-playbook:begin -->
## Project management — pm-playbook v1.1.0

Issue tracking in this repo follows the **pm-playbook** two-axis model. The full doctrine is
vendored at `.pm-playbook/` and is authoritative; this block is only a summary.

**Before you create, label, milestone, or close an issue — read `.pm-playbook/AGENT.md`.**
It is a short router: load only the reference section relevant to what you are doing.

**The two axes, and nothing else, organize work:**
- **Milestone** = *when* (a version release — the release spine). Assigning one means "scheduled."
- **Labels** = *what kind / how committed*. Epics decompose via **native sub-issues**, never
  checkboxes and never a Project field.
- There are **no Priority / Size / Workstream fields**. Do not propose adding any.

**Invariants — violating one is a bug, not a style preference:**
- `plan-next` and a milestone never coexist. Assigning a milestone means dropping `plan-next`.
- `idea` and `plan-next` never coexist.
- `experiment` never carries `idea`, `plan-next`, or a milestone. A spike's deliverable is a
  decision; it feeds the release spine, it never rides it.
- `release-gate` always has a milestone, and never carries `idea` / `plan-next` / `experiment`.
  An open `release-gate` means its milestone **cannot be tagged**.
- A non-core `surface:*` issue never rides a core `v*` milestone.

**Verify before opening a PR** — exit code 0 means compliant:

```bash
npx @hoodiecollin/pm-playbook check
```
<!-- pm-playbook:end -->
