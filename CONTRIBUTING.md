# Contributing to Optigon

This project runs on the [pm-playbook](https://github.com/hoodiecollin/ai-pm-playbook)
model. The full doctrine is vendored at [`.pm-playbook/PLAYBOOK.md`](./.pm-playbook/PLAYBOOK.md),
with a router at [`.pm-playbook/AGENT.md`](./.pm-playbook/AGENT.md) — that copy is
authoritative and version-pinned. This file is the newcomer's summary of it.

## The backlog is GitHub Issues

There is no `TODO.md`, no `TASKS.md`, and no roadmap file that lists work. Ask "what's
next" with `gh issue list`, not with a file. `docs/architecture.md` describes how
shipped code works; [`VERSION_ROADMAP.md`](./VERSION_ROADMAP.md) and
[`WHAT_IT_IS.md`](./WHAT_IT_IS.md) describe state and guarantees. None of them are a
task list.

When you commit to a piece of work, file the issue *first* — with exactly one work type
(`improvement`, `bugfix` or `experiment`) — then implement. Leave it unmilestoned if it is
speculative; that is what "idea" means now, and it is derived rather than stuck on.

## Two axes, and nothing else

Work is organized by exactly two orthogonal axes:

| Axis | Answers | Mechanism |
|---|---|---|
| **Milestone** | *when* — assigning one means **committed**; being the cycle in flight means *scheduled* | `v0.1.0`, `v0.2.0`, … — the release spine |
| **Labels** | *what kind* | the taxonomy below |

Nothing else decomposes work. Epics decompose through GitHub **native sub-issues** —
not task-list checkboxes, not a Project field. The [Optigon Roadmap
board](https://github.com/users/hoodiecollin/projects/7) is a **view**, never a second
source of truth, and it deliberately has no Priority / Size / Workstream fields: those
are a parallel truth that drifts.

## The commitment ladder

The ladder is **derived**, not labelled. Every work item carries exactly one *type*; the type
decides its gate sub-issues; and the first gate that is not closed says where the work sits.

```
idea → design-next → design-pending → plan-next → plan-pending
     → impl-next → impl-pending → closed-in-milestone → released
```

Those are rung *names*, not labels — there is nothing to set and nothing to forget to unset, and a
rung can never disagree with the artifacts it summarizes. No GitHub filter can compute it, so ask:

```bash
npx @hoodiecollin/pm-playbook ladder
```

| Label | Means |
|---|---|
| `improvement` | Makes the product better: features, refactors, perf, debt. Gates: design → plan → impl. |
| `bugfix` | A defect in behavior that already exists. Gates: diagnose → fix. |
| `experiment` | The deliverable is a finding, not a shippable artifact. Gates: research → evaluate. Never milestoned. |
| `hotfix` | Urgent `bugfix` in released behavior, on its own patch milestone. Never alone — always with `bugfix`. |
| `epic` | Umbrella; decomposes via native sub-issues. Not a work type, and never carries gates. |
| `release-gate` | Blocks the tag: this milestone cannot be released until it closes. |
| `{type}:gate-{n}` | A gate sub-issue. Created by `materialize`, **never by hand**. |

### Invariants (CI enforces these)

- **Exactly one type label** per work item — never zero, never two (PM010).
- **`experiment` ⊕ milestone.** A spike never rides the release spine (PM003).
- **A gate's milestone equals its parent's** (PM011), and an `epic` never carries gates (PM012).
- **Work on the focused milestone carries its complete gate set** (PM013).
- **`release-gate` ⇒ milestone**, and never with `experiment` (PM004/PM005).

`.github/workflows/playbook.yml` runs `pm-playbook check` on every PR, so a violation
fails review rather than being discovered months later. Run it yourself with:

```bash
npx @hoodiecollin/pm-playbook check --repo hoodiecollin/optigon
```

## Nothing gets coded until the gates are closed

Gates are **native sub-issues** of the work item, created by `pm-playbook materialize` as a
complete set — **never by hand**, because a hand-made gate destroys the only thing that makes an
*absent* gate meaningful. An `improvement` takes three; a `bugfix` takes two (diagnose → fix).

**Closing a gate means accepted.** That is the whole signalling mechanism — there is no status
label, and the ladder is derived from which gates are closed.

1. **Gate 1 — design (WHAT & WHY).** Problem, desired behavior, solution *shape*, alternatives,
   explicit non-goals. **Design lives on the gate issue, never as a committed `proposal-*.md`.**
   The only design docs in this tree are durable architecture references for *shipped* features —
   that is what `docs/architecture.md` is.
2. **Gate 2 — plan (HOW).** Written after the design gate closes: files to touch, build order,
   blockers, interfaces, and the scenarios to write.
3. **Gate 3 — spec-first, RED → GREEN.** Write the failing tests, implement to green,
   refactor under green. For Optigon that means `cargo test` at the core level plus the
   binding-level demos in `examples/`.

If you reopen an accepted gate, **purge the issue body first** and replace it with the
withdrawal placeholder (§9.1). A superseded design left in a body does not read as
superseded — it reads as *the* design, and the next planner builds on it.

## Prioritize on engineering merit, never demand

Never justify building or deferring on "demand", "usage", or "when users want it" —
Optigon is pre-launch and those signals do not exist. Justify on scope, risk,
foundational sequencing (does X unblock Y?), identity fit, and the in-codebase YAGNI
test: does another crate or a binding actually call this?

The domain backlog is sequenced on exactly that basis — see the ordering rationale on
each `domain: *` issue and on epic #11.

## How branches land

**Merge commits, always.** Squash and rebase are disabled in optigon's GitHub settings, so a
merge commit is the only method the merge button will accept. A branch is a unit of work and
its history is worth keeping — a squash throws away the story the commits were split up to
tell, and a rebase erases the fact that the work happened on a branch at all.

The subject line names the branch and what it did, with the issue in parentheses:

```bash
git merge --no-ff <branch> -m "Merge <branch>: <what it did> (#<issue>)"
```

`--no-ff` is load-bearing. Without it a branch that is merely ahead fast-forwards, and the
branch boundary disappears exactly as if it had been rebased.

Work merges into `main`, which is the default branch, so a `Closes #<n>` line in the PR body
closes the issue on merge. That stops being true the moment an integration branch is
introduced — GitHub honours closing keywords only for PRs targeting the *default* branch, and
an issue merged into an integration branch stays open however the body is written.

## Local checks before you push

Everything runs from the repo root — no `cd` required (`make help` lists every target):

```bash
make test     # cargo test across the workspace
make lint     # biome + clippy
make fmt      # biome format + cargo fmt
```
