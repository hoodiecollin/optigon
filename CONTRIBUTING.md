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

When you commit to a piece of work, file the issue *first* (`tech-debt` for a grounded
gap in shipped code, `idea` for a speculative feature), then implement.

## Two axes, and nothing else

Work is organized by exactly two orthogonal axes:

| Axis | Answers | Mechanism |
|---|---|---|
| **Milestone** | *when* | `v0.1.0`, `v0.2.0`, … — the release spine |
| **Labels** | *what kind / how mature* | the taxonomy below |

Nothing else decomposes work. Epics decompose through GitHub **native sub-issues** —
not task-list checkboxes, not a Project field. The [Optigon Roadmap
board](https://github.com/users/hoodiecollin/projects/7) is a **view**, never a second
source of truth, and it deliberately has no Priority / Size / Workstream fields: those
are a parallel truth that drifts.

## The commitment ladder

```
speculative      committed         scheduled          shipping           shipped
──────────       ─────────         ─────────          ────────           ───────
label: idea  →   label: plan-next → milestone assigned → merged/closed  → GitHub Release
(needs an RFC)   (unscheduled)      (drop plan-next)     (into milestone)  (roadmap flips)
```

| Label | Means |
|---|---|
| `idea` | Speculative. Needs a design-doc before implementation. |
| `plan-next` | Committed, not yet scheduled to a version. |
| `rfc` | A design captured as an issue (Gate 1). |
| `experiment` | A spike to measure; the deliverable is a decision, never an artifact. Never milestoned. |
| `epic` | Umbrella; decomposes via native sub-issues. |
| `tech-debt` | Known gap or stub in shipped code. |
| `perf` | Performance cost / triage item. |
| `config` | Configurable-runtime-behavior work. |
| `legacy-audit` | Prune dead / product-misaligned code. |
| `release-gate` | Blocks the tag: this milestone cannot be released until it closes. |

### Invariants (CI enforces these)

- **`plan-next` ⊕ milestone.** Assigning a milestone *is* scheduling — drop `plan-next`.
- **`idea` ⊕ `plan-next`.** Speculative and committed are opposites.
- **`experiment` ⊕ {`idea`, `plan-next`, milestone}.** A spike never rides the spine.
- **`release-gate` ⇒ milestone**, and never with `idea` / `plan-next` / `experiment`.

`.github/workflows/playbook.yml` runs `pm-playbook check` on every PR, so a violation
fails review rather than being discovered months later. Run it yourself with:

```bash
npx @hoodiecollin/pm-playbook check --repo hoodiecollin/optigon
```

## Nothing gets coded until design → plan → spec

Three gates, in series, before implementation:

1. **Gate 1 — design-doc (WHAT & WHY).** An **`rfc` issue**: problem, desired behavior,
   solution *shape*, alternatives, explicit non-goals. **Design lives as an issue, never
   as a committed `proposal-*.md`.** The only design docs in this tree are durable
   architecture references for *shipped* features — that is what `docs/architecture.md`
   is. Accepted → drop `idea`, add `plan-next`.
2. **Gate 2 — implementation-plan (HOW).** Written after the design is accepted and the
   item is scheduled: files to touch, build order, blockers, interfaces, and the
   scenarios to write. Lives on the issue.
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
