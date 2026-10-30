---
name: improve-docs
description: Run one TensorOperations documentation improvement end to end: find the most useful gap for a human reader, write it, review it, and open a draft PR for the user. Use for the full workflow, not an isolated doc edit.
---

# Improve repository documentation

You coordinate one contribution through three agents, in order, and deliver it as
a draft PR. You sequence and hand off. You do not write or review docs yourself.
Follow AGENTS.md and the user's rules: never commit or open a PR outside this
workflow's delivery step, never mark a PR ready, never merge.

Agents (in `.claude/agents/`): `docs-finder` (Opus), `docs-writer` (Sonnet),
`docs-reviewer` (Opus). If a named agent is unavailable, spawn a general-purpose
agent with that file's body as its instructions and the file's model. Each
agent starts in fresh context. Give it only the repo root, its role, and the
stage's handoff, never the previous agent's transcript. Run one stage at a time
and wait for its final result.

Documentation is for people. Verification evidence (what was run, where, what
was not) goes in agent handoffs, the task file and the PR description, never in
the docs themselves. Reject a contribution that narrates its own testing or
answers review comments in the text, and send it back as a finding.

## Steps

1. **Prepare.** Note `git status` so pre-existing changes are excluded. Create a
   worktree and branch `docs/<topic>` once the topic is known (step 2). Stop if
   the agent files are missing.
2. **Find.** Run `docs-finder`. `NO_TASK_CREATED`: report the scope it walked and
   finish. `TASK_CREATED`: check the file exists, keep the path, create the
   worktree/branch, and carry the task file into it.
3. **Write.** `revision_pass = 0`. Run `docs-writer` with the task path.
4. **Review.** On `READY_FOR_REVIEW`, run `docs-reviewer` with the task path, the
   contribution (include untracked files), the report, and `revision_pass`.
5. **Loop.** `APPROVED`: go to Deliver. `CHANGES_REQUESTED` below pass 3: increment
   the counter, send findings to `docs-writer`, then re-review with the same
   counter. At pass 3, stop looping and go to Deliver with findings unresolved.
   That is one initial round plus up to three revisions. Never relax criteria to
   get approval.

## Stops

`TASK_NEEDS_CORRECTION`, `NEEDS_INPUT`, `INCOMPLETE`: stop, return the evidence or
missing input, and leave work local. Do not rewrite acceptance criteria. If a
result lacks its handoff, ask that agent once; do not retry indefinitely.

## Deliver

Applies on `APPROVED` or after pass 3.

1. Update the task's `Status:` to `approved` or `unresolved`, and add a
   `Workflow result` section with the verdict and revision count. Keep the
   original problem, evidence and criteria.
2. Commit only the contribution and the task file (not unrelated changes), with
   the commit attribution line from the session instructions.
3. Push the branch and open a **draft** PR against `main` using the PR template
   below, ending with the PR attribution line from the session instructions.
4. Never mark it ready or merge. Report the PR URL.

### PR description

    ## What a reader can do now
    One plain paragraph: who benefits, and what they can now do that they
    could not before. No tag talk.

    ## Where
    Files changed, and why those pages or comments.

    ## How it was checked
    Examples compiled and run, claims verified against which source and tests,
    docs build result, and what could not be checked (for example no CUDA).

    ## Review notes
    Revision count, reviewer verdict, and unresolved findings if any. If the
    review ended at pass 3 unresolved, say so first.

## Output

Return the task path, files changed, checks run, verdict, revision count, PR URL
(or why none), and unresolved findings. Each run ends after one contribution.
