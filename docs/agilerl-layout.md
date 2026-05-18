# AgileRL layout policy (FoL repository)

This repository ships **modified AgileRL** code (for example `maddpg_local_abrl`, `maddpg_abrl`) under **`AgileRL/`**. That is **fully supported** and remains the default workflow: install with `pip install -e ./AgileRL` from the repo root (see `README.md`, `Dockerfile`, and CI).

The question is only **how `AgileRL/` is represented in Git** when you collaborate or publish. Pick **one** policy below and stick to it for that repo.

---

## Option A — Vendored tree (recommended if you want a single clone with no submodule steps)

**What it is.** `AgileRL/` is normal tracked files inside this repo (no submodule pointer).

**When to use.** You want contributors to run `git clone` once and have everything; you are fine carrying AgileRL diffs in this repo’s history.

**Policy.**

1. **Remove the nested repository** inside `AgileRL/` before publishing, so you do not accidentally ship two repos in one tree:
   ```bash
   rm -rf AgileRL/.git
   ```
2. Commit the resulting files as usual. Document the **upstream AgileRL commit** you last merged from (optional but good practice) in this file or in `AgileRL/README.md` (e.g. “Based on upstream commit `abc1234`”).

**Pros.** Simplest for readers and CI. **Cons.** Larger repo; you must merge upstream AgileRL manually when you want updates.

---

## Option B — Git submodule (recommended if you want a clear fork boundary)

**What it is.** `AgileRL/` is a **submodule**: this repo stores a pointer to a **specific commit** on another Git remote (almost always **your fork** of AgileRL, because you have local modifications).

**When to use.** You want `AgileRL` changes reviewed in a separate repository, or you want to pin an exact revision and update it deliberately.

**Prerequisites.**

- A **remote Git repository** that contains your modified AgileRL (your fork on GitHub/GitLab, or an internal mirror). Upstream alone is not enough if your tree differs.

**High-level steps** (run from the **parent** repo root; adjust remote URLs and branch names).

1. Push your current `AgileRL/` contents to your fork (one-time or when migrating), e.g. create a new repo `my-org/AgileRL-fol` and push the `AgileRL` subtree or a fresh clone you’ve copied into.

2. Remove the existing directory from the parent index (this does **not** delete your work if you already pushed it to the fork):
   ```bash
   git rm -rf AgileRL
   ```

3. Add the submodule at the same path:
   ```bash
   git submodule add https://github.com/YOUR_ORG/YOUR_AGILERL_FORK.git AgileRL
   git submodule update --init --recursive
   ```

4. Commit `.gitmodules` and the submodule gitlink. In `README.md`, tell clones to run:
   ```bash
   git submodule update --init --recursive
   ```
   or `git clone --recurse-submodules …`.

5. **CI:** use `actions/checkout` with submodules enabled (this repo’s `.github/workflows/smoke.yml` already sets `submodules: recursive` so `AgileRL` is present for `pip install -e ./AgileRL` once you switch to a submodule).

**Pros.** Clear separation; pin exact AgileRL revision. **Cons.** Extra clone steps; you must maintain the fork.

---

## Option C — Git subtree

**What it is.** Upstream (or your fork) history is merged into this repo via `git subtree`. One repo, but periodic `git subtree pull` / `push` workflows.

**When to use.** You want something between A and B and are comfortable with subtree commands.

**Pros.** Single clone like A. **Cons.** History can be heavier; subtree commands are easy to get wrong without practice.

---

## What this repo assumes today

Until you explicitly migrate to B or C, treat **`AgileRL/` as Option A (vendored)**. The nested **`AgileRL/.git`** directory has been removed so this tree is a single Git repository. Upstream revision and optional pruning notes are recorded in **`AgileRL/VENDOR_NOTES.md`**.
