# Highway RL - Release Changelog

This file tracks all feature releases and their associated artifacts.

## Release History

### F004 - [Bump torch from 2.1.1 to 2.8.0 in /model-serving](https://github.com/T-Py-T/Carla_RL/pull/5) - 2025-09-27 18:48:35 UTC

**Release ID:** `F004-dependabot-pip-model-serving-torch-2-8-0`  
**Branch:** `dependabot/pip/model-serving/torch-2.8.0`  
**Author:** dependabot[bot]  
**Artifact:** `release-artifacts/F004-dependabot-pip-model-serving-torch-2-8-0/`

Bumps [torch](https://github.com/pytorch/pytorch) from 2.1.1 to 2.8.0.
<details>
<summary>Release notes</summary>
<p><em>Sourced from <a href="https://github.com/pytorch/pytorch/releases">torch's releases</a>.</em></p>
<blockquote>
<h1>PyTorch 2.8.0 Release Notes</h1>
<ul>
<li><a href="https://github.com/pytorch/pytorch/blob/HEAD/#highlights">Highlights</a></li>
<li><a href="https://github.com/pytorch/pytorch/blob/HEAD/#backwards-incompatible-changes">Backwards Incompatible Changes</a></li>
<li><a href="https://github.com/pytorch/pytorch/blob/HEAD/#deprecations">Deprecations</a></li>
<li><a href="https://github.com/pytorch/pytorch/blob/HEAD/#new-features">New Features</a></li>
<li><a href="https://github.com/pytorch/pytorch/blob/HEAD/#improvements">Improvements</a></li>
<li><a href="https://github.com/pytorch/pytorch/blob/HEAD/#bug-fixes">Bug fixes</a></li>
<li><a href="https://github.com/pytorch/pytorch/blob/HEAD/#performance">Performance</a></li>
<li><a href="https://github.com/pytorch/pytorch/blob/HEAD/#documentation">Documentation</a></li>
<li><a href="https://github.com/pytorch/pytorch/blob/HEAD/#developers">Developers</a></li>
</ul>
<h1>Highlights</h1>
<!-- raw HTML omitted -->
</blockquote>
<p>... (truncated)</p>
</details>
<details>
<summary>Commits</summary>
<ul>
<li><a href="https://github.com/pytorch/pytorch/commit/ba56102387ef21a3b04b357e5b183d48f0afefc7"><code>ba56102</code></a> Cherrypick: Add the RunLLM widget to the website (<a href="https://redirect.github.com/pytorch/pytorch/issues/159592">#159592</a>)</li>
<li><a href="https://github.com/pytorch/pytorch/commit/c525a02c89217181b5731d8043c7309a84e84066"><code>c525a02</code></a> [dynamo, docs] cherry pick torch.compile programming model docs into 2.8 (<a href="https://redirect.github.com/pytorch/pytorch/issues/15">#15</a>...</li>
<li><a href="https://github.com/pytorch/pytorch/commit/a1cb3cc05d46d198467bebbb6e8fba50a325d4e7"><code>a1cb3cc</code></a> [Release Only] Remove nvshmem from list of preload libraries (<a href="https://redirect.github.com/pytorch/pytorch/issues/158925">#158925</a>)</li>
<li><a href="https://github.com/pytorch/pytorch/commit/c76b2356bc31654de2af0c98cce1bef291f06f89"><code>c76b235</code></a> Move out super large one off foreach_copy test (<a href="https://redirect.github.com/pytorch/pytorch/issues/158880">#158880</a>)</li>
<li><a href="https://github.com/pytorch/pytorch/commit/20a0e225a01d4ebbffd44a6a59acff628359c772"><code>20a0e22</code></a> Revert &quot;[Dynamo] Allow inlining into AO quantization modules (<a href="https://redirect.github.com/pytorch/pytorch/issues/152934">#152934</a>)&quot; (<a href="https://redirect.github.com/pytorch/pytorch/issues/158">#158</a>...</li>
<li><a href="https://github.com/pytorch/pytorch/commit/9167ac8c75481e2beb3746aa37b7f48a213c631e"><code>9167ac8</code></a> [MPS] Switch Cholesky  decomp to column wise (<a href="https://redirect.github.com/pytorch/pytorch/issues/158237">#158237</a>)</li>
<li><a href="https://github.com/pytorch/pytorch/commit/5534685c62399db8d1e51b47e2dcbc17deaab230"><code>5534685</code></a> [MPS] Reimplement <code>tri[ul]</code> as Metal shaders (<a href="https://redirect.github.com/pytorch/pytorch/issues/158867">#158867</a>)</li>
<li><a href="https://github.com/pytorch/pytorch/commit/d19e08d74b2a27e661bf57a9015014b757e8ea31"><code>d19e08d</code></a> Cherry pick PR 158746 (<a href="https://redirect.github.com/pytorch/pytorch/issues/158801">#158801</a>)</li>
<li><a href="https://github.com/pytorch/pytorch/commit/a6c044ab9aa14f0864c6a572f7c023432511c5ea"><code>a6c044a</code></a> [cherry-pick] Unify torch.tensor and torch.ops.aten.scalar_tensor behavior (#...</li>
<li><a href="https://github.com/pytorch/pytorch/commit/620ebd0646252bbb22524f5c252ec7e9ab977bee"><code>620ebd0</code></a> [Dynamo] Use proper sources for constructing dataclass defaults (<a href="https://redirect.github.com/pytorch/pytorch/issues/158689">#158689</a>)</li>
<li>Additional commits viewable in <a href="https://github.com/pytorch/pytorch/compare/v2.1.1...v2.8.0">compare view</a></li>
</ul>
</details>
<br />


[![Dependabot compatibility score](https://dependabot-badges.githubapp.com/badges/compatibility_score?dependency-name=torch&package-manager=pip&previous-version=2.1.1&new-version=2.8.0)](https://docs.github.com/en/github/managing-security-vulnerabilities/about-dependabot-security-updates#about-compatibility-scores)

Dependabot will resolve any conflicts with this PR as long as you don't alter it yourself. You can also trigger a rebase manually by commenting .

[//]: # (dependabot-automerge-start)
[//]: # (dependabot-automerge-end)

---

<details>
<summary>Dependabot commands and options</summary>
<br />

You can trigger Dependabot actions by commenting on this PR:
-  will rebase this PR
-  will recreate this PR, overwriting any edits that have been made to it
-  will merge this PR after your CI passes on it
-  will squash and merge this PR after your CI passes on it
-  will cancel a previously requested merge and block automerging
-  will reopen this PR if it is closed
-  will close this PR and stop Dependabot recreating it. You can achieve the same result by closing it manually
-  will show all of the ignore conditions of the specified dependency
-  will close this PR and stop Dependabot creating any more for this major version (unless you reopen the PR or upgrade to it yourself)
-  will close this PR and stop Dependabot creating any more for this minor version (unless you reopen the PR or upgrade to it yourself)
-  will close this PR and stop Dependabot creating any more for this dependency (unless you reopen the PR or upgrade to it yourself)
You can disable automated security fix PRs for this repo from the [Security Alerts page](https://github.com/T-Py-T/Carla_RL/network/alerts).

</details>

**Key Changes:**



---

### F003 - [Staging](https://github.com/T-Py-T/Carla_RL/pull/4) - 2025-09-27 18:16:17 UTC

**Release ID:** `F003-staging`  
**Branch:** `staging`  
**Author:** T-Py-T  
**Artifact:** `release-artifacts/F003-staging/`



**Key Changes:**
- .cursor/rules/create-prd.mdc
- .cursor/rules/generate-tasks.mdc
- .cursor/rules/process-task-list.mdc
- .cursor/rules/release-workflow.mdc
- .github/BRANCH_PROTECTION.md
- .github/workflows/README.md
- .github/workflows/feature-release-artifact.yml
- .github/workflows/merge-validation.yml
- .github/workflows/ml-pipeline-trigger.yml
- .github/workflows/pr-checks.yml
- ... and 91 more files

---

### F002 - [Dev](https://github.com/T-Py-T/Carla_RL/pull/3) - 2025-09-27 18:15:20 UTC

**Release ID:** `F002-dev`  
**Branch:** `dev`  
**Author:** T-Py-T  
**Artifact:** `release-artifacts/F002-dev/`



**Key Changes:**
- .cursor/rules/create-prd.mdc
- .cursor/rules/generate-tasks.mdc
- .cursor/rules/process-task-list.mdc
- .cursor/rules/release-workflow.mdc
- .github/BRANCH_PROTECTION.md
- .github/workflows/README.md
- .github/workflows/feature-release-artifact.yml
- .github/workflows/merge-validation.yml
- .github/workflows/ml-pipeline-trigger.yml
- .github/workflows/pr-checks.yml
- ... and 87 more files

---

### F001 - [Add model serving](https://github.com/T-Py-T/Carla_RL/pull/2) - 2025-09-27 18:12:52 UTC

**Release ID:** `F001-add_model_serving`  
**Branch:** `add_model_serving`  
**Author:** T-Py-T  
**Artifact:** `release-artifacts/F001-add_model_serving/`

<!-- CURSOR_SUMMARY -->
> [!NOTE]
> Rebrands the project to , adds strict Ruff/linting and test tooling, introduces serving PRD/task docs, removes legacy CARLA remnants, and refreshes dependencies/lockfile for Python 3.11+.
> 
> - **Project**:
>   - Rename  → ; update description and  to  in .
> - **Tooling/Config**:
>   - Add comprehensive Ruff config () with linting rules, isort, per-file ignores; dev extras include , , .
> - **Docs/Planning**:
>   - Add serving PRD and task breakdown (, ).
> - **Cleanup**:
>   - Remove legacy CARLA files () and obsolete test compatibility shims under .
> - **Dependencies**:
>   - Overhaul : drop legacy ML stacks, add/upgrade core tooling (, , , ), and align with Python 3.11+.
> 
> <sup>Written by [Cursor Bugbot](https://cursor.com/dashboard?tab=bugbot) for commit 7e60c865ef38a413313cd0ca77987539699958ba. This will update automatically on new commits. Configure [here](https://cursor.com/dashboard?tab=bugbot).</sup>
<!-- /CURSOR_SUMMARY -->

**Key Changes:**
- .cursor/rules/create-prd.mdc
- .cursor/rules/generate-tasks.mdc
- .cursor/rules/process-task-list.mdc
- .cursor/rules/release-workflow.mdc
- .github/BRANCH_PROTECTION.md
- .github/workflows/README.md
- .github/workflows/feature-release-artifact.yml
- .github/workflows/merge-validation.yml
- .github/workflows/ml-pipeline-trigger.yml
- .github/workflows/pr-checks.yml
- ... and 82 more files

---

