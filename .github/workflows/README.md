# GitHub Actions Workflows

This directory contains automated workflows for the Highway RL project.

## Workflows

### Feature Release Artifact (`feature-release-artifact.yml`)

**Trigger:** When a pull request is merged to `main`/`master`

**Purpose:** Automatically creates release artifacts from feature development

**What it does:**
1. **Extracts feature info** from the merged PR (branch name, title, author, etc.)
2. **Creates artifact folder** in `release-artifacts/feature-{branch}-{pr-number}/`
3. **Copies tasks folder** contents (PRDs, task lists, documentation)
4. **Generates feature summary** with PR details, commits, and changed files
5. **Updates release changelog** (`RELEASE_CHANGELOG.md`) with new entry
6. **Creates GitHub release** with feature artifacts attached
7. **Commits artifacts** back to the repository

**Artifacts Created:**
```
release-artifacts/F001-feature-name/
├── prd-*.md                 # Product Requirements Document
├── tasks-*.md               # Task breakdown and tracking
├── FEATURE_SUMMARY.md       # Auto-generated feature summary
└── metadata.json            # Machine-readable metadata
```

**Benefits:**
- **Preserves feature planning** - PRDs and task lists are archived
- **Maintains project history** - Complete feature development record
- **Enables retrospectives** - Easy access to what was planned vs delivered
- **Fully automated** - No manual work required
- **GitHub releases** - Professional release management

## Usage

1. **Develop feature** with tasks in `tasks/` folder
2. **Create PR** to main branch
3. **Merge PR** - Action automatically triggers
4. **View artifacts** in `release-artifacts/` or GitHub Releases

## Configuration

The workflow uses standard GitHub tokens and requires no additional setup. It will:
- Only run on merged PRs to main/master
- Skip if no `tasks/` folder exists
- Auto-generate sequential feature numbers (F001, F002, etc.)
- Create release tags like `F001-add-model-serving`
- Use clean folder names with sanitized branch names

## Customization

You can customize the workflow by:
- Modifying the changelog format in the workflow file
- Adding additional artifact processing steps
- Changing the release naming convention
- Adding Slack/Teams notifications
