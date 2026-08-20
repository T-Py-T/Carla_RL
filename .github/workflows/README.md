# GitHub Actions policy

This dormant repository has no GitHub-hosted workflows. Its previous checks were not reliable merge gates and consumed runner time after pushes, schedules, and manual dispatches.

If active development resumes, validate locally with the repository's pre-commit hooks and tests first. Add a hosted workflow only when it is a passing `pull_request` merge gate, and rehearse it locally with `act pull_request` before opening the PR.
