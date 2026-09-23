# Security policy

## Supported code

The current `master` branch is the only supported version. Carla_RL is a local
reinforcement-learning research workspace for highway-env simulation, policy
training, and optional local model-serving experiments. It does not operate a
hosted service.

## Report a vulnerability

Do not open a public issue for an unpatched vulnerability.

Email the repository owner at [tnt850910@aol.com](mailto:tnt850910@aol.com).
When the repository Security tab offers it, you may also use GitHub's private
vulnerability reporting.

Include the affected commit, the vulnerable path (`model-sim/`, `model-serving/`,
or root), the impact, and the smallest reproduction that does not expose
sensitive data. You can expect an acknowledgment within seven days. A fix
schedule depends on the severity and the affected component.

## Keep reports and evidence safe

- Do not send or commit API keys, provider tokens, training checkpoints with
  sensitive metadata, authentication seeds, or personal data.
- Do not attach unredacted agent transcripts, Docker or Kubernetes secrets, or
  host-specific deployment evidence.
- Use synthetic observations, credentials, and local fixtures when reproducing
  training or inference defects.
- Treat captured third-party simulator output under its original license and
  terms.

## Repository boundary

Runtime secrets and provider credentials belong outside the repository. Never
commit a secret value, decrypted configuration, private backup, or unredacted
run export.

`model-sim` training scripts and `model-serving` are local operator tools. They
help you train or serve policies on your own machine. They do not certify
third-party simulators, model providers, or generated artifacts as secure.

The validation gate uses disposable fixtures, synthetic credentials, and local
pytest suites. Local checks do not certify a harness, simulator, serving stack,
or generated change as secure.
