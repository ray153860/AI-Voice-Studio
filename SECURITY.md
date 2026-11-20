# Security policy

Report vulnerabilities privately to the repository owner through GitHub Security Advisories. Do not include private audio, credentials, or model files in reports unless essential and authorized.

Every relevant push and pull request runs Bandit against `src/` and pip-audit against the resolved Python environment. JSON reports are uploaded as the `security-reports` workflow artifact, and findings fail the Security workflow. Automated results require human triage; do not publish private recordings, models, paths, or credentials in issue reports.
