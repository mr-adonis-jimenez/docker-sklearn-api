Security Policy
Supported Versions

This project follows a container-first release model. Only the most recent version built from the main branch is considered supported.

Version	Supported
main (latest)	✅ Yes
Older commits / images	❌ No

Users are encouraged to rebuild images regularly and keep dependencies up to date via automated tooling (e.g., Dependabot).

Reporting a Vulnerability

If you discover a security vulnerability, do not open a public issue.

Instead, report it responsibly using one of the following methods:

GitHub Security Advisories (preferred):
Use the “Report a vulnerability” button in the repository’s Security tab.

Private contact (if configured):
Provide a clear description, reproduction steps, and potential impact.

Please include:

Affected component (API, Docker image, dependency, CI, etc.)

Steps to reproduce (if applicable)

Any relevant logs, stack traces, or proof-of-concept details

Response Expectations

Initial acknowledgment: within a reasonable timeframe

Severity assessment: based on exploitability and impact

Fix strategy: patch, dependency update, or mitigation

Disclosure: coordinated and responsible when applicable

This is an open-source project; response times may vary, but all credible reports are taken seriously.

Security Scope
In Scope

Dependency vulnerabilities (Python, Docker base images, CI tooling)

API exposure issues (input handling, serialization, deserialization)

Container configuration risks (privileges, base image flaws)

CI/CD pipeline risks related to this repository

Out of Scope

Vulnerabilities in user-deployed infrastructure

Misconfiguration of downstream systems

Denial-of-service via unrealistic traffic volumes

Issues requiring physical access to host machines

Security Practices

This repository applies the following baseline security measures:

Dependency monitoring via Dependabot

Minimal Docker images to reduce attack surface

Stateless API design to limit persistence risk

No embedded secrets in source control

PR-based changes for traceability and review

Security is treated as an engineering concern, not an afterthought.

Disclaimer

This project is provided as-is, without warranty of any kind. While reasonable steps are taken to reduce risk, users are responsible for securing their own deployment environments and data.
