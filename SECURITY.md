# Security Policy

## Overview

The mlxcel team takes security seriously. We appreciate the security research community's efforts in helping us keep mlxcel and its users safe. This document outlines how we accept vulnerability reports, what to expect from our response process, and what is in and out of scope.

## Supported Versions

We provide security updates for the following versions:

| Version              | Supported          | End of Support              |
| -------------------- | ------------------ | --------------------------- |
| 0.0.x (>=0.0.27)     | :white_check_mark: | Current pre-1.0 line        |
| < 0.0.27             | :x:                | Pre-public release; unsupported |

> **Note**: mlxcel is currently in the pre-1.0 (`0.0.x`) line. Until the project cuts its first minor release, the latest published `0.0.x` stable is the only supported version and security fixes ship as a new patch release. End-of-support dates will be defined when the project reaches `0.1.0`.

We strongly recommend always running the latest stable release to ensure you have the most recent security fixes.

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues, discussions, or pull requests.**

If you discover a security vulnerability in mlxcel, please report it through one of the following channels:

### Preferred: GitHub Security Advisories

1. Navigate to the [Security Advisories](https://github.com/lablup/mlxcel/security/advisories) page
2. Click **Report a vulnerability**
3. Fill out the form with as much detail as possible

This channel keeps the report private until a fix is ready and lets us collaborate with you on the patch.

### Alternative: Email

Send an email to **security@lablup.com** with the following information:

- Subject line: `[SECURITY] mlxcel - <brief description>`
- Detailed description of the vulnerability
- Steps to reproduce
- Affected versions
- Potential impact and attack scenarios
- Proof of concept (if available)
- Your contact information for follow-up

For sensitive disclosures, you may request our PGP key by emailing security@lablup.com with subject `[PGP KEY REQUEST]`.

### What to Include

A good vulnerability report should include:

1. **Type of vulnerability** (e.g., authentication bypass, deserialization, command injection)
2. **Affected component** (e.g., `mlxcel-server` HTTP endpoint, weight loader, downloader)
3. **Affected versions**
4. **Detailed description** of the vulnerability and its potential impact
5. **Step-by-step reproduction** with example requests, commands, or input files
6. **Proof of concept** (code, scripts, or demonstration)
7. **Suggested mitigation** (optional but appreciated)
8. **CVE identifier** (if already assigned)

### What to Expect

- **Initial acknowledgment**: within **48-72 hours** for all severity levels
- **Status updates**: at least weekly during investigation and remediation
- **Validation**: our team will validate the report and determine severity
- **Fix timeline**: see the severity table below
- **Disclosure**: coordinated with you per the policy below
- **Credit**: with your permission, we will publicly credit you in the security advisory and release notes

## Disclosure Policy

We follow a **90-day coordinated disclosure policy**:

1. **Day 0**: vulnerability reported
2. **Day 0-7**: triage, validation, severity assessment
3. **Day 7-60**: develop, test, and review the fix
4. **Day 60-75**: prepare the security advisory and coordinate with the reporter
5. **Day 75-90**: release patched versions and publish the advisory
6. **Day 90**: public disclosure (may be extended by mutual agreement)

### Severity-Based Timelines

| Severity | Target Fix Timeline | Target Disclosure |
| -------- | ------------------- | ----------------- |
| Critical | 24-48 hours         | 7-14 days         |
| High     | 1 week              | 30 days           |
| Medium   | 2-4 weeks           | 60 days           |
| Low      | Next patch release  | 90 days           |

We may request an extension if the fix requires significant architectural changes, coordination with upstream MLX libraries, or additional testing across model architectures.

## Security Update Process

When we ship a security fix we will:

1. Publish a **GitHub Security Advisory** with CVE (if applicable), description, affected/fixed versions, mitigation guidance, and reporter credit
2. Release a patched version following semantic versioning
3. Update `CHANGELOG.md` with the security fix
4. Announce through GitHub Releases

Users can stay informed by:

- Watching the repository with **Security alerts** enabled
- Following the [Security Advisories](https://github.com/lablup/mlxcel/security/advisories) page
- Monitoring the [Releases](https://github.com/lablup/mlxcel/releases) page

## Scope

### In Scope

We are interested in reports about the following classes of vulnerability in mlxcel itself (the `mlxcel` CLI, `mlxcel-server` HTTP server, `mlxcel-core` runtime, and `mlxcel-surgery` operations):

#### Authentication & Authorization
- Authentication bypass on `mlxcel-server` HTTP endpoints
- API-key handling weaknesses (leakage, weak comparison, missing validation)
- Authorization bypass between sessions or tenants on a shared server

#### Code Execution & Deserialization
- Crafted model checkpoints (safetensors, GGUF, weight shards) that trigger memory-corruption, OOB reads/writes, integer overflows, or arbitrary code execution during weight loading
- Crafted YAML inputs (e.g., `--surgery <config.yaml>`) that lead to unsafe deserialization or path traversal
- Crafted prompts or HTTP payloads that lead to native crashes, panics with secret exposure, or memory corruption in the FFI boundary

#### Data Exposure
- Cross-request leakage of KV-cache contents on shared `mlxcel-server` instances
- Sensitive data in logs, error messages, or traces (API keys, prompt contents from other sessions, file paths beyond the server's intended scope)
- Path-traversal in `mlxcel download` or weight loaders

#### Denial of Service
- Resource exhaustion via crafted prompts, weight shapes, or HTTP request patterns that disproportionately consume CPU, GPU, memory, or disk
- Algorithmic-complexity attacks in the sampler, tokenizer, scheduler, or speculative-decoding paths
- Crashes that take down `mlxcel-server` reliably from a single request
- Rate-limit bypass (if rate limiting is configured)

#### Cryptographic Issues
- Missing or weak TLS verification when fetching models via `mlxcel download`
- Insecure handling of checksums or signatures on downloaded artifacts
- Use of weak random sources where cryptographic randomness is required

#### Distributed Mode (Pipeline / Tensor Parallel)
- Unauthenticated peer discovery (mDNS) accepting hostile peers
- Tampering with inter-node tensor traffic
- Replay or injection attacks against the pipeline-parallel control plane

#### Release & Supply Chain
- Tampering with published GitHub release assets, the Homebrew formula, or workflow definitions that would let an attacker substitute the binary
- Workflows that would allow a forked PR or arbitrary contributor to obtain release-signing secrets or self-hosted runner access

### Out of Scope

The following are **not** treated as security vulnerabilities by this policy:

#### Model Behavior
- Model output quality, factual errors, hallucinations, or jailbreaks
- Prompt-injection effects on model output (this is a model-behavior concern, not a runtime vulnerability)
- Bias, fairness, or content-moderation issues

#### Upstream MLX
- Vulnerabilities in the underlying MLX C++ libraries (please report to [ml-explore/mlx](https://github.com/ml-explore/mlx))
- Issues that only reproduce against unmaintained or upstream-deprecated model architectures

#### Dependency Vulnerabilities Without mlxcel Impact
- Known advisories on third-party crates that do not affect any code path mlxcel actually exercises (please open a regular issue or PR with the relevant `[advisories.ignore]` justification for `deny.toml`)

#### Non-Security Bugs
- General bugs without security implications (please file a regular issue)
- Performance regressions without DoS implications
- Feature requests

#### Social Engineering / Physical
- Phishing or social engineering against maintainers or users
- Physical access to user hardware

## Safe Harbor

Lablup Inc. supports the security research community and will not pursue legal action against researchers who:

1. **Act in good faith** to report vulnerabilities responsibly
2. **Avoid privacy violations** by not accessing, modifying, or deleting data beyond what is needed to demonstrate the issue
3. **Avoid service disruption** by not degrading availability for others
4. **Do not publicly disclose** the vulnerability before we have had a reasonable time to respond
5. **Follow this security policy** and coordinate disclosure timelines

We will work with you to understand and validate your report. We will not file legal claims or contact law enforcement about your research if you comply with this policy.

Safe harbor applies to research conducted on:

- Local development environments
- Test environments you have explicit permission to test
- Publicly-accessible instances of `mlxcel-server` that you operate

Do **not** test against production systems you do not own, access other users' data, or conduct testing that could harm availability for others.

## Security Best Practices for Operators

While we work to make mlxcel secure by default, operators running `mlxcel-server` in production should:

### Server Configuration
- **Bind to a trusted interface** — `mlxcel-server` does not assume the network it is exposed on is trusted; put it behind a TLS-terminating reverse proxy for any non-loopback exposure
- **Enable authentication** if you expose the server to multiple clients; do not assume API keys are optional
- **Restrict network access** with firewalls or network policies; assume any unauthenticated path is reachable from the open internet if it is bound to `0.0.0.0`
- **Keep model files under controlled paths** — the server's file-system access is your responsibility

### Model Sourcing
- **Verify checksums** of downloaded model checkpoints (`.sha256` files are published alongside release assets and Hugging Face provides hashes for model files)
- **Treat checkpoints as code** — a malicious safetensors file can attempt to exploit loader bugs; only load models from sources you trust

### Distributed Mode
- **Do not enable mDNS-based discovery on untrusted networks** — use explicit static peer lists in any environment where the local network is not fully under your control

### Updates
- **Subscribe to security advisories** and apply patches promptly
- **Test updates** in staging before production

## Security Features

mlxcel currently ships the following security-relevant practices and features:

- **Hardened release pipeline**: top-level default-deny GitHub Actions permissions, per-job grants, `github.repository` guards against fork `workflow_dispatch`, and `persist-credentials: false` on every checkout to prevent token leakage into `.git/config` on self-hosted runners
- **Repository ruleset on `main`**: blocks force-push and branch deletion (Ruleset `main protection`, no bypass actors)
- **`cargo-deny` audit**: vulnerability, license, and source-provenance checks gate every PR
- **Automated dependency updates**: Dependabot runs weekly across all Cargo crates and GitHub Actions
- **Verified secret-scanning baseline**: `.gitleaksignore` documents known false positives, and GitHub native Secret Scanning is enabled on this public repository
- **Reproducible release artifacts**: every release asset is published with a `.sha256` companion file

## Dependency Security Auditing

mlxcel uses automated tools to catch vulnerable dependencies before they reach a release.

### Automated CI/CD Checks
- **cargo-deny**: runs on every PR and push to `main` (advisories, licenses, bans, sources)
- **Dependabot**: weekly scans across `cargo` (root, `mlxcel-core`, `mlxcel-surgery`) and `github-actions` ecosystems
- **GitHub Security Alerts**: real-time vulnerability notifications via Dependabot

### Local Security Audits

Before submitting a PR, run security checks locally:

```bash
# Install (one-time)
cargo install cargo-deny --locked

# Comprehensive check (advisories + licenses + bans + sources)
cargo deny check

# Or, for vulnerability-only check
cargo install cargo-audit --locked
cargo audit
```

### Configuration Files

- **`deny.toml`**: `cargo-deny` configuration (advisories, license allow-list, sources)
- **`.github/dependabot.yml`**: Dependabot schedule, grouping, and ecosystems
- **`.gitleaksignore`**: justifications for verified-false-positive secret scanner findings

### Accepting an Unfixable Advisory

If an advisory exists for a dependency mlxcel cannot upgrade out of (e.g., transitive, no patched version published, not exploitable in our context), document the exception in `deny.toml`:

```toml
[advisories]
ignore = [
    { id = "RUSTSEC-XXXX-XXXX", reason = "Not exploitable in our usage: <specific reason>" },
]
```

## Security Contacts

- **Security advisories**: [Report a vulnerability](https://github.com/lablup/mlxcel/security/advisories/new)
- **Email**: security@lablup.com
- **Organization**: Lablup Inc.

## Acknowledgments

We will thank security researchers here as we receive and address reports.

---

**Last Updated**: 2026-05-18
**Version**: 1.0
