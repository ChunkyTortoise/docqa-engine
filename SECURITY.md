# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.4.x   | Yes       |
| 0.3.x   | Yes       |
| < 0.3   | No        |

## Reporting a Vulnerability

If you discover a security vulnerability in docqa-engine, please report it responsibly.

**Do not open a public GitHub issue for security vulnerabilities.**

Instead, send an email to **chunkytortoise@proton.me** with:

1. A description of the vulnerability
2. Steps to reproduce
3. Potential impact assessment
4. Any suggested fixes (optional)

### What to Expect

- **Acknowledgment**: Within 48 hours of your report
- **Assessment**: Within 7 days, we will confirm whether the issue is accepted
- **Fix timeline**: Critical vulnerabilities will be patched within 14 days; lower-severity issues within 30 days
- **Disclosure**: We will coordinate public disclosure with you after a fix is released

### Scope

The following are in scope:

- Authentication bypass in the REST API
- Injection vulnerabilities (path traversal in file ingestion, etc.)
- Denial of service via malformed input
- Information disclosure (API keys, internal paths, etc.)

The following are out of scope:

- Issues in third-party dependencies (report upstream)
- Issues requiring physical access to the server
- Social engineering attacks

## Security Best Practices

When deploying docqa-engine:

- Use environment variables for API keys (never hardcode)
- Deploy behind a reverse proxy (nginx, Caddy) with TLS
- Set appropriate rate limits for your use case
- Regularly update dependencies (`pip install --upgrade`)
