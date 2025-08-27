# Security Notes

- No secrets committed. Use environment variables or a secrets manager for cloud deploys.
- Container runs as a **non-root** user.
- Dependencies are pinned; CI includes a basic Trivy vulnerability scan.
