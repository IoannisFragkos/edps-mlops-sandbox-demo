# Risk Register (Excerpt)

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Data distribution shift | Medium | Medium | Monitor feature stats; retrain trigger |
| Adversarial perturbations | Low | Medium | Add input sanitisation; adversarial training (optional); monitoring |
| Resource exhaustion / DoS | Low | High | Autoscaling in cloud; rate limiting; health checks |
| Secrets leakage | Low | High | Use env vars/secret managers; never commit secrets; least privilege |
