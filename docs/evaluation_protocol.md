# Evaluation Protocol

**Functional tests**: accuracy on holdout set; input validation. 
**Request shape validation**: accepts 64-length flattened or nested 8×8; single-sample shorthand accepted.
**Golden inputs**: dataset-backed examples (`artifacts/example_payloads.json`) are exposed in Swagger Examples and used for smoke tests. 
**Stress tests**: latency under load (locust/hey), noise perturbations.  
**Robustness**: simple noise perturbation; optional adversarial attacks with IBM ART.  
**Monitoring**: latency histogram, request counts via Prometheus; error rates from logs.  
**Drift checks**: (illustrative) compare summary stats of incoming features vs training data; raise alerts if thresholds exceeded.  
**Incident response**: capture inputs/outputs for failed requests; rollback by pinning previous image tag; document post-incident analysis.
