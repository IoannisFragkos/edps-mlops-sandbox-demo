# Model Card — Digits Classifier (Demo)

**Intended use**: Educational/demo for MLOps, sandboxes, and audits (not for production decisions).  
**Deployed service**: https://edps-mlops-demo-270205068451.europe-west1.run.app (HTTPS :443)  
**API health**: `/health` — expected `{"status":"ok"}`  
**Monitoring**: `/metrics` (Prometheus exposition)  
**Task**: Multiclass classification on sklearn digits dataset.  
**Data**: Public digits dataset; 8×8 pixel grayscale images.  
**Performance**: See `artifacts/metadata.json` after training for test accuracy.  
**Ethical, security, and robustness considerations**: Limited scope; demonstrates documentation, monitoring, and robustness testing workflow, not a safety-critical use case.
