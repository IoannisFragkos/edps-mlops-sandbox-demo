from pydantic import BaseModel, ConfigDict, field_validator
from typing import List

class PredictRequest(BaseModel):
    samples: List[List[float]]

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"summary": "Flattened 64-length vectors", "value": {"samples": [[0.0] * 64]}},
                {"summary": "Nested 8x8 arrays (auto-flattened)", "value": {"samples": [[[0.0 for _ in range(8)] for _ in range(8)]]}},
            ]
        }
    )

    @field_validator("samples")
    @classmethod
    def validate_and_normalize(cls, v):
        if not v:
            raise ValueError("samples must not be empty")
        normalized = []
        for i, sample in enumerate(v):
            if isinstance(sample, list) and len(sample) == 64 and all(isinstance(x, (int, float)) for x in sample):
                normalized.append([float(x) for x in sample]); continue
            if isinstance(sample, list) and len(sample) == 8 and all(isinstance(row, list) and len(row) == 8 for row in sample):
                flat = [float(x) for row in sample for x in row]; normalized.append(flat); continue
            raise ValueError(f"sample {i} must be either length-64 list or 8x8 nested list of floats")
        return normalized

class PredictResponse(BaseModel):
    predictions: List[int]

class ExplainResponse(BaseModel):
    importances: List[float]
