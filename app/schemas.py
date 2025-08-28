from pydantic import BaseModel, field_validator
from typing import List

class PredictRequest(BaseModel):
    """Accept flattened 64-length vectors or 8x8 arrays; allow single-sample or list-of-samples."""
    samples: List[List[float]]

    @field_validator("samples", mode="before")
    @classmethod
    def normalize_samples(cls, v):
        if v is None:
            raise ValueError("samples must not be empty")

        def is_flat64(x):
            return isinstance(x, list) and len(x) == 64 and all(isinstance(t, (int, float)) for t in x)

        def is_8x8(x):
            return (
                isinstance(x, list)
                and len(x) == 8
                and all(isinstance(row, list) and len(row) == 8 and all(isinstance(t,(int,float)) for t in row) for row in x)
            )

        # Case A: shorthand single-sample provided directly (64 floats)
        if isinstance(v, list) and v and all(isinstance(t, (int, float)) for t in v):
            if len(v) != 64:
                raise ValueError("Single-sample shorthand must be 64 floats.")
            return [ [float(t) for t in v] ]

        # Case B: shorthand single-sample provided as a single 8x8 list
        if is_8x8(v):
            flat = [float(t) for row in v for t in row]
            return [ flat ]

        # Case C: proper list-of-samples
        if isinstance(v, list):
            normalized = []
            for i, sample in enumerate(v):
                if is_flat64(sample):
                    normalized.append([float(t) for t in sample])
                elif is_8x8(sample):
                    normalized.append([float(t) for row in sample for t in row])
                else:
                    raise ValueError(f"sample {i} must be either a 64-length list or an 8x8 nested list of numbers")
            return normalized

        raise ValueError("Invalid format for 'samples'")

class PredictResponse(BaseModel):
    predictions: List[int]

class ExplainResponse(BaseModel):
    importances: List[float]
