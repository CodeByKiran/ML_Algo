from typing import List
from pydantic import BaseModel

class DBSCANRequest(BaseModel):
    data: List[List[float]]
    eps: float = 0.5
    min_samples: int = 5
