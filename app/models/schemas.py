from typing import Literal

from pydantic import BaseModel


class CNNPredictResult(BaseModel):
    """CNN 品种识别结果"""
    breed_en: str
    breed_cn: str
    confidence: float
    top3: list[dict]
    status: Literal["success", "low_confidence", "failed"]
