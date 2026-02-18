from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class SubjectBase(BaseModel):
    code: str
    name: str
    name_ar: Optional[str] = None
    category: str


class SubjectResponse(SubjectBase):
    id: int

    class Config:
        from_attributes = True


class StreamBase(BaseModel):
    code: str
    name: str
    name_ar: Optional[str] = None
    has_options: bool = False


class StreamResponse(StreamBase):
    id: int

    class Config:
        from_attributes = True


class CoefficientResponse(BaseModel):
    id: int
    stream_id: int
    subject_id: int
    coefficient: int
    is_specialty: bool
    specialty_option: Optional[str] = None
    subject: SubjectResponse

    class Config:
        from_attributes = True


class StreamDetailResponse(StreamResponse):
    coefficients: List[CoefficientResponse]

    class Config:
        from_attributes = True


class UserBase(BaseModel):
    email: str
    full_name: str
    stream_id: Optional[int] = None
    specialty_option: Optional[str] = None


class UserCreate(UserBase):
    password: str


class UserResponse(UserBase):
    id: int
    is_active: bool
    is_admin: bool
    created_at: datetime
    stream: Optional[StreamResponse] = None

    class Config:
        from_attributes = True


class BacAverageRequest(BaseModel):
    stream_id: int
    marks: dict  # {subject_code: mark}
    specialty_option: Optional[str] = None


class BacAverageResponse(BaseModel):
    average: float
    total_points: float
    total_coefficients: int
    mention: str
    passed: bool
    subject_results: List[dict]
