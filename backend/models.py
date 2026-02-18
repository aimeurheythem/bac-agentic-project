from datetime import datetime

from database import Base
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Table,
)
from sqlalchemy.orm import relationship


class Stream(Base):
    __tablename__ = "streams"

    id = Column(Integer, primary_key=True, index=True)
    code = Column(String, unique=True, index=True)
    name = Column(String, index=True)
    name_ar = Column(String, nullable=True)
    description = Column(String, nullable=True)
    has_options = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    coefficients = relationship("Coefficient", back_populates="stream")
    users = relationship("User", back_populates="stream")


class Subject(Base):
    __tablename__ = "subjects"

    id = Column(Integer, primary_key=True, index=True)
    code = Column(String, unique=True, index=True)
    name = Column(String, index=True)
    name_ar = Column(String, nullable=True)
    category = Column(String)  # scientific, literary, technical, language
    created_at = Column(DateTime, default=datetime.utcnow)

    coefficients = relationship("Coefficient", back_populates="subject")


class Coefficient(Base):
    __tablename__ = "coefficients"

    id = Column(Integer, primary_key=True, index=True)
    stream_id = Column(Integer, ForeignKey("streams.id"))
    subject_id = Column(Integer, ForeignKey("subjects.id"))
    coefficient = Column(Integer)
    is_specialty = Column(Boolean, default=False)
    specialty_option = Column(String, nullable=True)  # For Technique Math options

    stream = relationship("Stream", back_populates="coefficients")
    subject = relationship("Subject", back_populates="coefficients")


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    full_name = Column(String)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    stream_id = Column(Integer, ForeignKey("streams.id"), nullable=True)
    specialty_option = Column(String, nullable=True)  # For Technique Math
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    stream = relationship("Stream", back_populates="users")
