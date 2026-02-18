"""
Unit Tests for RAG Pipeline API (Task 2.2)
Tests the vector search and context retrieval functionality.
"""

import os
import sys

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import Base
from init_db import COEFFICIENTS_DATA, STREAMS_DATA, SUBJECTS_DATA
from main import app, get_db, rag_pipeline
from models import Coefficient, Stream, Subject

# Create test database
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    """Override database dependency for testing."""
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


@pytest.fixture(scope="module")
def setup_database():
    """Set up test database with seed data."""
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()

    # Seed streams
    for stream_data in STREAMS_DATA:
        existing = db.query(Stream).filter(Stream.code == stream_data["code"]).first()
        if not existing:
            stream = Stream(**stream_data)
            db.add(stream)
    db.commit()

    # Seed subjects
    for subject_data in SUBJECTS_DATA:
        existing = (
            db.query(Subject).filter(Subject.code == subject_data["code"]).first()
        )
        if not existing:
            subject = Subject(**subject_data)
            db.add(subject)
    db.commit()

    # Seed coefficients
    streams = {s.code: s.id for s in db.query(Stream).all()}
    subjects = {s.code: s.id for s in db.query(Subject).all()}

    for coef_data in COEFFICIENTS_DATA:
        stream_id = streams.get(coef_data["stream_code"])
        subject_id = subjects.get(coef_data["subject_code"])

        if stream_id and subject_id:
            existing = (
                db.query(Coefficient)
                .filter(
                    Coefficient.stream_id == stream_id,
                    Coefficient.subject_id == subject_id,
                )
                .first()
            )

            if not existing:
                coef = Coefficient(
                    stream_id=stream_id,
                    subject_id=subject_id,
                    coefficient=coef_data["coefficient"],
                    is_specialty=coef_data.get("is_specialty", False),
                    specialty_option=coef_data.get("specialty_option"),
                )
                db.add(coef)

    db.commit()
    db.close()

    # Add some test data to RAG pipeline
    _add_test_rag_data()

    yield
    Base.metadata.drop_all(bind=engine)


def _add_test_rag_data():
    """Add test data to RAG pipeline."""
    # Add a sample lesson
    rag_pipeline.index_lesson(
        content="Les nombres complexes sont de la forme z = a + ib. Le module est |z| = sqrt(a² + b²).",
        source="test_lesson.md",
        stream_code="MATH",
        subject_code="MATH",
        topic="complex_numbers",
    )

    # Add a sample exercise
    rag_pipeline.index_exam(
        content="Exercice 1: Calculer le module de z = 3 + 4i. Solution: |z| = 5.",
        source="test_exam.pdf",
        stream_code="MATH",
        subject_code="MATH",
        year=2023,
    )


class TestRAGStatsEndpoint:
    """Test RAG statistics endpoint."""

    def test_get_rag_stats(self, setup_database):
        """Test getting RAG pipeline statistics."""
        response = client.get("/rag-stats")
        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert data["status"] == "active"
        assert "embedding_model" in data
        assert "total_chunks" in data
        assert "mock_mode" in data
        assert data["mock_mode"] is True  # We're using mock embeddings in tests


class TestSearchContextEndpoint:
    """Test vector search endpoint."""

    def test_search_basic(self, setup_database):
        """Test basic search functionality."""
        response = client.post(
            "/search-context", json={"query": "module complex number", "top_k": 5}
        )

        assert response.status_code == 200
        data = response.json()

        assert "query" in data
        assert "results" in data
        assert "total_results" in data
        assert data["query"] == "module complex number"

    def test_search_with_filters(self, setup_database):
        """Test search with stream and subject filters."""
        response = client.post(
            "/search-context",
            json={
                "query": "complex numbers",
                "top_k": 5,
                "stream_code": "MATH",
                "subject_code": "MATH",
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert "filters_applied" in data
        assert data["filters_applied"]["stream_code"] == "MATH"
        assert data["filters_applied"]["subject_code"] == "MATH"

    def test_search_with_doc_type_filter(self, setup_database):
        """Test search with document type filter."""
        response = client.post(
            "/search-context",
            json={"query": "module", "top_k": 5, "doc_type": "lesson"},
        )

        assert response.status_code == 200
        data = response.json()

        # If we have results, they should be lessons
        for result in data["results"]:
            assert result["doc_type"] == "lesson"

    def test_search_min_score_filter(self, setup_database):
        """Test search with minimum score filter."""
        response = client.post(
            "/search-context",
            json={
                "query": "some random query",
                "top_k": 10,
                "min_score": 0.9,  # High threshold
            },
        )

        assert response.status_code == 200
        data = response.json()

        # All results should have score >= 0.9
        for result in data["results"]:
            assert result["score"] >= 0.9

    def test_search_response_structure(self, setup_database):
        """Test that search response has correct structure."""
        response = client.post("/search-context", json={"query": "complex", "top_k": 3})

        assert response.status_code == 200
        data = response.json()

        if data["total_results"] > 0:
            result = data["results"][0]
            assert "chunk_id" in result
            assert "content" in result
            assert "source" in result
            assert "doc_type" in result
            assert "stream_code" in result
            assert "subject_code" in result
            assert "score" in result
            assert "rank" in result
            assert isinstance(result["rank"], int)
            assert isinstance(result["score"], float)

    def test_search_invalid_top_k(self, setup_database):
        """Test search with invalid top_k value."""
        response = client.post(
            "/search-context",
            json={"query": "test", "top_k": 0},  # Invalid: must be >= 1
        )

        assert response.status_code == 422  # Validation error

    def test_search_missing_query(self, setup_database):
        """Test search without required query field."""
        response = client.post("/search-context", json={"top_k": 5})

        assert response.status_code == 422  # Validation error


class TestGetContextForLLMEndpoint:
    """Test context retrieval for LLM endpoint."""

    def test_get_context_basic(self, setup_database):
        """Test basic context retrieval."""
        response = client.post(
            "/get-context-for-llm",
            json={"query": "complex numbers", "max_tokens": 1000},
        )

        assert response.status_code == 200
        data = response.json()

        assert "context" in data
        assert "chunks_used" in data
        assert "character_count" in data
        assert "estimated_tokens" in data
        assert "query" in data
        assert data["query"] == "complex numbers"

    def test_get_context_with_filters(self, setup_database):
        """Test context retrieval with stream/subject filters."""
        response = client.post(
            "/get-context-for-llm",
            json={
                "query": "module",
                "max_tokens": 500,
                "stream_code": "MATH",
                "subject_code": "MATH",
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert "context" in data
        # Context should be related to MATH stream
        assert data["query"] == "module"

    def test_get_context_token_limit(self, setup_database):
        """Test that context respects token limit."""
        max_tokens = 200
        response = client.post(
            "/get-context-for-llm",
            json={"query": "complex numbers", "max_tokens": max_tokens},
        )

        assert response.status_code == 200
        data = response.json()

        # Estimated tokens should be close to or less than max_tokens
        assert data["estimated_tokens"] <= max_tokens

    def test_get_context_invalid_max_tokens(self, setup_database):
        """Test context retrieval with invalid max_tokens."""
        response = client.post(
            "/get-context-for-llm",
            json={"query": "test", "max_tokens": 50},  # Below minimum of 100
        )

        assert response.status_code == 422  # Validation error


class TestRAGIntegration:
    """Integration tests for RAG pipeline."""

    def test_search_then_context_flow(self, setup_database):
        """Test that search and context endpoints work together."""
        query = "module complexe"

        # First, search
        search_response = client.post(
            "/search-context", json={"query": query, "top_k": 5}
        )

        assert search_response.status_code == 200
        search_data = search_response.json()

        # Then, get context
        context_response = client.post(
            "/get-context-for-llm", json={"query": query, "max_tokens": 1500}
        )

        assert context_response.status_code == 200
        context_data = context_response.json()

        # The context should contain information from search results
        assert context_data["query"] == query
        assert isinstance(context_data["context"], str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
