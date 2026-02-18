"""
Unit Tests for Coefficient Engine API (Task 2.1)
Tests the Bac average calculator with various scenarios and edge cases.
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
from init_db import seed_data
from main import app, calculate_subject_points, get_db, get_mention

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

    # Import and seed data directly to test database
    from init_db import COEFFICIENTS_DATA, STREAMS_DATA, SUBJECTS_DATA
    from models import Coefficient, Stream, Subject

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
    yield
    Base.metadata.drop_all(bind=engine)


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check(self, setup_database):
        """Test health check returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["database"] == "connected"


class TestStreamsEndpoints:
    """Test streams endpoints."""

    def test_get_streams(self, setup_database):
        """Test getting all streams returns 7 official streams."""
        response = client.get("/streams")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 7

        # Check for specific streams
        stream_codes = [s["code"] for s in data]
        assert "MATH" in stream_codes
        assert "SCI_EXP" in stream_codes
        assert "TECH_MATH" in stream_codes
        assert "GESTION" in stream_codes

    def test_get_stream_detail(self, setup_database):
        """Test getting stream details with coefficients."""
        # Get Math stream (id=2 based on seeding order)
        response = client.get("/streams/2")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == "MATH"
        assert data["name"] == "Mathématiques"
        assert "coefficients" in data
        assert len(data["coefficients"]) > 0

    def test_get_stream_not_found(self, setup_database):
        """Test getting non-existent stream returns 404."""
        response = client.get("/streams/999")
        assert response.status_code == 404
        assert "detail" in response.json()

    def test_get_tech_math_specialties(self, setup_database):
        """Test Technique Math stream returns specialty options."""
        # Technique Math should be id=3
        response = client.get("/streams/3")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == "TECH_MATH"
        assert "specialty_options" in data
        assert len(data["specialty_options"]) == 4  # Civil, Mech, Elec, Proc

    def test_get_stream_specialties_endpoint(self, setup_database):
        """Test dedicated specialties endpoint."""
        response = client.get("/streams/3/specialties")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 4
        specialty_codes = [s["code"] for s in data]
        assert "CIVIL" in specialty_codes
        assert "MECA" in specialty_codes
        assert "ELEC" in specialty_codes
        assert "PROC" in specialty_codes


class TestSubjectsEndpoints:
    """Test subjects endpoints."""

    def test_get_all_subjects(self, setup_database):
        """Test getting all subjects."""
        response = client.get("/subjects")
        assert response.status_code == 200
        data = response.json()
        assert len(data) > 0

        # Check for core subjects
        subject_codes = [s["code"] for s in data]
        assert "MATH" in subject_codes
        assert "ARABIC" in subject_codes
        assert "PHYSICS" in subject_codes

    def test_get_subjects_by_category(self, setup_database):
        """Test filtering subjects by category."""
        response = client.get("/subjects?category=scientific")
        assert response.status_code == 200
        data = response.json()
        assert len(data) > 0
        for subject in data:
            assert subject["category"] == "scientific"


class TestCoefficientsEndpoints:
    """Test coefficients endpoints."""

    def test_get_all_coefficients(self, setup_database):
        """Test getting all coefficients."""
        response = client.get("/coefficients")
        assert response.status_code == 200
        data = response.json()
        assert len(data) > 0

    def test_get_coefficients_by_stream(self, setup_database):
        """Test filtering coefficients by stream."""
        # Get coefficients for Math stream (id=2)
        response = client.get("/coefficients?stream_id=2")
        assert response.status_code == 200
        data = response.json()
        assert len(data) > 0

        # All coefficients should belong to Math stream
        for coef in data:
            assert coef["stream_id"] == 2

    def test_get_stream_subjects_with_coefficients(self, setup_database):
        """Test getting subjects with coefficients for a stream."""
        response = client.get("/streams/2/subjects-and-coefficients")
        assert response.status_code == 200
        data = response.json()

        assert "stream" in data
        assert "subjects" in data
        assert "total_coefficients" in data
        assert data["stream"]["code"] == "MATH"
        assert len(data["subjects"]) > 0


class TestCalculateAverageEndpoint:
    """Test the calculate average endpoint - core feature."""

    def test_calculate_average_math_stream(self, setup_database):
        """Test calculating average for Math stream."""
        marks = {
            "MATH": 15.0,
            "PHYSICS": 14.0,
            "SCIENCES": 13.0,
            "ARABIC": 12.0,
            "FRENCH": 11.0,
            "ENGLISH": 13.0,
            "ISLAMIC": 15.0,
            "CIVICS": 14.0,
            "HISTORY_GEO": 12.0,
            "SPORT": 15.0,
        }

        response = client.post(
            "/calculate-average",
            json={"stream_id": 2, "marks": marks, "sport_exempt": False},
        )

        assert response.status_code == 200
        data = response.json()

        assert data["stream_code"] == "MATH"
        assert "average" in data
        assert "mention" in data
        assert "passed" in data
        assert "subject_results" in data
        assert len(data["subject_results"]) == len(marks)

        # Check that average is calculated correctly
        # Math: coef 7, Physics: coef 6, Sciences: coef 4, Arabic: coef 3, etc.
        expected_total_coeff = 7 + 6 + 4 + 3 + 2 + 2 + 2 + 1 + 2 + 1  # = 30
        assert data["total_coefficients"] == expected_total_coeff
        assert data["passed"] == (data["average"] >= 10)

    def test_calculate_average_with_missing_marks(self, setup_database):
        """Test calculating average with some missing marks."""
        marks = {
            "MATH": 15.0,
            "PHYSICS": 14.0,
            # Missing some subjects
        }

        response = client.post(
            "/calculate-average", json={"stream_id": 2, "marks": marks}
        )

        assert response.status_code == 200
        data = response.json()
        assert "warnings" in data
        assert len(data["warnings"]) > 0

    def test_calculate_average_mention_bien(self, setup_database):
        """Test mention calculation for 'Bien' (14-16)."""
        marks = {
            "MATH": 15.0,
            "PHYSICS": 15.0,
            "SCIENCES": 14.0,
            "ARABIC": 14.0,
            "FRENCH": 14.0,
            "ENGLISH": 14.0,
            "ISLAMIC": 14.0,
            "CIVICS": 14.0,
            "HISTORY_GEO": 14.0,
            "SPORT": 14.0,
        }

        response = client.post(
            "/calculate-average", json={"stream_id": 2, "marks": marks}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["mention"] == "Bien"
        assert data["passed"] is True

    def test_calculate_average_mention_tres_bien(self, setup_database):
        """Test mention calculation for 'Très Bien' (16+)."""
        marks = {
            "MATH": 17.0,
            "PHYSICS": 17.0,
            "SCIENCES": 16.0,
            "ARABIC": 16.0,
            "FRENCH": 16.0,
            "ENGLISH": 16.0,
            "ISLAMIC": 16.0,
            "CIVICS": 16.0,
            "HISTORY_GEO": 16.0,
            "SPORT": 16.0,
        }

        response = client.post(
            "/calculate-average", json={"stream_id": 2, "marks": marks}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["mention"] == "Très Bien"
        assert data["passed"] is True

    def test_calculate_average_fail(self, setup_database):
        """Test calculation when student fails (average < 10)."""
        marks = {
            "MATH": 5.0,
            "PHYSICS": 6.0,
            "SCIENCES": 7.0,
            "ARABIC": 8.0,
            "FRENCH": 8.0,
            "ENGLISH": 8.0,
            "ISLAMIC": 9.0,
            "CIVICS": 9.0,
            "HISTORY_GEO": 8.0,
            "SPORT": 10.0,
        }

        response = client.post(
            "/calculate-average", json={"stream_id": 2, "marks": marks}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["mention"] == "Non Admis"
        assert data["passed"] is False

    def test_calculate_average_invalid_stream(self, setup_database):
        """Test calculating average for non-existent stream."""
        marks = {"MATH": 15.0}

        response = client.post(
            "/calculate-average", json={"stream_id": 999, "marks": marks}
        )

        assert response.status_code == 404

    def test_calculate_average_invalid_marks(self, setup_database):
        """Test validation rejects marks outside 0-20 range."""
        marks = {"MATH": 25.0, "PHYSICS": -5.0}  # Invalid: > 20  # Invalid: < 0

        response = client.post(
            "/calculate-average", json={"stream_id": 2, "marks": marks}
        )

        assert response.status_code == 422  # Validation error

    def test_calculate_average_sport_exemption(self, setup_database):
        """Test calculation with sport exemption."""
        marks = {
            "MATH": 15.0,
            "PHYSICS": 14.0,
            "SCIENCES": 13.0,
            "ARABIC": 12.0,
            "FRENCH": 11.0,
            "ENGLISH": 13.0,
            "ISLAMIC": 15.0,
            "CIVICS": 14.0,
            "HISTORY_GEO": 12.0,
            # Sport is exempted
        }

        response = client.post(
            "/calculate-average",
            json={"stream_id": 2, "marks": marks, "sport_exempt": True},
        )

        assert response.status_code == 200
        data = response.json()
        # Sport coefficient should not be included
        assert data["total_coefficients"] < 30  # Less than with Sport


class TestValidateMarksEndpoint:
    """Test the validate marks endpoint."""

    def test_validate_valid_marks(self, setup_database):
        """Test validating correct marks."""
        marks = {
            "MATH": 15.0,
            "PHYSICS": 14.0,
            "SCIENCES": 13.0,
            "ARABIC": 12.0,
            "FRENCH": 11.0,
            "ENGLISH": 13.0,
            "ISLAMIC": 15.0,
            "CIVICS": 14.0,
            "HISTORY_GEO": 12.0,
            "SPORT": 15.0,
        }

        response = client.post("/validate-marks", json={"stream_id": 2, "marks": marks})

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert len(data["errors"]) == 0

    def test_validate_missing_subjects(self, setup_database):
        """Test validation catches missing subjects."""
        marks = {"MATH": 15.0, "PHYSICS": 14.0}

        response = client.post("/validate-marks", json={"stream_id": 2, "marks": marks})

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert len(data["errors"]) > 0
        assert "Missing" in data["errors"][0]


class TestHelperFunctions:
    """Test helper functions."""

    def test_get_mention_tres_bien(self):
        """Test mention for Très Bien."""
        assert get_mention(16.0) == "Très Bien"
        assert get_mention(18.0) == "Très Bien"
        assert get_mention(19.5) == "Très Bien"

    def test_get_mention_bien(self):
        """Test mention for Bien."""
        assert get_mention(14.0) == "Bien"
        assert get_mention(15.0) == "Bien"
        assert get_mention(15.99) == "Bien"

    def test_get_mention_assez_bien(self):
        """Test mention for Assez Bien."""
        assert get_mention(12.0) == "Assez Bien"
        assert get_mention(13.0) == "Assez Bien"
        assert get_mention(13.99) == "Assez Bien"

    def test_get_mention_passable(self):
        """Test mention for Passable."""
        assert get_mention(10.0) == "Passable"
        assert get_mention(11.0) == "Passable"
        assert get_mention(11.99) == "Passable"

    def test_get_mention_non_admis(self):
        """Test mention for Non Admis."""
        assert get_mention(0.0) == "Non Admis"
        assert get_mention(5.0) == "Non Admis"
        assert get_mention(9.99) == "Non Admis"

    def test_calculate_subject_points(self):
        """Test subject points calculation."""
        assert calculate_subject_points(15.0, 7) == 105.0
        assert calculate_subject_points(10.0, 5) == 50.0
        assert calculate_subject_points(0.0, 3) == 0.0
        assert calculate_subject_points(20.0, 1) == 20.0


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_perfect_score(self, setup_database):
        """Test with perfect 20/20 in all subjects."""
        marks = {
            subject: 20.0
            for subject in [
                "MATH",
                "PHYSICS",
                "SCIENCES",
                "ARABIC",
                "FRENCH",
                "ENGLISH",
                "ISLAMIC",
                "CIVICS",
                "HISTORY_GEO",
                "SPORT",
            ]
        }

        response = client.post(
            "/calculate-average", json={"stream_id": 2, "marks": marks}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["average"] == 20.0
        assert data["mention"] == "Très Bien"
        assert data["passed"] is True

    def test_zero_score(self, setup_database):
        """Test with 0/20 in all subjects."""
        marks = {
            subject: 0.0
            for subject in [
                "MATH",
                "PHYSICS",
                "SCIENCES",
                "ARABIC",
                "FRENCH",
                "ENGLISH",
                "ISLAMIC",
                "CIVICS",
                "HISTORY_GEO",
                "SPORT",
            ]
        }

        response = client.post(
            "/calculate-average", json={"stream_id": 2, "marks": marks}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["average"] == 0.0
        assert data["mention"] == "Non Admis"
        assert data["passed"] is False

    def test_boundary_average_10(self, setup_database):
        """Test boundary case: exactly 10 average."""
        marks = {
            "MATH": 10.0,
            "PHYSICS": 10.0,
            "SCIENCES": 10.0,
            "ARABIC": 10.0,
            "FRENCH": 10.0,
            "ENGLISH": 10.0,
            "ISLAMIC": 10.0,
            "CIVICS": 10.0,
            "HISTORY_GEO": 10.0,
            "SPORT": 10.0,
        }

        response = client.post(
            "/calculate-average", json={"stream_id": 2, "marks": marks}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["average"] == 10.0
        assert data["mention"] == "Passable"
        assert data["passed"] is True

    def test_boundary_average_just_below_10(self, setup_database):
        """Test boundary case: just below 10 average."""
        marks = {
            "MATH": 9.9,
            "PHYSICS": 9.9,
            "SCIENCES": 9.9,
            "ARABIC": 9.9,
            "FRENCH": 9.9,
            "ENGLISH": 9.9,
            "ISLAMIC": 9.9,
            "CIVICS": 9.9,
            "HISTORY_GEO": 9.9,
            "SPORT": 9.9,
        }

        response = client.post(
            "/calculate-average", json={"stream_id": 2, "marks": marks}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["average"] < 10.0
        assert data["mention"] == "Non Admis"
        assert data["passed"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
