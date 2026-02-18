"""
Database initialization script for Algerian Baccalaureat system.
Creates all tables and seeds them with the 7 official streams, subjects, and coefficients.
"""

from database import Base, SessionLocal, engine
from models import Coefficient, Stream, Subject

STREAMS_DATA = [
    {
        "code": "SCI_EXP",
        "name": "Sciences Expérimentales",
        "name_ar": "علوم تجريبية",
        "has_options": False,
    },
    {
        "code": "MATH",
        "name": "Mathématiques",
        "name_ar": "رياضيات",
        "has_options": False,
    },
    {
        "code": "TECH_MATH",
        "name": "Technique Mathématique",
        "name_ar": "تقني رياضي",
        "has_options": True,
    },
    {
        "code": "GESTION",
        "name": "Gestion et Économie",
        "name_ar": "تسيير واقتصاد",
        "has_options": False,
    },
    {
        "code": "LANGUES",
        "name": "Langues Étrangères",
        "name_ar": "لغات أجنبية",
        "has_options": False,
    },
    {
        "code": "LETTRES",
        "name": "Lettres et Philosophie",
        "name_ar": "آداب وفلسفة",
        "has_options": False,
    },
    {
        "code": "ARTS",
        "name": "Arts",
        "name_ar": "فنون",
        "has_options": False,
    },
]


SUBJECTS_DATA = [
    # Core subjects (common to all)
    {"code": "ARABIC", "name": "Arabic", "name_ar": "العربية", "category": "language"},
    {"code": "FRENCH", "name": "French", "name_ar": "الفرنسية", "category": "language"},
    {
        "code": "ENGLISH",
        "name": "English",
        "name_ar": "الإنجليزية",
        "category": "language",
    },
    {
        "code": "ISLAMIC",
        "name": "Islamic Education",
        "name_ar": "تربية إسلامية",
        "category": "core",
    },
    {"code": "CIVICS", "name": "Civics", "name_ar": "تربية مدنية", "category": "core"},
    {
        "code": "HISTORY_GEO",
        "name": "History and Geography",
        "name_ar": "تاريخ وجغرافيا",
        "category": "core",
    },
    {
        "code": "SPORT",
        "name": "Physical Education",
        "name_ar": "تربية بدنية",
        "category": "core",
    },
    # Scientific subjects
    {
        "code": "MATH",
        "name": "Mathematics",
        "name_ar": "رياضيات",
        "category": "scientific",
    },
    {
        "code": "PHYSICS",
        "name": "Physics",
        "name_ar": "فيزياء",
        "category": "scientific",
    },
    {
        "code": "SCIENCES",
        "name": "Natural Sciences",
        "name_ar": "علوم طبيعية",
        "category": "scientific",
    },
    # Technical subjects
    {
        "code": "TECHNOLOGY",
        "name": "Technology",
        "name_ar": "تكنولوجيا",
        "category": "technical",
    },
    # Literary/Philosophy subjects
    {
        "code": "PHILOSOPHY",
        "name": "Philosophy",
        "name_ar": "فلسفة",
        "category": "literary",
    },
    {
        "code": "ARABIC_LIT",
        "name": "Arabic Literature",
        "name_ar": "أدب عربي",
        "category": "literary",
    },
    # Economics subjects
    {
        "code": "ECONOMICS",
        "name": "Economics",
        "name_ar": "اقتصاد",
        "category": "economic",
    },
    {
        "code": "ACCOUNTING",
        "name": "Accounting",
        "name_ar": "محاسبة",
        "category": "economic",
    },
    {"code": "LAW", "name": "Law", "name_ar": "قانون", "category": "economic"},
    {
        "code": "MANAGEMENT",
        "name": "Management",
        "name_ar": "تسيير",
        "category": "economic",
    },
    # Language subjects
    {
        "code": "SPANISH",
        "name": "Spanish",
        "name_ar": "إسبانية",
        "category": "language",
    },
    {"code": "GERMAN", "name": "German", "name_ar": "ألمانية", "category": "language"},
    {
        "code": "ITALIAN",
        "name": "Italian",
        "name_ar": "إيطالية",
        "category": "language",
    },
    # Arts subjects
    {"code": "DRAWING", "name": "Drawing", "name_ar": "رسم", "category": "art"},
    {"code": "MUSIC", "name": "Music", "name_ar": "موسيقى", "category": "art"},
    {"code": "CINEMA", "name": "Cinema", "name_ar": "سينما", "category": "art"},
]


COEFFICIENTS_DATA = [
    # Sciences Expérimentales
    {"stream_code": "SCI_EXP", "subject_code": "SCIENCES", "coefficient": 6},
    {"stream_code": "SCI_EXP", "subject_code": "PHYSICS", "coefficient": 5},
    {"stream_code": "SCI_EXP", "subject_code": "MATH", "coefficient": 5},
    {"stream_code": "SCI_EXP", "subject_code": "ARABIC", "coefficient": 3},
    {"stream_code": "SCI_EXP", "subject_code": "FRENCH", "coefficient": 2},
    {"stream_code": "SCI_EXP", "subject_code": "ENGLISH", "coefficient": 2},
    {"stream_code": "SCI_EXP", "subject_code": "ISLAMIC", "coefficient": 2},
    {"stream_code": "SCI_EXP", "subject_code": "CIVICS", "coefficient": 1},
    {"stream_code": "SCI_EXP", "subject_code": "HISTORY_GEO", "coefficient": 2},
    {"stream_code": "SCI_EXP", "subject_code": "SPORT", "coefficient": 1},
    # Mathématiques
    {"stream_code": "MATH", "subject_code": "MATH", "coefficient": 7},
    {"stream_code": "MATH", "subject_code": "PHYSICS", "coefficient": 6},
    {"stream_code": "MATH", "subject_code": "SCIENCES", "coefficient": 4},
    {"stream_code": "MATH", "subject_code": "ARABIC", "coefficient": 3},
    {"stream_code": "MATH", "subject_code": "FRENCH", "coefficient": 2},
    {"stream_code": "MATH", "subject_code": "ENGLISH", "coefficient": 2},
    {"stream_code": "MATH", "subject_code": "ISLAMIC", "coefficient": 2},
    {"stream_code": "MATH", "subject_code": "CIVICS", "coefficient": 1},
    {"stream_code": "MATH", "subject_code": "HISTORY_GEO", "coefficient": 2},
    {"stream_code": "MATH", "subject_code": "SPORT", "coefficient": 1},
    # Technique Mathématique
    {"stream_code": "TECH_MATH", "subject_code": "MATH", "coefficient": 6},
    {"stream_code": "TECH_MATH", "subject_code": "PHYSICS", "coefficient": 6},
    {
        "stream_code": "TECH_MATH",
        "subject_code": "TECHNOLOGY",
        "coefficient": 6,
        "is_specialty": True,
    },
    {"stream_code": "TECH_MATH", "subject_code": "ARABIC", "coefficient": 3},
    {"stream_code": "TECH_MATH", "subject_code": "FRENCH", "coefficient": 2},
    {"stream_code": "TECH_MATH", "subject_code": "ENGLISH", "coefficient": 2},
    {"stream_code": "TECH_MATH", "subject_code": "ISLAMIC", "coefficient": 2},
    {"stream_code": "TECH_MATH", "subject_code": "CIVICS", "coefficient": 1},
    {"stream_code": "TECH_MATH", "subject_code": "HISTORY_GEO", "coefficient": 2},
    {"stream_code": "TECH_MATH", "subject_code": "SPORT", "coefficient": 1},
    # Gestion et Économie
    {"stream_code": "GESTION", "subject_code": "ACCOUNTING", "coefficient": 6},
    {"stream_code": "GESTION", "subject_code": "ECONOMICS", "coefficient": 5},
    {"stream_code": "GESTION", "subject_code": "MANAGEMENT", "coefficient": 4},
    {"stream_code": "GESTION", "subject_code": "MATH", "coefficient": 4},
    {"stream_code": "GESTION", "subject_code": "LAW", "coefficient": 3},
    {"stream_code": "GESTION", "subject_code": "ARABIC", "coefficient": 3},
    {"stream_code": "GESTION", "subject_code": "FRENCH", "coefficient": 2},
    {"stream_code": "GESTION", "subject_code": "ENGLISH", "coefficient": 2},
    {"stream_code": "GESTION", "subject_code": "ISLAMIC", "coefficient": 2},
    {"stream_code": "GESTION", "subject_code": "CIVICS", "coefficient": 1},
    {"stream_code": "GESTION", "subject_code": "HISTORY_GEO", "coefficient": 2},
    {"stream_code": "GESTION", "subject_code": "SPORT", "coefficient": 1},
    # Langues Étrangères
    {"stream_code": "LANGUES", "subject_code": "ARABIC", "coefficient": 5},
    {"stream_code": "LANGUES", "subject_code": "FRENCH", "coefficient": 5},
    {"stream_code": "LANGUES", "subject_code": "ENGLISH", "coefficient": 5},
    {
        "stream_code": "LANGUES",
        "subject_code": "SPANISH",
        "coefficient": 4,
        "is_specialty": True,
    },  # or GERMAN or ITALIAN
    {"stream_code": "LANGUES", "subject_code": "PHILOSOPHY", "coefficient": 2},
    {"stream_code": "LANGUES", "subject_code": "ISLAMIC", "coefficient": 2},
    {"stream_code": "LANGUES", "subject_code": "CIVICS", "coefficient": 1},
    {"stream_code": "LANGUES", "subject_code": "HISTORY_GEO", "coefficient": 2},
    {"stream_code": "LANGUES", "subject_code": "SPORT", "coefficient": 1},
    # Lettres et Philosophie
    {"stream_code": "LETTRES", "subject_code": "PHILOSOPHY", "coefficient": 6},
    {"stream_code": "LETTRES", "subject_code": "ARABIC_LIT", "coefficient": 6},
    {"stream_code": "LETTRES", "subject_code": "ARABIC", "coefficient": 3},
    {"stream_code": "LETTRES", "subject_code": "FRENCH", "coefficient": 3},
    {"stream_code": "LETTRES", "subject_code": "ENGLISH", "coefficient": 2},
    {"stream_code": "LETTRES", "subject_code": "ISLAMIC", "coefficient": 2},
    {"stream_code": "LETTRES", "subject_code": "CIVICS", "coefficient": 1},
    {"stream_code": "LETTRES", "subject_code": "HISTORY_GEO", "coefficient": 2},
    {"stream_code": "LETTRES", "subject_code": "SPORT", "coefficient": 1},
    # Arts
    {
        "stream_code": "ARTS",
        "subject_code": "DRAWING",
        "coefficient": 6,
        "is_specialty": True,
    },
    {"stream_code": "ARTS", "subject_code": "MATH", "coefficient": 2},
    {"stream_code": "ARTS", "subject_code": "PHILOSOPHY", "coefficient": 2},
    {"stream_code": "ARTS", "subject_code": "ARABIC", "coefficient": 3},
    {"stream_code": "ARTS", "subject_code": "FRENCH", "coefficient": 2},
    {"stream_code": "ARTS", "subject_code": "ENGLISH", "coefficient": 2},
    {"stream_code": "ARTS", "subject_code": "ISLAMIC", "coefficient": 2},
    {"stream_code": "ARTS", "subject_code": "CIVICS", "coefficient": 1},
    {"stream_code": "ARTS", "subject_code": "HISTORY_GEO", "coefficient": 2},
    {"stream_code": "ARTS", "subject_code": "SPORT", "coefficient": 1},
]


def init_database():
    """Create all tables in the database."""
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("Tables created successfully!")


def seed_data():
    """Seed the database with initial data."""
    db = SessionLocal()

    try:
        print("Seeding streams...")
        for stream_data in STREAMS_DATA:
            existing = (
                db.query(Stream).filter(Stream.code == stream_data["code"]).first()
            )
            if not existing:
                stream = Stream(**stream_data)
                db.add(stream)
                print(f"  - Added {stream_data['name']}")

        db.commit()
        print(f"Streams seeded successfully!\n")

        print("Seeding subjects...")
        for subject_data in SUBJECTS_DATA:
            existing = (
                db.query(Subject).filter(Subject.code == subject_data["code"]).first()
            )
            if not existing:
                subject = Subject(**subject_data)
                db.add(subject)
                print(f"  - Added {subject_data['name']}")

        db.commit()
        print(f"Subjects seeded successfully!\n")

        print("Seeding coefficients...")
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
        print("Coefficients seeded successfully!\n")

        print("=" * 50)
        print("Database initialization completed!")
        print(f"Total streams: {db.query(Stream).count()}")
        print(f"Total subjects: {db.query(Subject).count()}")
        print(f"Total coefficients: {db.query(Coefficient).count()}")
        print("=" * 50)

    except Exception as e:
        print(f"Error seeding data: {e}")
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    init_database()
    seed_data()
