"""
Bac Exam Data Acquisition Pipeline
Scrapes Algerian Baccalaureat exams from various sources.
Stores metadata in JSON format and PDFs in organized directories.

Sources:
- dzexams.com
- ency-education.com

Data Structure:
{
  "year": 2023,
  "stream_code": "MATH",
  "session": "principal",  # principal or rattrapage
  "subject_code": "MATH",
  "subject_name": "Mathematics",
  "filename": "bac_math_2023_principal.pdf",
  "url": "https://...",
  "download_date": "2024-02-17",
  "topics": ["complex_numbers", "limits"],
  "has_correction": true
}
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


class BacExamScraper:
    """Base class for scraping Bac exam data."""

    def __init__(self, base_dir: str = "data/exams"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        )

    def download_pdf(self, url: str, filepath: Path) -> bool:
        """Download PDF from URL to filepath."""
        try:
            response = self.session.get(url, timeout=30, stream=True)
            response.raise_for_status()

            filepath.parent.mkdir(parents=True, exist_ok=True)

            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            print(f"  [OK] Downloaded: {filepath.name}")
            return True

        except Exception as e:
            print(f"  [FAIL] Failed to download {url}: {e}")
            return False

    def save_metadata(self, metadata: Dict, filepath: Path) -> None:
        """Save exam metadata to JSON file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)


class DzExamsScraper(BacExamScraper):
    """Scraper for dzexams.com"""

    BASE_URL = "https://www.dzexams.com"

    # Stream mapping from dzexams to our codes
    STREAM_MAPPING = {
        "sciences": "SCI_EXP",
        "math": "MATH",
        "tech": "TECH_MATH",
        "gestion": "GESTION",
        "langues": "LANGUES",
        "lettres": "LETTRES",
        "arts": "ARTS",
    }

    # Subject mapping
    SUBJECT_MAPPING = {
        "mathematiques": "MATH",
        "maths": "MATH",
        "physique": "PHYSICS",
        "sciences": "SCIENCES",
        "arabe": "ARABIC",
        "francais": "FRENCH",
        "anglais": "ENGLISH",
        "philo": "PHILOSOPHY",
        "philosophie": "PHILOSOPHY",
        "histoire": "HISTORY_GEO",
        "geographie": "HISTORY_GEO",
        "islamique": "ISLAMIC",
        "economie": "ECONOMICS",
        "comptabilite": "ACCOUNTING",
        "gestion": "MANAGEMENT",
        "droit": "LAW",
    }

    def __init__(self):
        super().__init__()

    def scrape_exams(self, year: int, stream: str = None) -> List[Dict]:
        """
        Scrape exams for a specific year and optionally a specific stream.

        Args:
            year: Year of the exams (2015-2024)
            stream: Stream code (optional)

        Returns:
            List of exam metadata dictionaries
        """
        exams = []

        print(f"\n{'='*60}")
        print(f"Scraping Bac exams for year {year}")
        print(f"{'='*60}\n")

        # For each stream
        streams_to_scrape = [stream] if stream else list(self.STREAM_MAPPING.keys())

        for stream_key in streams_to_scrape:
            stream_code = self.STREAM_MAPPING.get(stream_key, stream_key)
            print(f"\n[STREAM] Stream: {stream_code}")

            # Try to scrape both sessions (principal and rattrapage)
            for session_type in ["principal", "rattrapage"]:
                print(f"  [SESSION] Session: {session_type}")

                # Construct URL (this is a template - actual URLs may vary)
                url = f"{self.BASE_URL}/bac/{year}/{stream_key}/{session_type}"

                try:
                    # In a real implementation, we would parse the HTML
                    # For now, we'll create a template structure
                    # since the actual scraping requires proper URL discovery

                    stream_exams = self._discover_exams(
                        url, year, stream_code, session_type
                    )
                    exams.extend(stream_exams)

                    time.sleep(1)  # Be respectful to the server

                except Exception as e:
                    print(f"    [ERROR] Error scraping {url}: {e}")
                    continue

        return exams

    def _discover_exams(
        self, url: str, year: int, stream_code: str, session: str
    ) -> List[Dict]:
        """
        Discover exam PDFs on a page.
        In a real implementation, this would parse the HTML and find links.
        """
        exams = []

        try:
            response = self.session.get(url, timeout=10)
            soup = BeautifulSoup(response.content, "html.parser")

            # Look for PDF links
            pdf_links = soup.find_all("a", href=lambda x: x and ".pdf" in x.lower())

            for link in pdf_links:
                href = link.get("href", "")
                full_url = urljoin(self.BASE_URL, href)

                # Extract subject from filename or link text
                subject_code = self._extract_subject_code(link.text, href)

                if subject_code:
                    exam_metadata = {
                        "year": year,
                        "stream_code": stream_code,
                        "session": session,
                        "subject_code": subject_code,
                        "subject_name": self._get_subject_name(subject_code),
                        "filename": f"bac_{stream_code.lower()}_{year}_{session}_{subject_code.lower()}.pdf",
                        "url": full_url,
                        "download_date": datetime.now().isoformat(),
                        "topics": [],  # To be filled manually or via OCR
                        "has_correction": "correction" in href.lower()
                        or "corrige" in href.lower(),
                        "source": "dzexams.com",
                    }

                    # Download the PDF
                    year_dir = self.base_dir / str(year)
                    pdf_path = year_dir / exam_metadata["filename"]

                    if self.download_pdf(full_url, pdf_path):
                        # Save metadata
                        metadata_path = year_dir / f"{exam_metadata['filename']}.json"
                        self.save_metadata(exam_metadata, metadata_path)
                        exams.append(exam_metadata)

        except Exception as e:
            print(f"    [ERROR] Error discovering exams: {e}")

        return exams

    def _extract_subject_code(self, text: str, href: str) -> Optional[str]:
        """Extract subject code from text or href."""
        text_lower = (text + " " + href).lower()

        for key, code in self.SUBJECT_MAPPING.items():
            if key in text_lower:
                return code

        return None

    def _get_subject_name(self, code: str) -> str:
        """Get subject name from code."""
        names = {
            "MATH": "Mathematics",
            "PHYSICS": "Physics",
            "SCIENCES": "Natural Sciences",
            "ARABIC": "Arabic",
            "FRENCH": "French",
            "ENGLISH": "English",
            "PHILOSOPHY": "Philosophy",
            "HISTORY_GEO": "History and Geography",
            "ISLAMIC": "Islamic Education",
            "ECONOMICS": "Economics",
            "ACCOUNTING": "Accounting",
            "MANAGEMENT": "Management",
            "LAW": "Law",
        }
        return names.get(code, code)


class ManualExamImporter:
    """Import exams that are manually downloaded or provided."""

    def __init__(self, base_dir: str = "data/exams"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_metadata_template(
        self,
        year: int,
        stream_code: str,
        subject_code: str,
        session: str = "principal",
        topics: List[str] = None,
    ) -> Dict:
        """Create a metadata template for manually imported exam."""
        return {
            "year": year,
            "stream_code": stream_code,
            "session": session,
            "subject_code": subject_code,
            "subject_name": subject_code,  # To be filled
            "filename": f"bac_{stream_code.lower()}_{year}_{session}_{subject_code.lower()}.pdf",
            "url": None,
            "download_date": datetime.now().isoformat(),
            "topics": topics or [],
            "has_correction": False,
            "source": "manual",
            "notes": "Manually imported exam",
        }

    def save_exam(self, pdf_path: Path, metadata: Dict) -> None:
        """Save exam PDF and metadata."""
        year_dir = self.base_dir / str(metadata["year"])
        year_dir.mkdir(parents=True, exist_ok=True)

        # Copy PDF to organized location
        dest_pdf = year_dir / metadata["filename"]
        import shutil

        shutil.copy2(pdf_path, dest_pdf)

        # Save metadata
        metadata_path = year_dir / f"{metadata['filename']}.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"[OK] Saved exam: {dest_pdf}")


def create_sample_data():
    """Create sample metadata for demonstration purposes."""
    print("\n" + "=" * 60)
    print("Creating sample exam metadata")
    print("=" * 60 + "\n")

    base_dir = Path("data/exams")
    base_dir.mkdir(parents=True, exist_ok=True)

    sample_exams = [
        {
            "year": 2023,
            "stream_code": "MATH",
            "session": "principal",
            "subject_code": "MATH",
            "subject_name": "Mathematics",
            "filename": "bac_math_2023_principal_math.pdf",
            "url": "https://example.com/bac/math/2023/principal",
            "download_date": datetime.now().isoformat(),
            "topics": ["complex_numbers", "limits", "derivatives", "integrals"],
            "has_correction": True,
            "source": "sample",
        },
        {
            "year": 2023,
            "stream_code": "MATH",
            "session": "principal",
            "subject_code": "PHYSICS",
            "subject_name": "Physics",
            "filename": "bac_math_2023_principal_physics.pdf",
            "url": "https://example.com/bac/math/2023/principal",
            "download_date": datetime.now().isoformat(),
            "topics": ["electromagnetism", "mechanics", "thermodynamics"],
            "has_correction": True,
            "source": "sample",
        },
        {
            "year": 2023,
            "stream_code": "SCI_EXP",
            "session": "principal",
            "subject_code": "SCIENCES",
            "subject_name": "Natural Sciences",
            "filename": "bac_sci_exp_2023_principal_sciences.pdf",
            "url": "https://example.com/bac/sciences/2023/principal",
            "download_date": datetime.now().isoformat(),
            "topics": ["genetics", "ecology", "cell_biology"],
            "has_correction": True,
            "source": "sample",
        },
    ]

    for exam in sample_exams:
        year_dir = base_dir / str(exam["year"])
        year_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = year_dir / f"{exam['filename']}.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(exam, f, indent=2, ensure_ascii=False)

        print(f"[OK] Created sample metadata: {metadata_path}")

    print(f"\n{'='*60}")
    print(f"Sample data created: {len(sample_exams)} exams")
    print(f"{'='*60}")


if __name__ == "__main__":
    # For demonstration, create sample data
    # In production, you would run:
    # scraper = DzExamsScraper()
    # exams = scraper.scrape_exams(year=2023)

    create_sample_data()

    print("\n[OK] Data acquisition pipeline ready!")
    print("\nNote: Actual web scraping would require:")
    print("  1. Proper URL discovery from dzexams.com")
    print("  2. Handling JavaScript-rendered content (may need Selenium)")
    print("  3. Rate limiting and respectful crawling")
    print("  4. Manual verification of downloaded content")
