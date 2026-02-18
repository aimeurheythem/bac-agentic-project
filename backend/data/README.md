# Bac Exam Data Acquisition Pipeline

This module handles the scraping and organization of Algerian Baccalaureat exam data.

## Structure

```
data/exams/
├── 2015/
├── 2016/
├── 2017/
├── 2018/
├── 2019/
├── 2020/
├── 2021/
├── 2022/
├── 2023/
│   ├── bac_math_2023_principal_math.pdf
│   ├── bac_math_2023_principal_math.pdf.json
│   └── ...
└── 2024/
```

## Metadata Schema

Each exam has a corresponding JSON metadata file:

```json
{
  "year": 2023,
  "stream_code": "MATH",
  "session": "principal",
  "subject_code": "MATH",
  "subject_name": "Mathematics",
  "filename": "bac_math_2023_principal_math.pdf",
  "url": "https://example.com/...",
  "download_date": "2024-02-17T19:01:00",
  "topics": ["complex_numbers", "limits", "derivatives"],
  "has_correction": true,
  "source": "dzexams.com"
}
```

## Components

### 1. BacExamScraper (Base Class)
Base scraper with common functionality:
- PDF downloading
- Metadata management
- Directory organization

### 2. DzExamsScraper
Scraper for dzexams.com:
- Discovers exam PDFs
- Maps stream codes
- Downloads exams by year/stream

### 3. ManualExamImporter
For manually imported exams:
- Creates metadata templates
- Organizes existing PDFs
- Tracks manual uploads

## Usage

### Create Sample Data
```bash
cd backend
poetry run python scraper.py
```

### Scrape Specific Year
```python
from scraper import DzExamsScraper

scraper = DzExamsScraper()
exams = scraper.scrape_exams(year=2023)
```

### Import Manual Exams
```python
from scraper import ManualExamImporter
from pathlib import Path

importer = ManualExamImporter()
metadata = importer.create_metadata_template(
    year=2023,
    stream_code="MATH",
    subject_code="MATH",
    topics=["complex_numbers", "integrals"]
)
importer.save_exam(Path("path/to/exam.pdf"), metadata)
```

## Data Sources

1. **dzexams.com** - Primary source for recent exams
2. **ency-education.com** - Alternative source
3. **Manual uploads** - Direct PDF imports

## Important Notes

- Actual web scraping requires proper URL discovery
- Some sites may use JavaScript rendering (requires Selenium)
- Always implement rate limiting (1+ second delays)
- Respect robots.txt and terms of service
- Verify downloaded content accuracy

## Next Steps

1. Implement actual dzexams.com URL discovery
2. Add Selenium support for JavaScript sites
3. Implement OCR processing for exams (Task 1.4)
4. Set up cloud storage (S3/Supabase) for production
