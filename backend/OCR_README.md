# OCR Processing Module

## Overview

The OCR (Optical Character Recognition) module converts Algerian Baccalaureat exam PDFs into structured text and LaTeX formats. It's optimized for:

- **Mathematical equations** (Math, Physics streams)
- **Arabic text** (right-to-left support)
- **Technical diagrams** (Technique Math - Génie Civil/Mécanique/Électrique)
- **Mixed content** (French + Arabic + Math formulas)

## Providers

### 1. Mathpix OCR (Primary for Math)
**Best for:** Mathematical equations and formulas

**Features:**
- Converts images to LaTeX
- High accuracy on complex equations
- Supports handwritten and printed math

**Setup:**
```bash
export MATHPIX_APP_ID="your_app_id"
export MATHPIX_APP_KEY="your_app_key"
```

**Get API keys:** https://mathpix.com/

### 2. Google Cloud Vision (Primary for Arabic)
**Best for:** Arabic text and general OCR

**Features:**
- Excellent Arabic language support
- Handwriting recognition
- Document structure detection

**Setup:**
```bash
export GOOGLE_VISION_API_KEY="your_api_key"
```

### 3. Tesseract OCR (Local Fallback)
**Best for:** Local processing without API calls

**Features:**
- Free and open-source
- Supports 100+ languages including Arabic
- No internet required

**Installation:**
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-ara tesseract-ocr-fra

# macOS
brew install tesseract tesseract-lang

# Windows
# Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
```

**Python package:**
```bash
cd backend
poetry add pytesseract
```

## Usage

### Basic Usage

```python
from ocr_engine import OCREngine
from pathlib import Path

# Initialize OCR engine (auto-selects best provider)
ocr = OCREngine(provider="auto")

# Process a single image
result = ocr.process_image(Path("equation.png"))
print(result.text)   # Plain text
print(result.latex)  # LaTeX format
print(result.confidence)  # Accuracy score

# Process a PDF exam
metadata = {
    "year": 2023,
    "stream_code": "MATH",
    "subject_code": "MATH",
    "filename": "bac_math_2023.pdf"
}

processed = ocr.process_exam(Path("exam.pdf"), metadata)
print(processed.combined_text)   # All text
print(processed.combined_latex)  # All LaTeX
print(processed.extracted_problems)  # Individual exercises
```

### Provider Selection

```python
# Force specific provider
ocr = OCREngine(provider="mathpix")  # For math-heavy content
ocr = OCREngine(provider="google")   # For Arabic text
ocr = OCREngine(provider="tesseract")  # For local processing

# Auto-selection (recommended)
ocr = OCREngine(provider="auto")  # Chooses best based on content
```

## Output Format

### OCRResult
```python
@dataclass
class OCRResult:
    text: str              # Plain text content
    latex: str             # LaTeX formatted math
    confidence: float      # 0.0 to 1.0 accuracy score
    page_number: int       # Page number in document
    is_math: bool          # Contains mathematical content
    is_arabic: bool        # Contains Arabic text
    metadata: Dict         # Provider-specific data
    errors: List[str]      # Any processing errors
```

### ProcessedExam
```python
@dataclass
class ProcessedExam:
    filename: str
    stream_code: str
    subject_code: str
    year: int
    pages: List[OCRResult]
    combined_text: str
    combined_latex: str
    extracted_problems: List[Dict]  # Individual exercises
```

## Processing Pipeline

```
PDF Input
    ↓
PDF to Images (pdf2image)
    ↓
Content Type Detection
    ↓
Provider Selection
    ├── Math Content → Mathpix
    ├── Arabic Text → Google Vision
    └── General → Tesseract
    ↓
OCR Processing
    ↓
LaTeX Generation (if math)
    ↓
Problem Extraction
    ↓
JSON Output
```

## File Structure

```
data/
├── exams/
│   └── 2023/
│       └── bac_math_2023_principal_math.pdf
└── processed/
    └── bac_math_2023_principal_math.pdf.processed.json
```

## Validation

### Accuracy Testing

```python
from ocr_engine import validate_ocr_accuracy

# Compare OCR output with reference text
metrics = validate_ocr_accuracy(
    reference_text="Correct solution...",
    ocr_text="OCR output..."
)

print(f"Accuracy: {metrics['accuracy']:.1%}")
print(f"Characters correct: {metrics['chars_correct']}/{metrics['chars_total']}")
```

### Test Suite

Run validation on sample exams:

```bash
cd backend
poetry run python -c "
from ocr_engine import OCREngine
from pathlib import Path

# Test on sample data
ocr = OCREngine()
result = ocr.process_image(Path('data/test/math_equation.png'))
print(f'Confidence: {result.confidence}')
"
```

## Performance Tips

1. **Resolution**: Use 300+ DPI for PDF conversion
2. **Preprocessing**: Clean images (deskew, denoise) before OCR
3. **Batching**: Process multiple pages in parallel
4. **Caching**: Cache OCR results to avoid reprocessing
5. **Fallbacks**: Always have backup providers configured

## Cost Estimation

### Mathpix
- Free tier: 1000 requests/month
- Paid: ~$0.005 per request

### Google Vision
- Free tier: 1000 requests/month
- Paid: ~$1.50 per 1000 requests

### Tesseract
- Free (local processing)

## Troubleshooting

### Common Issues

1. **"Tesseract not available"**
   - Install tesseract-ocr system package
   - Install pytesseract Python package

2. **Mathpix returning mock data**
   - Check MATHPIX_APP_ID and MATHPIX_APP_KEY environment variables
   - Verify API keys are valid

3. **Arabic text not recognized**
   - Use Google Vision for better Arabic support
   - Install Arabic language pack for Tesseract

4. **Low confidence scores**
   - Increase DPI (300+ recommended)
   - Preprocess images (deskew, enhance contrast)
   - Try different provider

## Next Steps

1. Obtain Mathpix API credentials for math equations
2. Set up Google Cloud Vision for Arabic text
3. Install Tesseract for local development
4. Test on real exam PDFs
5. Fine-tune problem extraction algorithm
6. Implement confidence thresholds for quality control
