import math

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.flowables import Flowable

# ── Palette ───────────────────────────────────────────────────────────────────
BLACK = colors.HexColor("#0f172a")
DARK = colors.HexColor("#1e293b")
MID = colors.HexColor("#475569")
LIGHT = colors.HexColor("#94a3b8")
BORDER = colors.HexColor("#e2e8f0")
BG = colors.HexColor("#f8fafc")
ACCENT = colors.HexColor("#3b82f6")
GREEN = colors.HexColor("#22c55e")
ORANGE = colors.HexColor("#f59e0b")
WHITE = colors.white

W, H = A4  # 595 x 842


# ── Styles ────────────────────────────────────────────────────────────────────
def S(name, **kw):
    return ParagraphStyle(name, **kw)


slide_label = S(
    "sl", fontSize=8, fontName="Helvetica", textColor=LIGHT, spaceBefore=0, spaceAfter=2
)
section_head = S(
    "sh",
    fontSize=22,
    fontName="Helvetica-Bold",
    textColor=BLACK,
    spaceBefore=0,
    spaceAfter=6,
    leading=28,
)
body = S(
    "b", fontSize=10, fontName="Helvetica", textColor=DARK, leading=16, spaceAfter=4
)
body_bold = S(
    "bb",
    fontSize=10,
    fontName="Helvetica-Bold",
    textColor=BLACK,
    leading=16,
    spaceAfter=4,
)
small = S("sm", fontSize=8.5, fontName="Helvetica", textColor=MID, leading=13)
bullet_style = S(
    "bu",
    fontSize=9.5,
    fontName="Helvetica",
    textColor=DARK,
    leading=15,
    leftIndent=14,
    spaceAfter=3,
    bulletIndent=4,
    bulletFontName="Helvetica",
    bulletFontSize=9.5,
)
number_style = S(
    "num",
    fontSize=28,
    fontName="Helvetica-Bold",
    textColor=ACCENT,
    alignment=TA_CENTER,
    leading=32,
)
label_style = S(
    "lbl",
    fontSize=9,
    fontName="Helvetica-Bold",
    textColor=BLACK,
    alignment=TA_CENTER,
    leading=12,
)
tag_style = S(
    "tag", fontSize=9, fontName="Helvetica-Bold", textColor=WHITE, alignment=TA_CENTER
)

# Cell paragraph style — used for wrapping text inside table cells
cell_body = S("cb", fontSize=8.5, fontName="Helvetica", textColor=DARK, leading=13)
cell_mid = S("cm", fontSize=8.5, fontName="Helvetica", textColor=MID, leading=13)
cell_acc = S(
    "ca", fontSize=8.5, fontName="Helvetica-Bold", textColor=ACCENT, leading=13
)
cell_green = S(
    "cg", fontSize=8.5, fontName="Helvetica-Bold", textColor=GREEN, leading=13
)
cell_hdr = S("ch", fontSize=8.5, fontName="Helvetica-Bold", textColor=WHITE, leading=13)


def P(text, style=None):
    """Wrap text in a Paragraph so it wraps inside table cells."""
    return Paragraph(text, style or cell_body)


def rule(color=BORDER, w=1):
    return HRFlowable(
        width="100%", thickness=w, color=color, spaceAfter=8, spaceBefore=4
    )


def sp(h=8):
    return Spacer(1, h)


def bullet(text):
    return Paragraph(f"<bullet>\u2022</bullet> {text}", bullet_style)


# ── Custom Flowables ──────────────────────────────────────────────────────────


class CoverPage(Flowable):
    def wrap(self, *a):
        return W - 4 * cm, 220

    def draw(self):
        c = self.canv
        cw = W - 4 * cm

        # Background card
        c.setFillColor(BLACK)
        c.roundRect(0, 0, cw, 220, 14, fill=1, stroke=0)

        # Left accent bar
        c.setFillColor(ACCENT)
        c.rect(0, 0, 5, 220, fill=1, stroke=0)

        # Title
        c.setFillColor(WHITE)
        c.setFont("Helvetica-Bold", 32)
        c.drawCentredString(cw / 2, 168, "BAC AGENT")

        # Subtitle
        c.setFillColor(colors.HexColor("#93c5fd"))
        c.setFont("Helvetica-Bold", 14)
        c.drawCentredString(
            cw / 2, 143, "AI-Powered Algerian Baccalaur\u00e9at Platform"
        )

        # Divider
        c.setStrokeColor(colors.HexColor("#334155"))
        c.setLineWidth(0.7)
        c.line(cw / 2 - 120, 134, cw / 2 + 120, 134)

        # Tech badge row
        tags = [
            ("FastAPI", "#2563eb"),
            ("React + TypeScript", "#7c3aed"),
            ("RAG Pipeline", "#059669"),
            ("OpenAI GPT-4o", "#d97706"),
        ]
        bw = 84
        gap = 8
        total = len(tags) * bw + (len(tags) - 1) * gap
        tx = (cw - total) / 2
        for i, (t, col) in enumerate(tags):
            c.setFillColor(colors.HexColor(col))
            c.roundRect(tx + i * (bw + gap), 112, bw, 17, 8, fill=1, stroke=0)
            c.setFillColor(WHITE)
            c.setFont("Helvetica-Bold", 7.5)
            c.drawCentredString(tx + i * (bw + gap) + bw / 2, 117, t)

        # Description box
        c.setFillColor(colors.HexColor("#1e293b"))
        c.roundRect(cw / 2 - 185, 30, 370, 72, 8, fill=1, stroke=0)
        c.setFillColor(LIGHT)
        c.setFont("Helvetica", 9)
        lines2 = [
            "A smart tutoring system for Algerian Bac students.",
            "7 streams  \u2022  5 AI modes  \u2022  RAG on real past exams  \u2022  LaTeX rendering",
        ]
        for i, ln in enumerate(lines2):
            c.drawCentredString(cw / 2, 76 - i * 18, ln)

        # Bottom tag
        c.setFillColor(LIGHT)
        c.setFont("Helvetica", 8)
        c.drawCentredString(
            cw / 2, 14, "Product Architecture & Feature Overview   \u2022   2026"
        )


class ArchDiagram(Flowable):
    def __init__(self, width, height):
        Flowable.__init__(self)
        self.bw = width
        self.bh = height

    def wrap(self, *a):
        return self.bw, self.bh

    def draw(self):
        c = self.canv
        w, h = self.bw, self.bh

        c.setFillColor(BG)
        c.roundRect(0, 0, w, h, 10, fill=1, stroke=0)

        def box(x, y, bw, bh, label, sublabel="", fill=ACCENT, tx=WHITE):
            c.setFillColor(fill)
            c.setStrokeColor(colors.HexColor("#cbd5e1"))
            c.roundRect(x, y, bw, bh, 6, fill=1, stroke=1)
            c.setFillColor(tx)
            c.setFont("Helvetica-Bold", 8.5)
            c.drawCentredString(
                x + bw / 2, y + bh / 2 + 2 if sublabel else y + bh / 2 - 3, label
            )
            if sublabel:
                c.setFont("Helvetica", 7)
                c.setFillColor(
                    colors.HexColor("#cbd5e1")
                    if fill == BLACK
                    or fill == DARK
                    or fill == colors.HexColor("#0f172a")
                    else LIGHT
                )
                c.drawCentredString(x + bw / 2, y + bh / 2 - 10, sublabel)

        def arrowline(x1, y1, x2, y2):
            c.setStrokeColor(LIGHT)
            c.setLineWidth(0.8)
            c.line(x1, y1, x2, y2)
            ang = math.atan2(y2 - y1, x2 - x1)
            size = 5
            pts = [
                x2,
                y2,
                x2 - size * math.cos(ang - 0.4),
                y2 - size * math.sin(ang - 0.4),
                x2 - size * math.cos(ang + 0.4),
                y2 - size * math.sin(ang + 0.4),
            ]
            c.setFillColor(LIGHT)
            p = c.beginPath()
            p.moveTo(pts[0], pts[1])
            p.lineTo(pts[2], pts[3])
            p.lineTo(pts[4], pts[5])
            p.close()
            c.drawPath(p, fill=1, stroke=0)

        # Row 1 — Student (top center)
        box(w / 2 - 50, h - 58, 100, 38, "Student", fill=BLACK)

        # Arrow down
        arrowline(w / 2, h - 58, w / 2, h - 90)

        # Row 2 — FastAPI
        box(w / 2 - 60, h - 128, 120, 34, "FastAPI Backend", fill=DARK)

        # Row 3 — 3 Services
        bw2 = 100
        gap2 = 12
        total2 = 3 * bw2 + 2 * gap2
        sx = (w - total2) / 2
        fills3 = [
            colors.HexColor("#2563eb"),
            colors.HexColor("#7c3aed"),
            colors.HexColor("#059669"),
        ]
        labels3 = ["Tutor Agent", "RAG Pipeline", "OCR Engine"]
        for i, (lbl, fc) in enumerate(zip(labels3, fills3)):
            bx = sx + i * (bw2 + gap2)
            box(bx, h - 215, bw2, 44, lbl, fill=fc)
            arrowline(w / 2, h - 128, bx + bw2 / 2, h - 171)

        # Row 4 — Data layer
        bw3 = 92
        gap3 = 12
        total3 = 3 * bw3 + 2 * gap3
        sx3 = (w - total3) / 2
        data_labels = ["SQLite DB", "Vector Store", "PDF Exams"]
        for i, lbl in enumerate(data_labels):
            dbx = sx3 + i * (bw3 + gap3)
            box(dbx, h - 300, bw3, 40, lbl, fill=colors.HexColor("#0f172a"), tx=WHITE)
            arrowline(sx + i * (bw2 + gap2) + bw2 / 2, h - 215, dbx + bw3 / 2, h - 260)


class UXFlowDiagram(Flowable):
    def __init__(self, width, height):
        Flowable.__init__(self)
        self.bw = width
        self.bh = height

    def wrap(self, *a):
        return self.bw, self.bh

    def draw(self):
        c = self.canv
        w, h = self.bw, self.bh

        c.setFillColor(BG)
        c.roundRect(0, 0, w, h, 10, fill=1, stroke=0)

        steps = [
            ("1", "Onboarding", ""),
            ("2", "Dashboard", ""),
            ("3", "AI Reply", ""),
            ("4", "Follow-up", ""),
        ]
        bw = 86
        bh = 62
        gap = 24
        total = len(steps) * bw + (len(steps) - 1) * gap
        sx = (w - total) / 2
        fills = [BLACK, ACCENT, colors.HexColor("#7c3aed"), GREEN]

        for i, (num, title, desc) in enumerate(steps):
            x = sx + i * (bw + gap)
            y = (h - bh) / 2

            c.setFillColor(fills[i])
            c.roundRect(x, y, bw, bh, 8, fill=1, stroke=0)

            c.setFillColor(WHITE)
            c.setFont("Helvetica-Bold", 12)
            c.drawCentredString(x + bw / 2, y + bh - 18, num)
            c.setFont("Helvetica-Bold", 8)
            c.drawCentredString(x + bw / 2, y + bh - 32, title)

            lines = desc.split("\n")
            for j, ln in enumerate(lines):
                c.setFont("Helvetica", 7)
                c.setFillColor(colors.HexColor("#cbd5e1"))
                c.drawCentredString(x + bw / 2, y + 8 + (len(lines) - 1 - j) * 12, ln)

            if i < len(steps) - 1:
                ax = x + bw + 3
                ay = y + bh / 2
                c.setStrokeColor(LIGHT)
                c.setLineWidth(1)
                c.line(ax, ay, ax + gap - 6, ay)
                c.setFillColor(LIGHT)
                pts = [
                    ax + gap - 6,
                    ay,
                    ax + gap - 11,
                    ay + 3.5,
                    ax + gap - 11,
                    ay - 3.5,
                ]
                p = c.beginPath()
                p.moveTo(pts[0], pts[1])
                p.lineTo(pts[2], pts[3])
                p.lineTo(pts[4], pts[5])
                p.close()
                c.drawPath(p, fill=1, stroke=0)


class ClosingPage(Flowable):
    def wrap(self, *a):
        return W - 4 * cm, 280

    def draw(self):
        c = self.canv
        cw = W - 4 * cm

        c.setFillColor(BG)
        c.roundRect(0, 0, cw, 280, 12, fill=1, stroke=0)
        c.setFillColor(ACCENT)
        c.rect(0, 276, cw, 4, fill=1, stroke=0)

        c.setFillColor(BLACK)
        c.setFont("Helvetica-Bold", 20)
        c.drawCentredString(
            cw / 2, 242, "Build all this and the app will be ready to ship."
        )

        c.setStrokeColor(BORDER)
        c.setLineWidth(0.8)
        c.line(cw / 2 - 100, 230, cw / 2 + 100, 230)

        checkmarks = [
            "FastAPI backend with coefficient engine covering all 7 streams",
            "RAG pipeline trained on 2015–2024 Algerian Bac exam PDFs",
            "5 specialized AI chat modes each with custom tutor prompting",
            "React SPA with in-page chat and collapsible history sidebar",
            "OCR engine to ingest scanned PDF exams (Mathpix / Vision / Tesseract)",
            "7 streams with stream-aware and specialty-aware AI prompting",
        ]
        c.setFont("Helvetica", 9.5)
        for i, line in enumerate(checkmarks):
            c.setFillColor(GREEN)
            c.circle(cw / 2 - 175, 195 - i * 22 + 3, 4, fill=1, stroke=0)
            c.setFillColor(DARK)
            c.drawString(cw / 2 - 167, 192 - i * 22, line)

        c.setFillColor(LIGHT)
        c.setFont("Helvetica", 8)
        c.drawCentredString(
            cw / 2,
            18,
            "Bac Agent   \u2022   Built for Algerian Bac Students   \u2022   2026",
        )


# ── Build ─────────────────────────────────────────────────────────────────────
out = "c:/Users/rachid/Desktop/bac-agentic-project/BacAgent_Presentation.pdf"
doc = SimpleDocTemplate(
    out,
    pagesize=A4,
    leftMargin=2 * cm,
    rightMargin=2 * cm,
    topMargin=2 * cm,
    bottomMargin=2 * cm,
)

story = []
PM = 15 * cm  # printable max width

# ── SLIDE 1 — COVER ──────────────────────────────────────────────────────────
story += [
    sp(30),
    CoverPage(),
]
story.append(PageBreak())

# ── SLIDE 2 — OVERVIEW ───────────────────────────────────────────────────────
story += [
    Paragraph("OVERVIEW", slide_label),
    Paragraph("What is Bac Agent?", section_head),
    rule(),
    sp(4),
    Paragraph(
        "Bac Agent is an AI-powered tutoring platform built exclusively for Algerian "
        "Baccalauréat students. It combines a context-aware LLM tutor with a "
        "Retrieval-Augmented Generation (RAG) pipeline trained on real past exams "
        "(2015–2024), giving every student personalized, curriculum-aligned academic support.",
        body,
    ),
    sp(12),
]


def stat_card(num, label, sub=""):
    inner = [[Paragraph(num, number_style)], [Paragraph(label, label_style)]]
    if sub:
        inner.append([Paragraph(sub, small)])
    t = Table(inner, colWidths=[PM / 3 - 8])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), BG),
                ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 12),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    return t


stats_row = Table(
    [
        [
            stat_card("7", "Official Streams", "MATH · SCI · TECH · GESTION…"),
            stat_card("5", "AI Chat Modes", "Tutor · Exercise · Exam…"),
            stat_card("10", "Years of Exams", "2015 – 2024 PDFs processed"),
        ]
    ],
    colWidths=[PM / 3 - 4] * 3,
    hAlign="CENTER",
)
stats_row.setStyle(
    TableStyle(
        [
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]
    )
)
story.append(stats_row)
story += [
    sp(16),
    Paragraph(
        "<b>Core Problem:</b> Algerian students — especially in technical streams — "
        "have no structured AI tool that respects the exact Ministry of Education curriculum, "
        "official grading schemes (Barème), and each stream's unique coefficient weighting.",
        body,
    ),
    PageBreak(),
]

# ── SLIDE 3 — STREAMS ────────────────────────────────────────────────────────
story += [
    Paragraph("CURRICULUM COVERAGE", slide_label),
    Paragraph("7 Official Bac Streams (Filières)", section_head),
    rule(),
    sp(6),
]

streams_data = [
    ["Stream", "Top Subjects & Coefficients", "AI Complexity"],
    ["Mathématiques", "Math (7)  ·  Physics (6)", "Very High"],
    ["Sciences Expér.", "Sciences (6)  ·  Physics (5)", "High"],
    ["Technique Math", "Math (6)  ·  Physics (6)  ·  Tech (6)", "Extreme — 4 options"],
    ["Gestion & Économie", "Accounting (6)  ·  Economics", "Medium"],
    ["Langues Étrangères", "Arabic / French / English (5 ea.)", "Medium-High"],
    ["Lettres & Philosophie", "Philosophy (6)  ·  Arabic Lit (6)", "High"],
    ["Arts", "Drawing / Art Specialty (6)", "Niche"],
]
t_streams = Table(streams_data, colWidths=[5.2 * cm, 6.4 * cm, 4.2 * cm])
t_streams.setStyle(
    TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), BLACK),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, BG]),
            ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TEXTCOLOR", (2, 1), (-1, -1), MID),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ]
    )
)
story += [
    t_streams,
    sp(14),
    Paragraph(
        "<b>Technique Math</b> is the most complex stream — it has 4 specialty options "
        "(Civil, Mechanical, Electrical, Process Engineering). Each specialty gets a "
        "distinct system prompt and coefficient table, handled automatically by the "
        "Tutor Agent based on the student's profile.",
        body,
    ),
    PageBreak(),
]

# ── SLIDE 4 — SYSTEM ARCHITECTURE ────────────────────────────────────────────
story += [
    Paragraph("TECHNICAL DESIGN", slide_label),
    Paragraph("System Architecture", section_head),
    rule(),
    sp(8),
    ArchDiagram(PM, 310),
    PageBreak(),
]

# ── SLIDE 5 — TECH STACK ─────────────────────────────────────────────────────
story += [
    Paragraph("ENGINEERING", slide_label),
    Paragraph("Technology Stack", section_head),
    rule(),
    sp(6),
]

stack = [
    ["Layer", "Technology"],
    ["Frontend", "React 18 + TypeScript + Vite"],
    ["Styling", "Custom CSS"],
    ["Backend", "FastAPI (Python 3.11)"],
    ["AI / LLM", "OpenAI GPT-4o"],
    ["Embeddings", "OpenAI text-embedding-3-small"],
    ["Database", "SQLite → PostgreSQL (prod)"],
    ["OCR", "Mathpix / Google Vision / Tesseract"],
]
t_stack = Table(stack, colWidths=[4.2 * cm, 10.6 * cm])
t_stack.setStyle(
    TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), BLACK),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, BG]),
            ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("TEXTCOLOR", (1, 1), (-1, -1), MID),
            ("TEXTCOLOR", (0, 1), (0, -1), DARK),
        ]
    )
)
story += [t_stack, PageBreak()]

# ── SLIDE 6 — TUTOR AGENT + 5 MODES ──────────────────────────────────────────
# Arabic labels rendered as English transliterations to avoid font issues with Arabic in reportlab
modes_data = [
    [P("Mode", cell_hdr), P("Mode ID", cell_hdr), P("Behavior", cell_hdr)],
    [
        P("Orientation"),
        P("general", cell_acc),
        P(
            "Study planning & stream priorities — not concept teaching or exercise solving",
            cell_mid,
        ),
    ],
    [
        P("Exercises"),
        P("exercise_help", cell_acc),
        P(
            "Socratic method — guides with questions first, reveals full answer only after student attempt",
            cell_mid,
        ),
    ],
    [
        P("Concepts"),
        P("concept_explanation", cell_acc),
        P(
            "Structured: Definition → Theorem → Intuition → Formula → Examples → Misconceptions",
            cell_mid,
        ),
    ],
    [
        P("Exam Prep"),
        P("exam_prep", cell_acc),
        P(
            "Focuses on past Bac exams (2015–2024), recurring question patterns and time management",
            cell_mid,
        ),
    ],
    [
        P("Review"),
        P("solution_review", cell_acc),
        P(
            "Student submits solution → agent checks each step vs. official Barème and gives score",
            cell_mid,
        ),
    ],
]
t_modes = Table(modes_data, colWidths=[2.8 * cm, 3.6 * cm, 8.4 * cm])
t_modes.setStyle(
    TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), BLACK),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, BG]),
            ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]
    )
)
story += [
    Paragraph("AI CORE", slide_label),
    Paragraph("The Tutor Agent & 5 Chat Modes", section_head),
    rule(),
    sp(6),
    t_modes,
    PageBreak(),
]

# ── SLIDE 7 — RAG PIPELINE ────────────────────────────────────────────────────
story += [
    Paragraph("DATA & RETRIEVAL", slide_label),
    Paragraph("RAG Pipeline", section_head),
    rule(),
    sp(6),
]

rag_steps = [
    [
        "①",
        "PDF Ingestion",
        "Past Bac exams (2015–2024) loaded via ocr_engine.py\nMathpix / Google Vision / Tesseract fallback",
    ],
    [
        "②",
        "Text Chunking",
        "4 strategies via LangChain RecursiveCharacterTextSplitter:\nlesson (1000 chars), exercise (1500), solution (800), general",
    ],
    [
        "③",
        "Embedding",
        "Each chunk vectorized with OpenAI text-embedding-3-small\nStored as float array in data/vector_store/embeddings.npy",
    ],
    [
        "④",
        "Vector Storage",
        "chunks.json (metadata) + embeddings.npy (numpy flat-file)\nFiltered by stream_code and subject_code at query time",
    ],
    [
        "⑤",
        "Query Retrieval",
        "Student message embedded → cosine similarity → top-k chunks\nReturned via POST /search-context endpoint",
    ],
    [
        "⑥",
        "Prompt Injection",
        "Top chunks appended to system prompt as context\nAgent cites exam year and subject in answer",
    ],
]
t_rag = Table(rag_steps, colWidths=[0.8 * cm, 3.8 * cm, 10.2 * cm])
t_rag.setStyle(
    TableStyle(
        [
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("TEXTCOLOR", (0, 0), (0, -1), ACCENT),
            ("FONTNAME", (1, 0), (1, -1), "Helvetica-Bold"),
            ("TEXTCOLOR", (1, 0), (1, -1), BLACK),
            ("FONTNAME", (2, 0), (2, -1), "Helvetica"),
            ("TEXTCOLOR", (2, 0), (2, -1), MID),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [WHITE, BG]),
            ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]
    )
)
story += [t_rag, PageBreak()]

# ── SLIDE 8 — UX FLOW ────────────────────────────────────────────────────────
story += [
    Paragraph("USER EXPERIENCE", slide_label),
    Paragraph("Student Journey", section_head),
    rule(),
    sp(8),
    UXFlowDiagram(PM, 116),
    sp(14),
]

ux_detail = [
    [P("Step", cell_hdr), P("Location", cell_hdr), P("What Happens", cell_hdr)],
    [
        P("1 — Onboarding"),
        P("/onboarding", cell_acc),
        P(
            "Pick stream (e.g. Mathematiques) + specialty if applicable. Saved in Zustand store.",
            cell_mid,
        ),
    ],
    [
        P("2 — Dashboard"),
        P("/dashboard", cell_acc),
        P(
            "5 mode chips + input box. Student selects mode, types query, presses send arrow.",
            cell_mid,
        ),
    ],
    [
        P("3 — Chat state"),
        P("/dashboard", cell_acc),
        P(
            "Page transitions in-place (no URL change). Messages rendered with Markdown + KaTeX.",
            cell_mid,
        ),
    ],
    [
        P("4 — History"),
        P("Right sidebar", cell_acc),
        P(
            "Collapsible panel lists all past sessions. Click any session to fully restore it.",
            cell_mid,
        ),
    ],
    [
        P("5 — New Chat"),
        P("Header button", cell_acc),
        P(
            "RotateCcw icon resets state and session ID. Returns to idle hero screen.",
            cell_mid,
        ),
    ],
]
t_ux = Table(ux_detail, colWidths=[3.4 * cm, 2.8 * cm, 8.6 * cm])
t_ux.setStyle(
    TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), BLACK),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, BG]),
            ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]
    )
)
story += [t_ux, PageBreak()]

# ── SLIDE 9 — DATA MODELS ────────────────────────────────────────────────────
story += [
    Paragraph("DATABASE", slide_label),
    Paragraph("Data Models", section_head),
    rule(),
    sp(6),
]

db_data = [
    ["Table", "Key Fields", "Relationships"],
    [
        "streams",
        "id, code, name, name_ar, has_options",
        "→ coefficients (1:N)\n→ users (1:N)",
    ],
    ["subjects", "id, code, name, name_ar, category", "→ coefficients (1:N)"],
    [
        "coefficients",
        "stream_id, subject_id, coefficient\nspecialty_option, is_specialty",
        "← streams\n← subjects",
    ],
    [
        "users",
        "id, email, full_name, stream_id\nspecialty_option, is_admin",
        "← streams",
    ],
]
t_db = Table(db_data, colWidths=[3.2 * cm, 6.0 * cm, 5.6 * cm])
t_db.setStyle(
    TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), BLACK),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, BG]),
            ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("TEXTCOLOR", (0, 1), (0, -1), ACCENT),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]
    )
)
story += [
    t_db,
    sp(16),
    Paragraph(
        "The <b>Coefficient Engine</b> uses these tables to compute a student's weighted Bac "
        "average. All stream/subject/coefficient data is seeded via <b>init_db.py</b> at startup. "
        "The endpoint <b>POST /calculate-average</b> accepts marks and returns the weighted "
        "average with official mention classification (Passable / Assez Bien / Bien / Très Bien).",
        body,
    ),
    PageBreak(),
]

# ── SLIDE 10 — API ENDPOINTS ─────────────────────────────────────────────────
story += [
    Paragraph("API REFERENCE", slide_label),
    Paragraph("Key API Endpoints", section_head),
    rule(),
    sp(6),
]

api_data = [
    [P("Method", cell_hdr), P("Endpoint", cell_hdr), P("Description", cell_hdr)],
    [
        P("GET", cell_green),
        P("/streams", cell_acc),
        P("List all 7 streams with Arabic names and has_options flag", cell_mid),
    ],
    [
        P("GET", cell_green),
        P("/streams/{id}", cell_acc),
        P("Stream detail with full coefficient table", cell_mid),
    ],
    [
        P("GET", cell_green),
        P("/streams/{id}/specialties", cell_acc),
        P("Technique Math sub-options (Civil / Meca / Elec / Proc)", cell_mid),
    ],
    [
        P("POST", cell_green),
        P("/calculate-average", cell_acc),
        P("Compute weighted Bac average from subject marks dict", cell_mid),
    ],
    [
        P("POST", cell_green),
        P("/chat", cell_acc),
        P("Send message to Tutor Agent — requires mode + stream context", cell_mid),
    ],
    [
        P("POST", cell_green),
        P("/search-context", cell_acc),
        P("RAG: embed query → cosine search → return top-k exam chunks", cell_mid),
    ],
    [
        P("GET", cell_green),
        P("/subjects", cell_acc),
        P("List all subjects with optional category filter", cell_mid),
    ],
]
t_api = Table(api_data, colWidths=[1.8 * cm, 5.5 * cm, 7.5 * cm])
t_api.setStyle(
    TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), BLACK),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, BG]),
            ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]
    )
)
story += [t_api, PageBreak()]

# ── SLIDE 11 — ROADMAP ───────────────────────────────────────────────────────
story += [
    Paragraph("ROADMAP", slide_label),
    Paragraph("Implementation Phases", section_head),
    rule(),
    sp(8),
]

phases = [
    (
        "Phase 1",
        "Data & Foundation",
        [
            "Build the coefficient engine for all 7 streams",
            "Seed SQLite DB with streams, subjects, coefficients",
            "Build OCR engine to convert PDFs to LaTeX/Arabic",
            "Collect and process exam JSON files 2015–2023",
        ],
        GREEN,
    ),
    (
        "Phase 2",
        "MVP AI + Frontend",
        [
            "Build RAG pipeline with embeddings and retrieval",
            "Build Tutor Agent with 5 specialized chat modes",
            "Build React SPA with onboarding and chat flows",
            "Add inline chat and collapsible history sidebar",
        ],
        ACCENT,
    ),
    (
        "Phase 3",
        "Advanced Features",
        [
            "Add image input to process photos of exercises",
            "Build timed Mock Exam mode with auto-scoring",
            "Add voice explanation support for oral subjects",
            "Migrate to Supabase pgvector for production scale",
        ],
        ORANGE,
    ),
]

phase_cells = []
for title, name, items, col in phases:
    inner = [
        [
            Paragraph(
                title,
                S(
                    "pt",
                    fontSize=8,
                    fontName="Helvetica-Bold",
                    textColor=WHITE,
                    alignment=TA_CENTER,
                    leading=11,
                ),
            )
        ],
        [
            Paragraph(
                name,
                S(
                    "pn",
                    fontSize=10,
                    fontName="Helvetica-Bold",
                    textColor=BLACK,
                    leading=14,
                ),
            )
        ],
        *[[bullet(it)] for it in items],
    ]
    t = Table(inner, colWidths=[PM / 3 - 10])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, 0), col),
                ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
                ("TOPPADDING", (0, 0), (0, 0), 6),
                ("BOTTOMPADDING", (0, 0), (0, 0), 6),
                ("TOPPADDING", (0, 1), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    phase_cells.append(t)

t_phases = Table([phase_cells], colWidths=[PM / 3 - 4] * 3, hAlign="CENTER")
t_phases.setStyle(
    TableStyle(
        [
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]
    )
)
story += [t_phases, PageBreak()]

# ── SLIDE 12 — CLOSING ───────────────────────────────────────────────────────
story += [sp(60), ClosingPage()]

# ── Build PDF ─────────────────────────────────────────────────────────────────
doc.build(story)
print("PDF created:", out)
