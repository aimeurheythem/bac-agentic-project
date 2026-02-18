# Project Roadmap: Algerian Baccalauréat Feature

This roadmap outlines the complete development process from setup to MVP deployment.

**Core Tech Stack:**
- **Frontend:** React + Vite + Tailwind CSS
- **Backend:** Python + FastAPI + LangChain
- **Database:** Supabase (PostgreSQL + pgvector)
- **AI:** GPT-4o / Gemini 1.5 Pro via API

---

## Phase 1: Project Initialization & Data Layer

### Task 1.1: Environment Setup
**Description:** Initialize the Frontend and Backend repositories.
**Tech Stack:** `git`, `vite`, `fastapi`, `poetry/pip`
- [ ] Initialize `admin-panel/client` using `npm create vite@latest` (React + TS)
- [ ] Initialize `backend/` using `poetry init` (FastAPI)
- [ ] Set up pre-commit hooks (linting/formatting)

### Task 1.2: Database Schema Design (Supabase)
**Description:** Create tables for Users, Streams (Filières), Subjects, and Coefficients.
**Tech Stack:** `Supabase`, `PostgreSQL`
- [ ] Design `streams` table (7 official streams)
- [ ] Design `subjects` table (Math, Physics, etc.)
- [ ] Design `coefficients` table (linking streams-subjects-values)
- [ ] Design `users` table (auth + stream preference)

### Task 1.3: Data Acquisition Pipeline
**Description:** Build scripts to scrape and structure Bac exam data.
**Tech Stack:** `Python`, `BeautifulSoup`, `PyPDF2`
- [ ] Scrape exams from `dzexams.com` or `ency-education` (2015-2024)
- [ ] Organize PDF storage (S3/Supabase Storage)
- [ ] Create JSON metadata schema (Year, Stream, Subject, Topic)

### Task 1.4: OCR Processing Strategy
**Description:** Convert PDF images (especially Math/Physics) to Latex/Text.
**Tech Stack:** `Mathpix API` or `Google Vision AI`
- [ ] Implement OCR script for "Technique Math" diagrams
- [ ] Validate accuracy on complex equations

---

## Phase 2: Core Backend Services

### Task 2.1: Coefficient Engine API
**Description:** The "Calculator" logic. Key feature for student validation.
**Tech Stack:** `FastAPI`, `Pydantic`
- [ ] Endpoint `GET /streams`: List all 7 streams + specialties
- [ ] Endpoint `POST /calculate-average`: Input marks -> Return Average + Mention
- [ ] Unit tests for all edge cases (e.g., Sport exemption)

### Task 2.2: RAG Pipeline Setup (Vector Store)
**Description:** Indexing the curriculum for AI retrieval.
**Tech Stack:** `LangChain`, `Supabase pgvector`, `OpenAI Embeddings`
- [ ] Chunking strategy (splitting lessons vs exercises)
- [ ] Embedding generation script
- [ ] Vector search endpoint `POST /search-context`

### Task 2.3: Chat Completion API
**Description:** The main "Tutor" endpoint using the LLM.
**Tech Stack:** `FastAPI`, `OpenAI API/Anthropic`
- [ ] Implement `POST /chat` with streaming response
- [ ] Integrate System Prompts (Context-Aware for Streams)
- [ ] Connect RAG retrieval to the prompt context

---

## Phase 3: Frontend Development (React)

### Task 3.1: Onboarding & Stream Selection
**Description:** User must select their stream before accessing the app.
**Tech Stack:** `React`, `Zustand` (State Management), `Tailwind CSS`
- [ ] Create "Select Your Stream" Screen (Cards for specific streams)
- [ ] Persist selection in User Profile
- [ ] Handle "Technique Math" specific option selection (Civil/Mech/etc.)

### Task 3.2: Bac Average Simulator (The Hook)
**Description:** Interactive calculator UI.
**Tech Stack:** `React Hook Form`, `Radix UI`
- [ ] Dynamic form based on selected stream (subjects auto-populate)
- [ ] Real-time calculation as user types marks
- [ ] Visual feedback for "Mention" (Admitted/Failed)

### Task 3.3: Intelligent Tutor Interface
**Description:** Chat UI for asking questions.
**Tech Stack:** `React`, `Markdown Renderer` (for Math), `Latex`
- [ ] Chat window with history
- [ ] Markdown/Latex rendering for math formulas ($$x^2$$)
- [ ] "Ask about this exercise" context menu

### Task 3.4: Exam Repository Browser
**Description:** Browse past exams by year/subject.
**Tech Stack:** `React Query`, `PDF Viewer`
- [ ] Filterable list (Year -> Subject -> Session)
- [ ] PDF Viewer component
- [ ] "Solve with AI" button on each exam paper

---

## Phase 4: Refinement & Launch

### Task 4.1: Prompt Tuning & Validation
**Description:** Verify AI understands Algerian context.
**Tech Stack:** `Prompt Engineering`
- [ ] Validate "Philosophy" essay grading logic
- [ ] Verify "Technique Math" terminology accuracy
- [ ] Test bilingual (Darja/French) handling

### Task 4.2: Performance Optimization
**Description:** Ensure fast load times.
**Tech Stack:** `Vite Build Analysis`, `Redis` (Caching)
- [ ] Cache frequent RAG queries (e.g., "Bac 2023 Math Correction")
- [ ] Optimize PDF loading

### Task 4.3: Deployment
**Description:** Deploy to production.
**Tech Stack:** `Vercel` (Frontend), `Railway/Render` (Backend)
- [ ] CI/CD Pipelines
- [ ] Environment variable configuration
- [ ] Domain setup
