"""
Coefficient Engine API - Task 2.1
Implements the Bac average calculator logic with proper validation and edge cases.
"""

from typing import List, Optional

from database import get_db
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from models import Coefficient, Stream, Subject, User
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session

app = FastAPI(
    title="Algerian Baccalaureat AI Backend",
    description="AI-powered tutoring system for Algerian Baccalaureat",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# SCHEMAS
# ============================================================================


class StreamBase(BaseModel):
    code: str
    name: str
    name_ar: Optional[str] = None
    has_options: bool = False
    description: Optional[str] = None


class StreamResponse(StreamBase):
    id: int

    class Config:
        from_attributes = True


class SubjectBase(BaseModel):
    code: str
    name: str
    name_ar: Optional[str] = None
    category: str


class SubjectResponse(SubjectBase):
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


class SpecialtyOption(BaseModel):
    """Specialty options for Technique Math stream."""

    code: str
    name: str
    name_ar: str


TECH_MATH_OPTIONS = [
    SpecialtyOption(code="CIVIL", name="Génie Civil", name_ar="هندسة مدنية"),
    SpecialtyOption(code="MECA", name="Génie Mécanique", name_ar="هندسة ميكانيكية"),
    SpecialtyOption(code="ELEC", name="Génie Électrique", name_ar="هندسة كهربائية"),
    SpecialtyOption(code="PROC", name="Génie des Procédés", name_ar="هندسة الطرائق"),
]


class BacAverageRequest(BaseModel):
    """Request model for calculating Bac average."""

    stream_id: int = Field(..., description="ID of the stream")
    marks: dict = Field(..., description="Dictionary of {subject_code: mark}")
    specialty_option: Optional[str] = Field(
        None, description="Specialty option for Technique Math"
    )
    sport_exempt: bool = Field(
        False, description="Whether student is exempt from Sport"
    )

    @validator("marks")
    def validate_marks(cls, v):
        """Validate that marks are between 0 and 20."""
        for subject_code, mark in v.items():
            if not isinstance(mark, (int, float)):
                raise ValueError(f"Mark for {subject_code} must be a number")
            if mark < 0 or mark > 20:
                raise ValueError(f"Mark for {subject_code} must be between 0 and 20")
        return v


class SubjectResult(BaseModel):
    """Individual subject result."""

    subject_code: str
    subject_name: str
    mark: float
    coefficient: int
    points: float
    is_specialty: bool = False


class BacAverageResponse(BaseModel):
    """Response model for Bac average calculation."""

    stream_code: str
    stream_name: str
    average: float = Field(..., ge=0, le=20)
    total_points: float
    total_coefficients: int
    mention: str
    passed: bool
    subject_results: List[SubjectResult]
    specialty_option: Optional[str] = None
    warnings: List[str] = []


class StreamWithOptions(StreamDetailResponse):
    """Stream response with specialty options if applicable."""

    specialty_options: Optional[List[SpecialtyOption]] = None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def get_mention(average: float) -> str:
    """Determine the mention based on average."""
    if average >= 16:
        return "Très Bien"
    elif average >= 14:
        return "Bien"
    elif average >= 12:
        return "Assez Bien"
    elif average >= 10:
        return "Passable"
    else:
        return "Non Admis"


def calculate_subject_points(mark: float, coefficient: int) -> float:
    """Calculate points for a subject."""
    return mark * coefficient


def validate_marks_against_stream(
    marks: dict, coefficients: List[Coefficient], stream_code: str
) -> tuple[List[str], List[SubjectResult]]:
    """
    Validate marks and create subject results.
    Returns warnings and validated subject results.
    """
    warnings = []
    subject_results = []

    # Check for missing subjects
    required_subjects = {coef.subject.code for coef in coefficients}
    provided_subjects = set(marks.keys())

    missing = required_subjects - provided_subjects
    if missing:
        warnings.append(f"Missing marks for subjects: {', '.join(missing)}")

    # Check for extra subjects
    extra = provided_subjects - required_subjects
    if extra:
        warnings.append(f"Extra subjects not in stream: {', '.join(extra)}")

    # Create subject results
    for coef in coefficients:
        subject_code = coef.subject.code
        mark = marks.get(subject_code, 0.0)
        points = calculate_subject_points(mark, coef.coefficient)

        subject_results.append(
            SubjectResult(
                subject_code=subject_code,
                subject_name=coef.subject.name,
                mark=mark,
                coefficient=coef.coefficient,
                points=points,
                is_specialty=coef.is_specialty,
            )
        )

    return warnings, subject_results


# ============================================================================
# API ENDPOINTS
# ============================================================================


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to Algerian Baccalaureat AI Backend",
        "version": "0.1.0",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check(db: Session = Depends(get_db)):
    """Health check endpoint."""
    try:
        # Check database connection
        from sqlalchemy import text

        db.execute(text("SELECT 1"))
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database error: {str(e)}")


@app.get("/streams", response_model=List[StreamResponse])
async def get_streams(db: Session = Depends(get_db)):
    """
    Get all available streams (Filières).

    Returns the 7 official Algerian Bac streams with their basic information.
    """
    streams = db.query(Stream).all()
    return streams


@app.get("/streams/{stream_id}", response_model=StreamWithOptions)
async def get_stream(stream_id: int, db: Session = Depends(get_db)):
    """
    Get detailed information about a specific stream including coefficients.

    For Technique Math stream, also returns available specialty options.
    """
    stream = db.query(Stream).filter(Stream.id == stream_id).first()
    if not stream:
        raise HTTPException(status_code=404, detail="Stream not found")

    # Build response
    response_data = {
        "id": stream.id,
        "code": stream.code,
        "name": stream.name,
        "name_ar": stream.name_ar,
        "has_options": stream.has_options,
        "description": stream.description,
        "coefficients": stream.coefficients,
    }

    # Add specialty options for Technique Math
    if stream.code == "TECH_MATH":
        response_data["specialty_options"] = TECH_MATH_OPTIONS

    return response_data


@app.get("/streams/{stream_id}/specialties", response_model=List[SpecialtyOption])
async def get_stream_specialties(stream_id: int, db: Session = Depends(get_db)):
    """
    Get specialty options for a stream (mainly for Technique Math).
    """
    stream = db.query(Stream).filter(Stream.id == stream_id).first()
    if not stream:
        raise HTTPException(status_code=404, detail="Stream not found")

    if stream.code == "TECH_MATH":
        return TECH_MATH_OPTIONS

    return []


@app.get("/subjects", response_model=List[SubjectResponse])
async def get_subjects(category: Optional[str] = None, db: Session = Depends(get_db)):
    """
    Get all subjects.

    Optional query parameter:
    - category: Filter by category (scientific, literary, technical, language, etc.)
    """
    query = db.query(Subject)
    if category:
        query = query.filter(Subject.category == category)
    return query.all()


@app.get("/coefficients", response_model=List[CoefficientResponse])
async def get_coefficients(
    stream_id: Optional[int] = None,
    subject_id: Optional[int] = None,
    db: Session = Depends(get_db),
):
    """
    Get coefficients, optionally filtered by stream or subject.

    Query parameters:
    - stream_id: Filter by stream
    - subject_id: Filter by subject
    """
    query = db.query(Coefficient)
    if stream_id:
        query = query.filter(Coefficient.stream_id == stream_id)
    if subject_id:
        query = query.filter(Coefficient.subject_id == subject_id)
    return query.all()


@app.post("/calculate-average", response_model=BacAverageResponse)
async def calculate_average(request: BacAverageRequest, db: Session = Depends(get_db)):
    """
    Calculate Bac average (Moyenne) based on marks and coefficients.

    This is the core calculator feature for students to validate their Bac average.

    Example request:
    ```json
    {
        "stream_id": 2,
        "marks": {
            "MATH": 15.5,
            "PHYSICS": 14.0,
            "SCIENCES": 13.5,
            "ARABIC": 12.0,
            "FRENCH": 11.5,
            "ENGLISH": 13.0
        },
        "specialty_option": null,
        "sport_exempt": false
    }
    ```

    Returns:
    - average: Calculated average (0-20)
    - total_points: Sum of (mark × coefficient)
    - total_coefficients: Sum of all coefficients
    - mention: Academic distinction (Très Bien, Bien, Assez Bien, Passable, Non Admis)
    - passed: Whether average >= 10
    - subject_results: Detailed breakdown per subject
    """
    # Get stream
    stream = db.query(Stream).filter(Stream.id == request.stream_id).first()
    if not stream:
        raise HTTPException(status_code=404, detail="Stream not found")

    # Get coefficients for stream
    coefficients_query = db.query(Coefficient).filter(
        Coefficient.stream_id == request.stream_id
    )

    # Handle specialty options for Technique Math
    if stream.code == "TECH_MATH" and request.specialty_option:
        # Filter by specialty option if provided
        coefficients_query = coefficients_query.filter(
            (Coefficient.specialty_option == request.specialty_option)
            | (Coefficient.specialty_option.is_(None))
        )

    coefficients = coefficients_query.all()

    if not coefficients:
        raise HTTPException(
            status_code=400, detail=f"No coefficients found for stream {stream.name}"
        )

    # Handle Sport exemption
    effective_coefficients = []
    for coef in coefficients:
        if coef.subject.code == "SPORT" and request.sport_exempt:
            # Sport is typically exempt for certain students
            # In this case, we might still include it with a note
            continue
        effective_coefficients.append(coef)

    # Validate marks and create subject results
    warnings, subject_results = validate_marks_against_stream(
        request.marks, effective_coefficients, stream.code
    )

    # Calculate totals
    total_points = sum(r.points for r in subject_results)
    total_coefficients = sum(r.coefficient for r in subject_results)

    if total_coefficients == 0:
        raise HTTPException(
            status_code=400, detail="No valid coefficients found for calculation"
        )

    average = total_points / total_coefficients
    mention = get_mention(average)

    return BacAverageResponse(
        stream_code=stream.code,
        stream_name=stream.name,
        average=round(average, 2),
        total_points=round(total_points, 2),
        total_coefficients=total_coefficients,
        mention=mention,
        passed=average >= 10,
        subject_results=subject_results,
        specialty_option=request.specialty_option,
        warnings=warnings,
    )


@app.get("/streams/{stream_id}/subjects-and-coefficients")
async def get_stream_subjects_with_coefficients(
    stream_id: int, db: Session = Depends(get_db)
):
    """
    Get all subjects with their coefficients for a specific stream.

    Useful for populating the calculator form UI.
    """
    stream = db.query(Stream).filter(Stream.id == stream_id).first()
    if not stream:
        raise HTTPException(status_code=404, detail="Stream not found")

    coefficients = (
        db.query(Coefficient).filter(Coefficient.stream_id == stream_id).all()
    )

    return {
        "stream": {
            "id": stream.id,
            "code": stream.code,
            "name": stream.name,
            "name_ar": stream.name_ar,
        },
        "subjects": [
            {
                "subject_id": coef.subject_id,
                "subject_code": coef.subject.code,
                "subject_name": coef.subject.name,
                "subject_name_ar": coef.subject.name_ar,
                "coefficient": coef.coefficient,
                "is_specialty": coef.is_specialty,
                "category": coef.subject.category,
            }
            for coef in coefficients
        ],
        "total_coefficients": sum(c.coefficient for c in coefficients),
    }


@app.post("/validate-marks")
async def validate_marks(request: BacAverageRequest, db: Session = Depends(get_db)):
    """
    Validate marks without calculating average.

    Returns validation errors and warnings.
    """
    stream = db.query(Stream).filter(Stream.id == request.stream_id).first()
    if not stream:
        raise HTTPException(status_code=404, detail="Stream not found")

    coefficients = (
        db.query(Coefficient).filter(Coefficient.stream_id == request.stream_id).all()
    )

    errors = []
    warnings = []

    # Check for missing subjects
    required_subjects = {coef.subject.code for coef in coefficients}
    provided_subjects = set(request.marks.keys())

    missing = required_subjects - provided_subjects
    if missing:
        errors.append(f"Missing marks for: {', '.join(missing)}")

    # Validate mark ranges
    for subject_code, mark in request.marks.items():
        if mark < 0 or mark > 20:
            errors.append(f"{subject_code}: Mark must be between 0 and 20")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "stream": stream.name,
        "subjects_count": len(request.marks),
        "expected_subjects_count": len(required_subjects),
    }


# ============================================================================
# RAG VECTOR SEARCH ENDPOINTS (Task 2.2)
# ============================================================================

from typing import List as TypingList

from rag_pipeline import RAGPipeline, SearchResult

# Initialize RAG pipeline (singleton)
rag_pipeline = RAGPipeline(mock_embeddings=True)


class SearchRequest(BaseModel):
    """Request model for vector search."""

    query: str = Field(..., description="Search query text")
    top_k: int = Field(5, ge=1, le=20, description="Number of results to return")
    stream_code: Optional[str] = Field(None, description="Filter by stream code")
    subject_code: Optional[str] = Field(None, description="Filter by subject code")
    doc_type: Optional[str] = Field(
        None, description="Filter by document type (lesson, exercise, solution)"
    )
    min_score: float = Field(
        0.0, ge=0.0, le=1.0, description="Minimum similarity score"
    )


class SearchResultResponse(BaseModel):
    """Response model for a single search result."""

    chunk_id: str
    content: str
    source: str
    doc_type: str
    stream_code: str
    subject_code: str
    year: Optional[int] = None
    score: float
    rank: int


class SearchResponse(BaseModel):
    """Response model for vector search."""

    query: str
    results: List[SearchResultResponse]
    total_results: int
    filters_applied: dict


class ContextRequest(BaseModel):
    """Request model for getting context for LLM."""

    query: str = Field(..., description="User query")
    max_tokens: int = Field(2000, ge=100, le=4000, description="Maximum context length")
    stream_code: Optional[str] = Field(None, description="Filter by stream")
    subject_code: Optional[str] = Field(None, description="Filter by subject")


@app.post("/search-context", response_model=SearchResponse)
async def search_context(request: SearchRequest):
    """
    Search for relevant context in the curriculum database.

    This endpoint uses vector similarity search to find the most relevant
    content for a given query. It's the core of the RAG (Retrieval-Augmented
    Generation) pipeline.

    The search:
    1. Embeds the query using OpenAI embeddings
    2. Finds similar chunks in the vector store
    3. Returns the most relevant content with similarity scores

    Example request:
    ```json
    {
        "query": "How to calculate the module of a complex number?",
        "top_k": 5,
        "stream_code": "MATH",
        "subject_code": "MATH",
        "doc_type": "lesson"
    }
    ```

    Returns:
    - Relevant chunks ordered by similarity score
    - Source information for each chunk
    - Document type (lesson, exercise, solution)
    """
    try:
        results = rag_pipeline.search(
            query=request.query,
            top_k=request.top_k,
            stream_code=request.stream_code,
            subject_code=request.subject_code,
            doc_type=request.doc_type,
            min_score=request.min_score,
        )

        # Convert to response format
        response_results = [
            SearchResultResponse(
                chunk_id=r.chunk.chunk_id,
                content=r.chunk.content,
                source=r.chunk.source,
                doc_type=r.chunk.doc_type,
                stream_code=r.chunk.stream_code,
                subject_code=r.chunk.subject_code,
                year=r.chunk.year,
                score=r.score,
                rank=r.rank,
            )
            for r in results
        ]

        return SearchResponse(
            query=request.query,
            results=response_results,
            total_results=len(response_results),
            filters_applied={
                "stream_code": request.stream_code,
                "subject_code": request.subject_code,
                "doc_type": request.doc_type,
                "min_score": request.min_score,
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@app.post("/get-context-for-llm")
async def get_context_for_llm(request: ContextRequest):
    """
    Get formatted context for LLM prompt.

    This endpoint returns context formatted for direct insertion into
    an LLM prompt. It handles:
    - Token limit management
    - Context formatting with source attribution
    - Automatic chunk selection based on relevance

    Use this endpoint when building chat completions with the LLM.

    Example request:
    ```json
    {
        "query": "Explain complex numbers",
        "max_tokens": 1500,
        "stream_code": "MATH"
    }
    ```

    Returns:
    - Formatted context string ready for LLM prompt
    - Number of chunks included
    - Total character count
    """
    try:
        context = rag_pipeline.get_context_for_query(
            query=request.query,
            max_tokens=request.max_tokens,
            stream_code=request.stream_code,
            subject_code=request.subject_code,
        )

        # Also get the raw results for metadata
        results = rag_pipeline.search(
            query=request.query,
            top_k=10,
            stream_code=request.stream_code,
            subject_code=request.subject_code,
        )

        return {
            "context": context,
            "chunks_used": len(results),
            "character_count": len(context),
            "estimated_tokens": len(context) // 4,  # Rough estimate
            "query": request.query,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Context retrieval error: {str(e)}"
        )


@app.get("/rag-stats")
async def get_rag_stats():
    """
    Get statistics about the RAG vector store.

    Returns information about:
    - Total indexed chunks
    - Available streams and subjects
    - Document types distribution
    """
    try:
        stats = rag_pipeline.vector_store.get_stats()
        return {
            "status": "active",
            "embedding_model": rag_pipeline.embedding_service.model,
            "mock_mode": rag_pipeline.embedding_service.mock_mode,
            **stats,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats error: {str(e)}")


# ============================================================================
# CHAT COMPLETION API - Task 2.3
# ============================================================================

import uuid

from chat_service import ChatContext, ChatMessage, ChatMode, RAGChatService
from pydantic import Field

# Initialize chat service (singleton)
chat_service = RAGChatService(rag_pipeline=rag_pipeline, mock_mode=True)


class ChatRequest(BaseModel):
    """Request model for chat completion."""

    message: str = Field(..., description="User message")
    stream_code: Optional[str] = Field(None, description="Student's stream code")
    subject_code: Optional[str] = Field(None, description="Subject being discussed")
    specialty_option: Optional[str] = Field(
        None, description="Specialty option for Technique Math"
    )
    session_id: Optional[str] = Field(None, description="Conversation session ID")
    mode: str = Field(
        "general",
        description="Chat mode: general, exercise_help, exam_prep, concept_explanation, solution_review",
    )
    use_rag: bool = Field(True, description="Whether to use RAG context")
    max_tokens: int = Field(
        2000, ge=100, le=4000, description="Maximum tokens in response"
    )
    temperature: float = Field(
        0.7,
        ge=0.0,
        le=2.0,
        description="Response creativity (0=deterministic, 2=very creative)",
    )


class ChatResponseChunk(BaseModel):
    """Response chunk for streaming."""

    content: str
    is_complete: bool
    session_id: str
    context_used: bool


@app.post("/chat")
async def chat_completion(request: ChatRequest):
    """
    Chat completion endpoint with streaming support.

    This is the main "Tutor" endpoint that provides AI assistance to students.
    It features:
    - Streaming responses for real-time feedback
    - Context-aware system prompts based on stream/subject
    - RAG integration for curriculum-grounded answers
    - Conversation history management

    Example request:
    ```json
    {
        "message": "Explique-moi les nombres complexes",
        "stream_code": "MATH",
        "subject_code": "MATH",
        "mode": "concept_explanation",
        "use_rag": true
    }
    ```

    For non-streaming response, returns complete response.
    For streaming, use /chat/stream endpoint.

    Returns:
    - Complete AI response
    - Session ID for conversation continuity
    - Context usage information
    """
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())

        # Parse mode
        try:
            mode = ChatMode(request.mode)
        except ValueError:
            mode = ChatMode.GENERAL

        # Build context
        context = ChatContext(
            stream_code=request.stream_code,
            subject_code=request.subject_code,
            specialty_option=request.specialty_option,
            mode=mode,
        )

        # Generate response (non-streaming for this endpoint)
        full_response = ""
        context_used = False

        async for chunk in chat_service.chat(
            message=request.message,
            context=context,
            session_id=session_id,
            use_rag=request.use_rag,
            max_context_tokens=min(request.max_tokens, 1500),
        ):
            full_response += chunk.content
            context_used = chunk.context_used
            if chunk.is_complete:
                break

        return {
            "response": full_response,
            "session_id": session_id,
            "context_used": context_used,
            "mode": mode.value,
            "stream_code": request.stream_code,
            "subject_code": request.subject_code,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@app.post("/chat/stream")
async def chat_completion_stream(request: ChatRequest):
    """
    Streaming chat completion endpoint (Server-Sent Events).

    This endpoint streams the AI response in real-time using Server-Sent Events.
    Ideal for interactive chat interfaces.

    The stream sends JSON chunks with:
    - content: Text chunk
    - is_complete: Whether this is the final chunk
    - session_id: Session identifier
    - context_used: Whether RAG context was used

    Example JavaScript usage:
    ```javascript
    const eventSource = new EventSource('/chat/stream');
    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.is_complete) {
            eventSource.close();
        } else {
            appendToChat(data.content);
        }
    };
    ```

    Returns:
    - Streaming text/event-stream response
    """
    import json

    from fastapi.responses import StreamingResponse

    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())

        # Parse mode
        try:
            mode = ChatMode(request.mode)
        except ValueError:
            mode = ChatMode.GENERAL

        # Build context
        context = ChatContext(
            stream_code=request.stream_code,
            subject_code=request.subject_code,
            specialty_option=request.specialty_option,
            mode=mode,
        )

        async def generate_stream():
            """Generator for streaming response."""
            async for chunk in chat_service.chat(
                message=request.message,
                context=context,
                session_id=session_id,
                use_rag=request.use_rag,
                max_context_tokens=min(request.max_tokens, 1500),
            ):
                response_chunk = {
                    "content": chunk.content,
                    "is_complete": chunk.is_complete,
                    "session_id": session_id,
                    "context_used": chunk.context_used,
                }
                yield f"data: {json.dumps(response_chunk)}\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Session-ID": session_id,
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stream error: {str(e)}")


@app.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str):
    """
    Get conversation history for a session.

    Returns the full conversation history including both user and assistant messages.
    """
    try:
        history = chat_service.conversations.get_history(session_id)

        return {
            "session_id": session_id,
            "message_count": len(history),
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                }
                for msg in history
            ],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"History error: {str(e)}")


@app.delete("/chat/history/{session_id}")
async def clear_chat_history(session_id: str):
    """
    Clear conversation history for a session.

    This permanently deletes the conversation history.
    """
    try:
        chat_service.conversations.clear_history(session_id)

        return {
            "session_id": session_id,
            "status": "cleared",
            "message": "Conversation history cleared successfully",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clear history error: {str(e)}")


@app.get("/chat/modes")
async def get_chat_modes():
    """
    Get available chat modes.

    Returns a list of available chat modes with descriptions.
    """
    return {
        "modes": [
            {
                "code": "general",
                "name": "General Assistance",
                "description": "General academic help and questions",
            },
            {
                "code": "exercise_help",
                "name": "Exercise Help",
                "description": "Help solving specific exercises with step-by-step guidance",
            },
            {
                "code": "exam_prep",
                "name": "Exam Preparation",
                "description": "Prepare for Bac exam with exam-style questions",
            },
            {
                "code": "concept_explanation",
                "name": "Concept Explanation",
                "description": "Detailed explanation of theoretical concepts",
            },
            {
                "code": "solution_review",
                "name": "Solution Review",
                "description": "Review your solution attempt with feedback",
            },
        ]
    }


@app.get("/chat/status")
async def get_chat_status():
    """
    Get chat service status.

    Returns information about the chat service configuration.
    """
    return {
        "status": "active",
        "model": (
            chat_service.chat_service.model
            if hasattr(chat_service.chat_service, "model")
            else "unknown"
        ),
        "mock_mode": chat_service.chat_service.mock_mode,
        "rag_enabled": True,
        "streaming_supported": True,
        "conversation_management": True,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
