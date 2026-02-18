"""
Chat Service - Task 2.3
Main "Tutor" endpoint with streaming responses and RAG integration.

Features:
- Streaming chat completions (Server-Sent Events)
- Context-aware system prompts (stream-specific)
- RAG context retrieval integration
- Support for multi-turn conversations
- Mock mode for testing without API keys
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import AsyncGenerator, Dict, List, Optional

from openai import AsyncOpenAI


class ChatMode(Enum):
    """Different chat modes for specialized assistance."""

    GENERAL = "general"
    EXERCISE_HELP = "exercise_help"
    EXAM_PREP = "exam_prep"
    CONCEPT_EXPLANATION = "concept_explanation"
    SOLUTION_REVIEW = "solution_review"


@dataclass
class ChatMessage:
    """A chat message."""

    role: str  # "system", "user", "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict = field(default_factory=dict)


@dataclass
class ChatContext:
    """Context for a chat session."""

    stream_code: Optional[str] = None
    subject_code: Optional[str] = None
    specialty_option: Optional[str] = None
    year: Optional[int] = None
    mode: ChatMode = ChatMode.GENERAL
    session_id: Optional[str] = None


@dataclass
class ChatRequest:
    """Request for chat completion."""

    message: str
    context: ChatContext
    conversation_history: List[ChatMessage] = field(default_factory=list)
    use_rag: bool = True
    max_tokens: int = 2000
    temperature: float = 0.7


@dataclass
class ChatResponse:
    """Response from chat completion."""

    content: str
    is_complete: bool = False
    citations: List[Dict] = field(default_factory=list)
    context_used: bool = False


class SystemPromptBuilder:
    """
    Builds context-aware system prompts for different streams and scenarios.

    The prompt adapts based on:
    - Student's stream (MATH, SCI_EXP, etc.)
    - Subject (Mathematics, Physics, etc.)
    - Specialty option (for Technique Math)
    - Chat mode (exercise help, concept explanation, etc.)
    """

    # Base prompt template
    BASE_PROMPT = """You are an expert Algerian Baccalaureat tutor.

CORE RULES:
1. STRICTLY follow the Algerian Ministry of Education curriculum
2. Answers must be structured: Definition -> Theorem -> Application -> Result
3. Use appropriate terminology for the student's specific stream and subject
4. Be precise and rigorous in mathematical/scientific explanations
5. When providing solutions, follow the official grading scheme (Barème)

LANGUAGE GUIDELINES:
- ALWAYS respond in Arabic (Modern Standard Arabic - الفصحى) regardless of the language the student uses
- ALL explanations, solutions, and feedback must be written in Arabic
- For scientific/mathematical terms, write the Arabic term first, then the French/English equivalent in parentheses: e.g., المشتقة (Dérivée), الأعداد المركبة (Nombres complexes), التكامل (Intégrale)
- LaTeX math expressions remain as-is: $x^2 + y^2 = z^2$
- If the student writes in Darja (Algerian dialect), respond in clear Modern Standard Arabic (MSA - الفصحى)

TEACHING APPROACH:
- Guide students to find answers rather than just giving them
- Ask clarifying questions when needed
- Provide step-by-step explanations
- Include relevant examples from Bac exams when possible
- Reference specific exam years if relevant (e.g., "This appeared in Bac 2023")

{stream_specific}

{subject_specific}

{mode_specific}

{context_section}

Remember: You are preparing this student for the Algerian Baccalaureat. Accuracy and curriculum alignment are critical.
"""

    STREAM_GUIDELINES = {
        "MATH": """
STREAM: MATHEMATIQUES
- Mathematics coefficient: 7 (highest weight)
- Physics coefficient: 6
- Focus on rigorous proofs and demonstrations
- Emphasize algebraic manipulation and calculus
- Complex numbers, limits, derivatives, integrals are core topics
- Expect formal mathematical language and notation
""",
        "SCI_EXP": """
STREAM: SCIENCES EXPERIMENTALES
- Natural Sciences coefficient: 6 (highest weight)
- Physics coefficient: 5
- Math coefficient: 5
- Balance between scientific reasoning and mathematical application
- Diagram interpretation is crucial
- Laboratory concepts and experimental methods
- Connect theory to real-world phenomena
""",
        "TECH_MATH": """
STREAM: TECHNIQUE MATHEMATIQUE
- Mathematics, Physics, and Technology all have coefficient: 6
- {specialty_specific}
- Emphasize practical applications and engineering concepts
- Technical drawing and diagram interpretation
- Focus on problem-solving methodologies
- Connect theory to technical implementations
""",
        "GESTION": """
STREAM: GESTION ET ECONOMIE
- Accounting/Math: high coefficients
- Focus on structured, logical reasoning
- Economic theory and practical applications
- Legal and management concepts
- Numerical analysis and data interpretation
- Real-world business and economic examples
""",
        "LANGUES": """
STREAM: LANGUES ETRANGERES
- Arabic, French, English: coefficient 5 each
- Third language (Spanish/German/Italian): coefficient 4
- Focus on grammar, vocabulary, and translation
- Literary analysis and essay writing
- Cultural and linguistic nuances
- Comparative language studies
""",
        "LETTRES": """
STREAM: LETTRES ET PHILOSOPHIE
- Philosophy coefficient: 6
- Arabic Literature coefficient: 6
- Emphasize abstract reasoning and critical thinking
- Textual analysis and interpretation
- Philosophical argumentation
- Literary criticism and analysis
- Essay structure and development
""",
        "ARTS": """
STREAM: ARTS
- Drawing/Art specialty coefficient: 6
- Focus on creative and artistic expression
- Art history and theory
- Technical artistic skills
- Portfolio development guidance
- Art critique and analysis
""",
    }

    SPECIALTY_GUIDELINES = {
        "CIVIL": """Specialty: GENIE CIVIL
- Focus on structural analysis, mechanics of materials
- Civil engineering principles and standards
- Construction and infrastructure concepts
- Technical drawings and blueprints""",
        "MECA": """Specialty: GENIE MECANIQUE
- Focus on mechanical systems and dynamics
- Thermodynamics and heat transfer
- Mechanical design principles
- Manufacturing processes""",
        "ELEC": """Specialty: GENIE ELECTRIQUE
- Focus on electrical circuits and systems
- Electromagnetism applications
- Electronic components and systems
- Power systems and distribution""",
        "PROC": """Specialty: GENIE DES PROCEDES
- Focus on chemical processes and reactions
- Process engineering and optimization
- Material balances and thermodynamics
- Industrial chemistry concepts""",
    }

    MODE_GUIDELINES = {
        ChatMode.EXERCISE_HELP: """
MODE: EXERCISE HELP
Help the student solve an exercise:
1. Read and understand the problem carefully
2. Identify the key concepts and theorems needed
3. Guide them step-by-step without giving the full answer immediately
4. Ask guiding questions to help them think
5. Verify their understanding at each step
6. Provide the solution only after they've attempted it
7. Explain the grading criteria (Barème) if relevant
""",
        ChatMode.EXAM_PREP: """
MODE: EXAM PREPARATION
Help the student prepare for Bac exam:
1. Focus on exam-style questions and format
2. Reference past Bac exams (2015-2024)
3. Highlight common exam topics and patterns
4. Practice time management strategies
5. Review grading schemes (Barème)
6. Identify weak areas for improvement
7. Provide tips for exam day
""",
        ChatMode.CONCEPT_EXPLANATION: """
MODE: CONCEPT EXPLANATION
Explain a theoretical concept:
1. Start with a clear definition
2. Provide the theorem or principle
3. Give intuitive understanding (why it works)
4. Show mathematical formulation if applicable
5. Provide concrete examples
6. Show applications in Bac context
7. Connect to prerequisite concepts
8. Warn about common misconceptions
""",
        ChatMode.SOLUTION_REVIEW: """
MODE: SOLUTION REVIEW
Review student's solution attempt:
1. Check each step for correctness
2. Identify errors and explain why they're wrong
3. Suggest corrections
4. Check if the method follows official grading scheme
5. Verify final answer
6. Suggest alternative approaches if applicable
7. Calculate approximate score based on Bac standards
""",
        ChatMode.GENERAL: """
MODE: GUIDANCE & ORIENTATION
Help the student plan and organize their studies — NOT for explaining concepts or solving exercises (use the dedicated modes for those):
- Recommend what subjects to prioritize based on their stream and coefficients
- Help build a revision schedule and study plan
- Advise on how to distribute study time across subjects
- Answer questions like "where do I start?", "what topics matter most?"
- Suggest strategies for managing exam stress and time on exam day
- Guide students who feel lost on how to approach Bac preparation overall
""",
    }

    @classmethod
    def build_prompt(cls, context: ChatContext, retrieved_context: str = "") -> str:
        """
        Build a context-aware system prompt.

        Args:
            context: Chat context with stream, subject, mode info
            retrieved_context: Context retrieved from RAG

        Returns:
            Complete system prompt
        """
        # Get stream-specific guidelines
        stream_specific = ""
        if context.stream_code:
            stream_guidelines = cls.STREAM_GUIDELINES.get(
                context.stream_code,
                "Follow standard Baccalaureat curriculum guidelines.",
            )

            # Add specialty info for Technique Math
            if context.stream_code == "TECH_MATH" and context.specialty_option:
                specialty_guidelines = cls.SPECIALTY_GUIDELINES.get(
                    context.specialty_option,
                    "Follow general technical mathematics guidelines.",
                )
                stream_specific = stream_guidelines.format(
                    specialty_specific=specialty_guidelines
                )
            else:
                stream_specific = stream_guidelines

        # Get mode-specific guidelines
        mode_specific = cls.MODE_GUIDELINES.get(
            context.mode, cls.MODE_GUIDELINES[ChatMode.GENERAL]
        )

        # Build subject-specific section
        subject_specific = ""
        if context.subject_code:
            subject_specific = f"""
SUBJECT: {context.subject_code}
- Use subject-specific terminology
- Follow the subject's curriculum and requirements
- Adapt explanation style to subject conventions
"""

        # Build context section from RAG
        context_section = ""
        if retrieved_context:
            context_section = f"""
RELEVANT CURRICULUM CONTEXT:
The following context has been retrieved from the official curriculum:
---
{retrieved_context}
---
Use this context to ground your answer in the official curriculum. Cite sources when applicable.
"""

        # Build final prompt
        prompt = cls.BASE_PROMPT.format(
            stream_specific=stream_specific,
            subject_specific=subject_specific,
            mode_specific=mode_specific,
            context_section=context_section,
        )

        return prompt


class ChatService:
    """
    Main chat service that handles completions with OpenAI.

    Supports:
    - Streaming responses
    - RAG context integration
    - Mock mode for development
    """

    def __init__(self, mock_mode: bool = False):
        """
        Initialize chat service.

        Args:
            mock_mode: If True, use mock responses (no API calls)
        """
        self.mock_mode = mock_mode or not os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

        if not self.mock_mode:
            self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            print("[WARNING] OPENAI_API_KEY not set. Using mock chat mode.")

    async def generate_response(
        self, request: ChatRequest, retrieved_context: str = ""
    ) -> AsyncGenerator[ChatResponse, None]:
        """
        Generate streaming chat response.

        Args:
            request: Chat request with message and context
            retrieved_context: Context from RAG

        Yields:
            ChatResponse chunks
        """
        if self.mock_mode:
            async for chunk in self._generate_mock_response(request):
                yield chunk
            return

        try:
            # Build system prompt
            system_prompt = SystemPromptBuilder.build_prompt(
                context=request.context, retrieved_context=retrieved_context
            )

            # Build messages
            messages = [{"role": "system", "content": system_prompt}]

            # Add conversation history
            for msg in request.conversation_history:
                messages.append({"role": msg.role, "content": msg.content})

            # Add current message
            messages.append({"role": "user", "content": request.message})

            # Stream completion
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=True,
            )

            full_content = ""
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_content += content
                    yield ChatResponse(
                        content=content,
                        is_complete=False,
                        context_used=request.use_rag and bool(retrieved_context),
                    )

            # Final chunk
            yield ChatResponse(
                content="",
                is_complete=True,
                context_used=request.use_rag and bool(retrieved_context),
            )

        except Exception as e:
            yield ChatResponse(content=f"Error: {str(e)}", is_complete=True)

    async def _generate_mock_response(
        self, request: ChatRequest
    ) -> AsyncGenerator[ChatResponse, None]:
        """Generate mock response for testing."""
        # Build a context-aware mock response
        stream_name = request.context.stream_code or "general"
        subject_name = request.context.subject_code or "general"

        mock_response = f"""[وضع الاختبار]
الشعبة: {stream_name}
المادة: {subject_name}
النمط: {request.context.mode.value}

هذا رد تجريبي لأغراض الاختبار. في بيئة الإنتاج، سيتم توليد هذا الرد بواسطة GPT-4o بناءً على:
- شعبتك الدراسية ({stream_name})
- سؤالك: "{request.message[:50]}..."
- السياق المسترجع من قاعدة البيانات (إذا كان مفعّلاً)

المساعد الحقيقي سيوفر:
1. شرحاً متوافقاً مع مناهج وزارة التربية الوطنية
2. إرشاداً خطوة بخطوة
3. مراجع لمحتوى البكالوريا الرسمي
4. صيغة رياضية بتنسيق LaTeX: $x^2 + y^2 = z^2$
5. ردوداً كاملة باللغة العربية الفصحى
"""

        # Stream character by character for realism
        for char in mock_response:
            yield ChatResponse(content=char, is_complete=False)

        yield ChatResponse(content="", is_complete=True)


# ============================================================================
# Conversation Management
# ============================================================================


class ConversationManager:
    """
    Manages conversation history and sessions.

    In production, this would use Redis or a database.
    For now, uses in-memory storage.
    """

    def __init__(self):
        self.sessions: Dict[str, List[ChatMessage]] = {}

    def get_history(self, session_id: str) -> List[ChatMessage]:
        """Get conversation history for a session."""
        return self.sessions.get(session_id, [])

    def add_message(self, session_id: str, message: ChatMessage):
        """Add a message to conversation history."""
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append(message)

        # Keep only last 20 messages to manage context window
        if len(self.sessions[session_id]) > 20:
            self.sessions[session_id] = self.sessions[session_id][-20:]

    def clear_history(self, session_id: str):
        """Clear conversation history."""
        if session_id in self.sessions:
            del self.sessions[session_id]


# ============================================================================
# Integration with RAG
# ============================================================================


class RAGChatService:
    """
    Combined service that integrates RAG with chat completions.
    """

    def __init__(self, rag_pipeline=None, mock_mode: bool = False):
        from rag_pipeline import RAGPipeline

        self.rag = rag_pipeline or RAGPipeline(mock_embeddings=mock_mode)
        self.chat_service = ChatService(mock_mode=mock_mode)
        self.conversations = ConversationManager()

    async def chat(
        self,
        message: str,
        context: ChatContext,
        session_id: Optional[str] = None,
        use_rag: bool = True,
        max_context_tokens: int = 1500,
    ) -> AsyncGenerator[ChatResponse, None]:
        """
        Complete chat flow with RAG integration.

        Args:
            message: User message
            context: Chat context (stream, subject, etc.)
            session_id: Optional session ID for conversation history
            use_rag: Whether to use RAG
            max_context_tokens: Max tokens for context

        Yields:
            ChatResponse chunks
        """
        # Get conversation history
        history = []
        if session_id:
            history = self.conversations.get_history(session_id)

        # Retrieve context from RAG if enabled
        retrieved_context = ""
        if use_rag:
            retrieved_context = self.rag.get_context_for_query(
                query=message,
                max_tokens=max_context_tokens,
                stream_code=context.stream_code,
                subject_code=context.subject_code,
            )

        # Create request
        request = ChatRequest(
            message=message,
            context=context,
            conversation_history=history,
            use_rag=use_rag,
        )

        # Generate response
        full_response = ""
        async for chunk in self.chat_service.generate_response(
            request, retrieved_context
        ):
            full_response += chunk.content
            yield chunk

        # Save to conversation history
        if session_id:
            # Add user message
            self.conversations.add_message(
                session_id, ChatMessage(role="user", content=message)
            )
            # Add assistant response
            self.conversations.add_message(
                session_id, ChatMessage(role="assistant", content=full_response)
            )


# ============================================================================
# Quick Test
# ============================================================================


async def test_chat_service():
    """Test the chat service."""
    print("\n" + "=" * 60)
    print("Testing Chat Service")
    print("=" * 60 + "\n")

    # Create service
    service = RAGChatService(mock_mode=True)

    # Test with Math stream
    context = ChatContext(
        stream_code="MATH", subject_code="MATH", mode=ChatMode.CONCEPT_EXPLANATION
    )

    print("[TEST 1] Concept explanation for Math stream")
    print("-" * 60)

    message = "Explique-moi les nombres complexes"
    print(f"User: {message}\n")
    print("Assistant: ", end="", flush=True)

    full_response = ""
    async for chunk in service.chat(message, context, session_id="test_1"):
        if chunk.content:
            print(chunk.content, end="", flush=True)
            full_response += chunk.content

    print("\n\n" + "=" * 60)
    print(f"Response complete: {len(full_response)} characters")
    print("=" * 60)

    # Test conversation history
    print("\n[TEST 2] Conversation history")
    print("-" * 60)

    history = service.conversations.get_history("test_1")
    print(f"Session has {len(history)} messages:")
    for msg in history:
        print(f"  [{msg.role}]: {msg.content[:50]}...")

    print("\n[OK] Chat service test complete!")


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_chat_service())
