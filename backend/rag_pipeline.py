"""
RAG (Retrieval-Augmented Generation) Pipeline
Task 2.2: Vector Store and Context Retrieval

This module provides:
- Text chunking strategies optimized for Bac content
- Embedding generation using OpenAI
- Vector storage and similarity search
- Context retrieval for the AI tutor

Architecture:
1. Documents are split into chunks (lessons vs exercises)
2. Each chunk is embedded using OpenAI embeddings
3. Embeddings stored in vector database (SQLite for local, Supabase pgvector for prod)
4. Query embeddings used for similarity search
5. Top-k relevant chunks retrieved as context
"""

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import (
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


@dataclass
class Chunk:
    """A text chunk with metadata."""

    content: str
    chunk_id: str
    source: str  # File or document source
    doc_type: str  # 'lesson', 'exercise', 'solution', 'correction'
    stream_code: str
    subject_code: str
    year: Optional[int] = None
    page_number: Optional[int] = None
    metadata: Dict = field(default_factory=dict)
    embedding: Optional[List[float]] = None


@dataclass
class SearchResult:
    """Result from vector search."""

    chunk: Chunk
    score: float  # Similarity score
    rank: int


class TextChunker:
    """
    Handles different text chunking strategies for Bac content.

    Strategies:
    1. Lesson chunking: Split by headers, keep context
    2. Exercise chunking: Keep exercises + solutions together
    3. Solution chunking: Split by problem steps
    4. General: Recursive character splitting
    """

    def __init__(self):
        self.splitters = {}
        self._init_splitters()

    def _init_splitters(self):
        """Initialize different text splitters."""
        # For lessons - split by headers but keep reasonable size
        self.splitters["lesson"] = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

        # For exercises - keep exercises intact if possible
        self.splitters["exercise"] = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=100,
            separators=["\nExercice", "\nExercise", "\nالتمرين", "\n\n", "\n"],
            length_function=len,
        )

        # For solutions - smaller chunks for step-by-step
        self.splitters["solution"] = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n## ", "\n### ", "\n\n", "\n", ". "],
            length_function=len,
        )

        # For general text
        self.splitters["general"] = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100, length_function=len
        )

    def chunk_document(
        self,
        content: str,
        source: str,
        doc_type: str,
        stream_code: str,
        subject_code: str,
        year: Optional[int] = None,
        metadata: Dict = None,
    ) -> List[Chunk]:
        """
        Chunk a document based on its type.

        Args:
            content: Full document text
            source: Document source (filename, URL, etc.)
            doc_type: Type of document ('lesson', 'exercise', 'solution', 'general')
            stream_code: Stream code (MATH, SCI_EXP, etc.)
            subject_code: Subject code (MATH, PHYSICS, etc.)
            year: Optional year for exam documents
            metadata: Additional metadata

        Returns:
            List of Chunk objects
        """
        # Get appropriate splitter
        splitter = self.splitters.get(doc_type, self.splitters["general"])

        # Split content
        docs = splitter.create_documents([content])

        chunks = []
        for i, doc in enumerate(docs):
            # Generate unique chunk ID
            chunk_hash = hashlib.md5(
                f"{source}_{doc_type}_{i}_{doc.page_content[:100]}".encode()
            ).hexdigest()[:16]

            chunk = Chunk(
                content=doc.page_content,
                chunk_id=f"chunk_{chunk_hash}",
                source=source,
                doc_type=doc_type,
                stream_code=stream_code,
                subject_code=subject_code,
                year=year,
                page_number=doc.metadata.get("page", None),
                metadata=metadata or {},
            )
            chunks.append(chunk)

        return chunks

    def chunk_lesson(
        self,
        content: str,
        source: str,
        stream_code: str,
        subject_code: str,
        topic: str = None,
    ) -> List[Chunk]:
        """Chunk a lesson document."""
        metadata = {"topic": topic} if topic else {}
        return self.chunk_document(
            content=content,
            source=source,
            doc_type="lesson",
            stream_code=stream_code,
            subject_code=subject_code,
            metadata=metadata,
        )

    def chunk_exam(
        self,
        content: str,
        source: str,
        stream_code: str,
        subject_code: str,
        year: int,
        session: str = "principal",
    ) -> List[Chunk]:
        """Chunk an exam paper."""
        metadata = {"year": year, "session": session, "document_type": "exam"}

        # First try to split by exercises
        chunks = self.chunk_document(
            content=content,
            source=source,
            doc_type="exercise",
            stream_code=stream_code,
            subject_code=subject_code,
            year=year,
            metadata=metadata,
        )

        return chunks

    def chunk_solution(
        self,
        content: str,
        source: str,
        stream_code: str,
        subject_code: str,
        year: int,
        exercise_number: int = None,
    ) -> List[Chunk]:
        """Chunk a solution/correction document."""
        metadata = {
            "year": year,
            "exercise_number": exercise_number,
            "document_type": "solution",
        }

        return self.chunk_document(
            content=content,
            source=source,
            doc_type="solution",
            stream_code=stream_code,
            subject_code=subject_code,
            year=year,
            metadata=metadata,
        )


class EmbeddingService:
    """
    Service for generating embeddings using OpenAI.

    Supports:
    - OpenAI text-embedding-3-small (cheap, fast)
    - OpenAI text-embedding-3-large (better quality)
    - Mock mode for development without API key
    """

    def __init__(self, model: str = "text-embedding-3-small", mock_mode: bool = False):
        """
        Initialize embedding service.

        Args:
            model: OpenAI embedding model name
            mock_mode: If True, use mock embeddings (for development)
        """
        self.model = model
        self.mock_mode = mock_mode or not os.getenv("OPENAI_API_KEY")

        if not self.mock_mode:
            self.embeddings = OpenAIEmbeddings(
                model=model, openai_api_key=os.getenv("OPENAI_API_KEY")
            )
        else:
            print("[WARNING] OPENAI_API_KEY not set. Using mock embeddings.")

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        if self.mock_mode:
            # Generate deterministic mock embedding
            np.random.seed(hash(text) % 2**32)
            return np.random.randn(1536).tolist()

        return self.embeddings.embed_query(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if self.mock_mode:
            return [self.embed_text(text) for text in texts]

        return self.embeddings.embed_documents(texts)

    def embed_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Generate embeddings for all chunks."""
        if not chunks:
            return chunks

        texts = [chunk.content for chunk in chunks]
        embeddings = self.embed_texts(texts)

        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding

        return chunks


class VectorStore:
    """
    Vector storage for document embeddings.

    For local development: Uses SQLite with simple vector operations
    For production: Will migrate to Supabase pgvector
    """

    def __init__(self, storage_path: str = "data/vector_store"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.chunks: Dict[str, Chunk] = {}
        self.embeddings: Dict[str, List[float]] = {}
        self._load_from_disk()

    def _load_from_disk(self):
        """Load existing chunks from disk."""
        chunks_file = self.storage_path / "chunks.json"
        embeddings_file = self.storage_path / "embeddings.npy"

        if chunks_file.exists():
            with open(chunks_file, "r", encoding="utf-8") as f:
                chunks_data = json.load(f)
                for chunk_data in chunks_data:
                    chunk = Chunk(
                        content=chunk_data["content"],
                        chunk_id=chunk_data["chunk_id"],
                        source=chunk_data["source"],
                        doc_type=chunk_data["doc_type"],
                        stream_code=chunk_data["stream_code"],
                        subject_code=chunk_data["subject_code"],
                        year=chunk_data.get("year"),
                        page_number=chunk_data.get("page_number"),
                        metadata=chunk_data.get("metadata", {}),
                    )
                    self.chunks[chunk.chunk_id] = chunk

        if embeddings_file.exists():
            embeddings_array = np.load(embeddings_file, allow_pickle=True).item()
            self.embeddings = embeddings_array

    def _save_to_disk(self):
        """Save chunks to disk."""
        chunks_file = self.storage_path / "chunks.json"
        embeddings_file = self.storage_path / "embeddings.npy"

        # Save chunks
        chunks_data = []
        for chunk in self.chunks.values():
            chunks_data.append(
                {
                    "content": chunk.content,
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                    "doc_type": chunk.doc_type,
                    "stream_code": chunk.stream_code,
                    "subject_code": chunk.subject_code,
                    "year": chunk.year,
                    "page_number": chunk.page_number,
                    "metadata": chunk.metadata,
                }
            )

        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)

        # Save embeddings
        if self.embeddings:
            np.save(embeddings_file, self.embeddings)

    def add_chunks(self, chunks: List[Chunk]):
        """Add chunks to vector store."""
        for chunk in chunks:
            self.chunks[chunk.chunk_id] = chunk
            if chunk.embedding:
                self.embeddings[chunk.chunk_id] = chunk.embedding

        self._save_to_disk()

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        stream_code: Optional[str] = None,
        subject_code: Optional[str] = None,
        doc_type: Optional[str] = None,
        min_score: float = 0.0,
    ) -> List[SearchResult]:
        """
        Search for similar chunks.

        Args:
            query_embedding: Embedding of the query
            top_k: Number of results to return
            stream_code: Filter by stream
            subject_code: Filter by subject
            doc_type: Filter by document type
            min_score: Minimum similarity score

        Returns:
            List of SearchResult ordered by relevance
        """
        scores = []

        for chunk_id, embedding in self.embeddings.items():
            chunk = self.chunks.get(chunk_id)
            if not chunk:
                continue

            # Apply filters
            if stream_code and chunk.stream_code != stream_code:
                continue
            if subject_code and chunk.subject_code != subject_code:
                continue
            if doc_type and chunk.doc_type != doc_type:
                continue

            # Calculate similarity
            score = self._cosine_similarity(query_embedding, embedding)

            if score >= min_score:
                scores.append((chunk, score))

        # Sort by score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)

        # Create SearchResult objects
        results = []
        for rank, (chunk, score) in enumerate(scores[:top_k], 1):
            results.append(SearchResult(chunk=chunk, score=score, rank=rank))

        return results

    def get_stats(self) -> Dict:
        """Get statistics about the vector store."""
        return {
            "total_chunks": len(self.chunks),
            "total_embeddings": len(self.embeddings),
            "streams": list(set(c.stream_code for c in self.chunks.values())),
            "subjects": list(set(c.subject_code for c in self.chunks.values())),
            "doc_types": list(set(c.doc_type for c in self.chunks.values())),
        }


class RAGPipeline:
    """
    Complete RAG pipeline for context retrieval.

    Usage:
        rag = RAGPipeline()

        # Index documents
        chunks = rag.chunker.chunk_lesson(content, source, stream, subject)
        chunks = rag.embedding_service.embed_chunks(chunks)
        rag.vector_store.add_chunks(chunks)

        # Search
        results = rag.search("How to solve quadratic equations?", stream_code="MATH")
    """

    def __init__(self, mock_embeddings: bool = False):
        self.chunker = TextChunker()
        self.embedding_service = EmbeddingService(mock_mode=mock_embeddings)
        self.vector_store = VectorStore()

    def index_lesson(
        self,
        content: str,
        source: str,
        stream_code: str,
        subject_code: str,
        topic: str = None,
    ) -> List[Chunk]:
        """Index a lesson document."""
        chunks = self.chunker.chunk_lesson(
            content=content,
            source=source,
            stream_code=stream_code,
            subject_code=subject_code,
            topic=topic,
        )

        chunks = self.embedding_service.embed_chunks(chunks)
        self.vector_store.add_chunks(chunks)

        return chunks

    def index_exam(
        self,
        content: str,
        source: str,
        stream_code: str,
        subject_code: str,
        year: int,
        session: str = "principal",
    ) -> List[Chunk]:
        """Index an exam paper."""
        chunks = self.chunker.chunk_exam(
            content=content,
            source=source,
            stream_code=stream_code,
            subject_code=subject_code,
            year=year,
            session=session,
        )

        chunks = self.embedding_service.embed_chunks(chunks)
        self.vector_store.add_chunks(chunks)

        return chunks

    def search(
        self,
        query: str,
        top_k: int = 5,
        stream_code: Optional[str] = None,
        subject_code: Optional[str] = None,
        doc_type: Optional[str] = None,
        min_score: float = 0.5,
    ) -> List[SearchResult]:
        """
        Search for relevant context.

        Args:
            query: Search query text
            top_k: Number of results
            stream_code: Filter by stream
            subject_code: Filter by subject
            doc_type: Filter by document type
            min_score: Minimum similarity threshold

        Returns:
            List of SearchResult
        """
        # Generate query embedding
        query_embedding = self.embedding_service.embed_text(query)

        # Search vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            stream_code=stream_code,
            subject_code=subject_code,
            doc_type=doc_type,
            min_score=min_score,
        )

        return results

    def get_context_for_query(
        self, query: str, max_tokens: int = 2000, **search_kwargs
    ) -> str:
        """
        Get context string for a query (for LLM prompt).

        Args:
            query: User query
            max_tokens: Maximum context length (approximate)
            **search_kwargs: Additional search filters

        Returns:
            Concatenated context string
        """
        results = self.search(query, **search_kwargs)

        if not results:
            return ""

        contexts = []
        current_length = 0

        for result in results:
            context = f"""
[Source: {result.chunk.source} | Type: {result.chunk.doc_type} | Score: {result.score:.3f}]
{result.chunk.content}
---
"""
            # Rough token estimation (1 token ≈ 4 characters)
            estimated_tokens = len(context) / 4

            if current_length + estimated_tokens > max_tokens:
                break

            contexts.append(context)
            current_length += estimated_tokens

        return "\n".join(contexts)

    def get_stats(self) -> Dict:
        """Get pipeline statistics."""
        return {
            "embedding_model": self.embedding_service.model,
            "mock_mode": self.mock_mode if hasattr(self, "mock_mode") else False,
            "vector_store": self.vector_store.get_stats(),
        }


def create_sample_index():
    """Create a sample vector index for demonstration."""
    print("\n" + "=" * 60)
    print("Creating Sample RAG Index")
    print("=" * 60 + "\n")

    rag = RAGPipeline(mock_embeddings=True)

    # Sample lesson content
    lesson_content = """
## Les nombres complexes

### Définition
Un nombre complexe est un nombre de la forme z = a + ib, où a et b sont des nombres réels, et i est l'unité imaginaire telle que i² = -1.

### Partie réelle et partie imaginaire
- La partie réelle de z = a + ib est Re(z) = a
- La partie imaginaire de z = a + ib est Im(z) = b

### Représentation géométrique
Un nombre complexe z = a + ib peut être représenté par le point M(a, b) dans le plan complexe.

### Module d'un nombre complexe
Le module de z = a + ib est |z| = √(a² + b²)

### Argument d'un nombre complexe
L'argument de z ≠ 0 est l'angle θ tel que cos(θ) = a/|z| et sin(θ) = b/|z|

### Forme trigonométrique
z = |z|(cos(θ) + i sin(θ)) = |z|e^(iθ)

### Opérations sur les nombres complexes
Addition: (a + ib) + (c + id) = (a+c) + i(b+d)
Multiplication: (a + ib)(c + id) = (ac - bd) + i(ad + bc)
"""

    # Index the lesson
    print("[1/2] Indexing sample lesson...")
    chunks = rag.index_lesson(
        content=lesson_content,
        source="cours_mathematiques_complexes.md",
        stream_code="MATH",
        subject_code="MATH",
        topic="complex_numbers",
    )
    print(f"      Created {len(chunks)} chunks")

    # Sample exam content
    exam_content = """
Exercice 1: (5 points)
Soit z un nombre complexe tel que z = 1 + i√3

1. Calculer le module de z
2. Déterminer un argument de z
3. Écrire z sous forme trigonométrique

Solution:
1. |z| = √(1² + (√3)²) = √(1 + 3) = √4 = 2

2. cos(θ) = 1/2 et sin(θ) = √3/2, donc θ = π/3

3. z = 2(cos(π/3) + i sin(π/3)) = 2e^(iπ/3)

Exercice 2: (5 points)
Résoudre dans C l'équation: z² + 2z + 5 = 0

Solution:
Δ = 4 - 20 = -16 = (4i)²
z1 = (-2 + 4i)/2 = -1 + 2i
z2 = (-2 - 4i)/2 = -1 - 2i
"""

    print("[2/2] Indexing sample exam...")
    chunks = rag.index_exam(
        content=exam_content,
        source="bac_math_2023_principal.pdf",
        stream_code="MATH",
        subject_code="MATH",
        year=2023,
        session="principal",
    )
    print(f"      Created {len(chunks)} chunks")

    # Test search
    print("\n[TEST] Searching for 'module complex number'...")
    results = rag.search("module complex number", top_k=3)

    print(f"\n      Found {len(results)} results:")
    for result in results:
        print(f"\n      #{result.rank} (Score: {result.score:.3f})")
        print(f"      Source: {result.chunk.source}")
        print(f"      Type: {result.chunk.doc_type}")
        print(f"      Content preview: {result.chunk.content[:100]}...")

    # Get context
    print("\n[TEST] Getting context for query...")
    context = rag.get_context_for_query(
        "Comment calculer le module d'un nombre complexe?",
        stream_code="MATH",
        max_tokens=500,
    )

    print(f"\n      Context retrieved ({len(context)} chars)")
    print("      " + "=" * 50)
    print(context[:500] + "..." if len(context) > 500 else context)

    # Stats
    print("\n" + "=" * 60)
    print("Vector Store Statistics")
    print("=" * 60)
    stats = rag.vector_store.get_stats()
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Total embeddings: {stats['total_embeddings']}")
    print(f"Streams: {', '.join(stats['streams'])}")
    print(f"Subjects: {', '.join(stats['subjects'])}")
    print(f"Doc types: {', '.join(stats['doc_types'])}")
    print("=" * 60)

    return rag


if __name__ == "__main__":
    # Create sample index
    rag = create_sample_index()

    print("\n[OK] RAG Pipeline ready!")
    print("\nFeatures:")
    print("  - Text chunking (lessons, exercises, solutions)")
    print("  - OpenAI embeddings (with mock mode for dev)")
    print("  - Vector similarity search")
    print("  - Context retrieval for LLM prompts")
    print("  - Filtering by stream, subject, doc type")
    print("\nTo use in production:")
    print("  1. Set OPENAI_API_KEY environment variable")
    print("  2. Index your documents using rag.index_lesson() or rag.index_exam()")
    print(
        "  3. Search with rag.search() or get context with rag.get_context_for_query()"
    )
    print("  4. For production: migrate to Supabase pgvector")
