# RAG Pipeline (Retrieval-Augmented Generation)

## Overview

The RAG pipeline enables the AI tutor to retrieve relevant curriculum content before generating responses. This improves accuracy and ensures answers are grounded in the Algerian Baccalaureat curriculum.

## Architecture

```
User Query
    ↓
Query Embedding (OpenAI)
    ↓
Vector Similarity Search
    ↓
Retrieve Top-K Chunks
    ↓
Format Context
    ↓
Add to LLM Prompt
    ↓
Generate Response
```

## Components

### 1. TextChunker (`rag_pipeline.py`)
Handles different chunking strategies:

- **Lesson Chunking**: Split by headers, 1000 chars with 200 overlap
- **Exercise Chunking**: Keep exercises intact, 1500 chars
- **Solution Chunking**: Step-by-step, 800 chars
- **General Chunking**: Recursive character splitting

### 2. EmbeddingService
Generates vector embeddings:

- **Model**: OpenAI text-embedding-3-small (or text-embedding-3-large)
- **Mock Mode**: Deterministic random embeddings for development
- **Dimensions**: 1536

### 3. VectorStore
Local vector storage (SQLite-based):

- **Storage**: `data/vector_store/`
- **Format**: JSON for chunks, NumPy for embeddings
- **Similarity**: Cosine similarity
- **Production**: Will migrate to Supabase pgvector

### 4. RAGPipeline
Complete pipeline combining all components:

```python
from rag_pipeline import RAGPipeline

rag = RAGPipeline()

# Index content
rag.index_lesson(content, source, stream_code, subject_code)
rag.index_exam(content, source, stream_code, subject_code, year)

# Search
results = rag.search("query", top_k=5)

# Get context for LLM
context = rag.get_context_for_query("query", max_tokens=2000)
```

## API Endpoints

### 1. Vector Search
```http
POST /search-context
```

Search for relevant content in the curriculum.

**Request:**
```json
{
    "query": "How to calculate module of complex number?",
    "top_k": 5,
    "stream_code": "MATH",
    "subject_code": "MATH",
    "doc_type": "lesson",
    "min_score": 0.5
}
```

**Response:**
```json
{
    "query": "How to calculate module of complex number?",
    "results": [
        {
            "chunk_id": "chunk_abc123",
            "content": "Le module de z = a + ib est |z| = √(a² + b²)...",
            "source": "cours_complexes.md",
            "doc_type": "lesson",
            "stream_code": "MATH",
            "subject_code": "MATH",
            "score": 0.89,
            "rank": 1
        }
    ],
    "total_results": 1,
    "filters_applied": {
        "stream_code": "MATH",
        "subject_code": "MATH",
        "doc_type": "lesson",
        "min_score": 0.5
    }
}
```

### 2. Get Context for LLM
```http
POST /get-context-for-llm
```

Get formatted context ready for LLM prompt.

**Request:**
```json
{
    "query": "Explain complex numbers",
    "max_tokens": 2000,
    "stream_code": "MATH"
}
```

**Response:**
```json
{
    "context": "[Source: cours_complexes.md...]\\nLe module de z...",
    "chunks_used": 3,
    "character_count": 1500,
    "estimated_tokens": 375,
    "query": "Explain complex numbers"
}
```

### 3. RAG Statistics
```http
GET /rag-stats
```

Get pipeline statistics.

**Response:**
```json
{
    "status": "active",
    "embedding_model": "text-embedding-3-small",
    "mock_mode": false,
    "total_chunks": 150,
    "total_embeddings": 150,
    "streams": ["MATH", "SCI_EXP", "TECH_MATH"],
    "subjects": ["MATH", "PHYSICS", "SCIENCES"],
    "doc_types": ["lesson", "exercise", "solution"]
}
```

## Usage Examples

### Indexing Content

```python
from rag_pipeline import RAGPipeline

rag = RAGPipeline()

# Index a lesson
chunks = rag.index_lesson(
    content="Les nombres complexes...",
    source="complex_numbers_lesson.md",
    stream_code="MATH",
    subject_code="MATH",
    topic="complex_numbers"
)
print(f"Indexed {len(chunks)} chunks")

# Index an exam
chunks = rag.index_exam(
    content="Exercice 1: Calculer...",
    source="bac_math_2023.pdf",
    stream_code="MATH",
    subject_code="MATH",
    year=2023,
    session="principal"
)
```

### Searching

```python
# Basic search
results = rag.search("module complexe", top_k=5)

# Filtered search
results = rag.search(
    "equation second degré",
    top_k=10,
    stream_code="MATH",
    subject_code="MATH",
    doc_type="exercise"
)

# Process results
for result in results:
    print(f"#{result.rank}: {result.chunk.content[:100]}...")
    print(f"   Score: {result.score:.3f}")
    print(f"   Source: {result.chunk.source}")
```

### Getting Context

```python
# Get context for LLM
context = rag.get_context_for_query(
    query="Comment résoudre une équation du second degré?",
    max_tokens=2000,
    stream_code="MATH",
    subject_code="MATH"
)

# Use in prompt
prompt = f"""
Context: {context}

Question: Comment résoudre une équation du second degré?
"""
```

## Configuration

### Environment Variables

```bash
# Required for real embeddings
export OPENAI_API_KEY="sk-..."

# Optional: Use larger model
export OPENAI_EMBEDDING_MODEL="text-embedding-3-large"
```

### Mock Mode

For development without OpenAI API:

```python
rag = RAGPipeline(mock_embeddings=True)
```

This generates deterministic random embeddings (based on content hash).

## Chunking Strategies

### Lessons
- Size: 1000 characters
- Overlap: 200 characters
- Separators: Headers (##, ###), paragraphs, sentences
- Best for: Theory, explanations, definitions

### Exercises
- Size: 1500 characters
- Overlap: 100 characters
- Separators: "Exercice", "Exercise", "التمرين"
- Best for: Problem statements, exam questions

### Solutions
- Size: 800 characters
- Overlap: 150 characters
- Separators: Steps, sub-headers
- Best for: Detailed solutions, step-by-step explanations

## Performance

### Embedding Costs (OpenAI)
- text-embedding-3-small: ~$0.02 per 1M tokens
- text-embedding-3-large: ~$0.13 per 1M tokens

### Local Storage
- Each chunk: ~2-5KB
- Each embedding: ~6KB (1536 floats × 4 bytes)
- 1000 chunks: ~8-11MB

### Search Performance
- In-memory search: <10ms for 1000 chunks
- Disk-based search: <100ms for 1000 chunks
- Scales to 10,000+ chunks locally

## Testing

Run RAG tests:

```bash
cd backend
poetry run python -m pytest tests/test_rag_pipeline.py -v
```

Tests cover:
- Vector search with filters
- Context retrieval
- Response structure validation
- Integration with API endpoints

## Migration to Production

### Current (Local)
- SQLite-based vector store
- File-based persistence
- Cosine similarity in Python

### Production (Supabase pgvector)
1. Enable pgvector extension
2. Create tables with vector columns
3. Update VectorStore class
4. Use `<=>` operator for similarity
5. Add indexing (IVFFlat or HNSW)

Example migration:
```sql
-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Create table
CREATE TABLE chunks (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding VECTOR(1536),
    stream_code TEXT,
    subject_code TEXT,
    doc_type TEXT
);

-- Create index
CREATE INDEX ON chunks USING ivfflat (embedding vector_cosine_ops);

-- Search query
SELECT * FROM chunks
ORDER BY embedding <=> query_embedding
LIMIT 5;
```

## Future Enhancements

1. **Hybrid Search**: Combine vector + keyword search
2. **Re-ranking**: Use cross-encoder for better results
3. **Caching**: Cache frequent queries
4. **Metadata Filtering**: Year range, difficulty level
5. **Multi-modal**: Support for diagrams/images
6. **Streaming**: Stream search results
7. **Analytics**: Track most searched topics

## Troubleshooting

### Low similarity scores
- Check if content is indexed
- Verify embedding model
- Adjust `min_score` threshold

### Slow search
- Reduce number of chunks
- Use approximate nearest neighbors (ANN)
- Add metadata filters

### Missing results
- Check filters (stream_code, subject_code)
- Verify content is indexed
- Try different query phrasing

### API errors
- Verify OPENAI_API_KEY is set
- Check rate limits
- Use mock mode for testing

## References

- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity)
