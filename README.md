# Brainstorming Agent v2

AI-powered chat assistant untuk brainstorming dan development tasks dengan dukungan memory management, PRD generation, code assistance, dan multi-tool integration menggunakan LangGraph, FastAPI, dan Supabase.

## Base URL

```
http://localhost:8008
```

## Tech Stack

- **FastAPI**: REST API framework
- **LangChain**: LLM integration & tooling
- **DeepAgents**: Advanced AI agent framework dengan subagents
- **Supabase/PostgreSQL**: Database & storage
- **LangMem**: Memory management untuk user context
- **Tavily**: Web search tool

---

## Endpoints

### 1. Health Check

**GET** `/api/`

Cek status API.

**Response:**
```json
{
  "message": "Brainstorming Agent v2",
  "status": "healthy"
}
```

---

### 2. Chat dengan Agent

**POST** `/api/chat/{thread_id}`

Stream conversation dengan AI agent King Abel yang bisa menggunakan berbagai tools (memory, search, PRD generation, code assistance, dll).

**Path Parameters:**
- `thread_id` (string, required): Unique identifier untuk conversation thread

**Request Body:**
```json
{
  "query": "Saya butuh PRD untuk fitur login social media",
  "user_id": "user_123"
}
```

**Fields:**
- `query` (string, 1-5000 chars): Pertanyaan atau instruksi user
- `user_id` (string, required): User identifier

**Response:**
- Content-Type: `text/event-stream`
- Streaming response dengan SSE format

**Example Request:**
```bash
curl -X POST "http://localhost:8008/api/chat/thread_abc123" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Buatkan PRD untuk fitur authentication",
    "user_id": "user_123"
  }'
```

**Agent Capabilities:**
- 🧠 Memory search & management (user preferences, context)
- 🔍 Web search (Tavily) untuk research & current events
- 📝 PRD generation & updates
- 💻 Code assistance (bash commands, reviews, testing)
- 🔧 HTTP requests untuk API integration
- ⏰ Current time information
- 💬 Multi-language support (auto-detect)
- 🤖 Subagents (code reviewer, test generator)

**Error Responses:**
- `400`: Invalid request payload
- `500`: Internal server error atau database connection issue

---

### 3. Get Chat History

**GET** `/api/chat/{thread_id}/history`

Ambil semua messages dan PRD info dari sebuah thread.

**Path Parameters:**
- `thread_id` (string, required): Thread identifier

**Query Parameters:**
- `user_id` (string, required): User identifier untuk authorization

**Response:**
```json
{
  "thread_id": "thread_abc123",
  "messages": [
    {
      "type": "human",
      "content": "Buatkan PRD untuk login feature"
    },
    {
      "type": "ai",
      "content": "Baik, saya akan buatkan PRD...",
      "tool_calls": [
        {
          "name": "generate_prd",
          "args": {
            "prd_id": "thread_abc123",
            "feature": "Social Login"
          }
        }
      ]
    },
    {
      "type": "tool",
      "content": "PRD created successfully",
      "tool_name": "generate_prd"
    }
  ],
  "has_prd": true,
  "prd": {
    "prd_id": "thread_abc123",
    "feature": "Social Login Authentication",
    "introduction": "Enable users to login using...",
    "version": 1,
    "created_at": "2025-10-19T10:30:00Z",
    "updated_at": "2025-10-19T10:30:00Z"
  }
}
```

**Message Types:**
- `human`: User messages
- `ai`: Assistant responses
- `tool`: Tool execution results

**Error Responses:**
- `403`: Unauthorized access (user bukan pemilik thread)
- `404`: Thread not found
- `500`: Internal server error

**Example Request:**
```bash
curl "http://localhost:8008/api/chat/thread_abc123/history?user_id=user_123"
```

---

### 4. List User Threads

**GET** `/api/chat/user/{user_id}/threads`

Ambil semua conversation threads milik user, sorted by most recent.

**Path Parameters:**
- `user_id` (string, required): User identifier

**Response:**
```json
[
  {
    "thread_id": "thread_abc123",
    "message_count": 8,
    "last_checkpoint_id": "1729338600000",
    "has_prd": true,
    "prd": {
      "prd_id": "thread_abc123",
      "feature": "Social Login",
      "version": 2
    }
  },
  {
    "thread_id": "thread_xyz789",
    "message_count": 3,
    "last_checkpoint_id": "1729252200000",
    "has_prd": false,
    "prd": null
  }
]
```

**Fields:**
- `message_count`: Total messages dalam thread
- `last_checkpoint_id`: Timestamp untuk sorting (larger = newer)
- `has_prd`: Boolean indicator ada PRD atau tidak
- `prd`: PRD summary jika ada

**Error Responses:**
- `500`: Internal server error

**Example Request:**
```bash
curl "http://localhost:8008/api/chat/user/user_123/threads"
```

---

## Environment Variables

```env
# Database
DB_URI=postgresql://user:password@localhost:5432/dbname

# AI Services
LITELLM_API_KEY=sk-...
LITELLM_BASE_URL=https://...

# Search & Scraping
TAVILY_API_KEY=tvly-...
FIRECRAWL_API_KEY=fc-...

# Supabase (optional, for additional features)
SUPABASE_URL=https://...
SUPABASE_ANON_KEY=...
```

## Database Schema

### Checkpoints Table
```sql
CREATE TABLE checkpoints (
  checkpoint_id BIGINT,
  thread_id TEXT,
  user_id TEXT,
  checkpoint_ns TEXT,
  -- ... LangGraph checkpoint columns
);
```

### PRDs Table
```sql
CREATE TABLE prds (
  id UUID PRIMARY KEY,
  user_id TEXT,
  feature TEXT,
  introduction TEXT,
  user_stories JSONB,
  functional_requirements JSONB,
  non_functional_requirements JSONB,
  assumptions JSONB,
  dependencies JSONB,
  risks_and_mitigations JSONB,
  timeline TEXT,
  stakeholders JSONB,
  metrics JSONB,
  version INTEGER,
  created_at TIMESTAMP,
  updated_at TIMESTAMP
);
```

## Agent Tools

### Memory Tools
- **manage_memory**: Save user preferences, context, goals
- **search_memory**: Retrieve saved user information

### PRD Tools
- **generate_prd**: Create new Product Requirements Document
- **update_prd**: Update existing PRD dengan versioning

### Code Tools
- **execute_bash**: Run shell commands (linting, testing, building)
- **web_search**: Search web untuk documentation & solutions
- **http_request**: Make HTTP requests untuk API integration

### Utility Tools
- **TavilySearch**: Web search untuk current events
- **get_current_time**: Get current datetime

### Subagents
- **code_reviewer_agent**: Expert code reviewer untuk quality & security
- **test_generator_agent**: Automated test suite generator

## Usage Flow

### 1. Membuat PRD
```bash
# Start conversation
POST /api/chat/new_thread_123
{
  "query": "Buatkan PRD untuk notification system",
  "user_id": "user_123"
}

# Get PRD hasil
GET /api/chat/new_thread_123/history?user_id=user_123
```

### 2. Update PRD
```bash
POST /api/chat/new_thread_123
{
  "query": "Update PRD, tambahkan push notification support",
  "user_id": "user_123"
}
```

### 3. Memory Management
```bash
# Agent akan auto-save preferences
POST /api/chat/thread_456
{
  "query": "Saya lebih suka menggunakan TypeScript untuk backend",
  "user_id": "user_123"
}

# Nanti agent akan ingat preferences
POST /api/chat/thread_789
{
  "query": "Rekomendasikan stack untuk project baru",
  "user_id": "user_123"
}
# Agent akan suggest TypeScript karena ingat preferences
```

## Error Handling

Semua endpoints menggunakan standard HTTP status codes:
- `200`: Success
- `400`: Bad Request (invalid payload)
- `403`: Forbidden (unauthorized access)
- `404`: Not Found
- `500`: Internal Server Error

Error response format:
```json
{
  "detail": "Error message here"
}
```

## Rate Limiting

(To be implemented)

## Authentication

(To be implemented - currently using user_id parameter)

---

## Development

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env

# Run migrations
alembic upgrade head

# Start server
uvicorn main:app --host 0.0.0.0 --port 8008 --reload
```

### Testing
```bash
# Run tests
pytest tests/

# With coverage
pytest --cov=src tests/

# Using Makefile
make test
make test-cov
```

### Additional Commands
```bash
# Development setup
make setup

# Run server
make run

# Clean up
make clean
```
