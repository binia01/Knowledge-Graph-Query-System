# Knowledge Graph Query System

A natural language query system for a Neo4j Stack Overflow knowledge graph. Ask questions in plain English — the system generates Cypher queries, validates them, executes against Neo4j, and returns human-friendly answers through a web dashboard.

**Stack:** Google Gemini · LangChain · Neo4j · Flask

---
# Features

1. End-to-end NL → Cypher → Neo4j → NL pipeline 
2. 5+ natural language query types 
3. Auto-detect and retry failed Cypher queries 
4. Ambiguous query detection with clarification 
5. Vector embeddings on Question nodes 
6. Hybrid search (vector + graph traversal) 
7. Confidence scoring (0.7 threshold) 
8. 3 validation test cases 
9. Conversation memory for follow-ups 
10.  Agent for multi-step complex questions 
11.  Query result caching (bonus) 
12.  Web dashboard 

---

## Architecture


```mermaid

flowchart TD
    A[User Question] --> B[Query Classifier]
    B -->|graph| C[Graph]
    B -->|vector| D[Vector]
    B -->|hybrid| E[Hybrid]
    B -->|agent| F[Agent]

    C --> G[Cypher Generator]
    D --> H[Embedding Search]
    E --> I[Vector + Graph Filter]
    F --> J[ReAct Loop (multi-step)]

    G --> K[Validator (confidence scoring)]
    H --> K
    I --> K
    J --> K

    K -->|score ≥ 0.7| L[Execute on Neo4j]
    K -->|0.4–0.69| M[Auto-correct & retry]
    K -->|score < 0.4| N[Reject, ask user to clarify]

    L --> O[Gemini humanizes results → Natural language answer]
    M --> O
    N --> O
   
```


---

## Setup

### Prerequisites

- Python 3.11+
- Neo4j instance (local or [Neo4j Sandbox](https://sandbox.neo4j.com)) with the **Stack Overflow** dataset loaded
- Google Gemini API key from [AI Studio](https://aistudio.google.com)

### 1. Clone and install

```bash
git clone https://github.com/your-username/Knowledge-Graph-Query-System.git
cd Knowledge-Graph-Query-System
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your actual keys:
#   GOOGLE_API_KEY=your_gemini_key
#   NEO4J_URI=bolt://localhost:7687
#   NEO4J_USERNAME=neo4j
#   NEO4J_PASSWORD=your_password
```

### 3. Load the Stack Overflow dataset

In Neo4j Browser, run `:play stackoverflow` and execute the provided Cypher script to create the graph.

### 4. Set up vector embeddings (for similarity & hybrid search)

```bash
python setup_embeddings.py
```

This generates embeddings for all Question nodes using Google Gemini and creates a vector index.

### 5. Run the dashboard

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## Usage

### Example Queries

| Type | Example |
|------|---------|
| **Graph traversal** | "What are the most viewed questions?" |
| **Graph traversal** | "Show me questions tagged with 'python'" |
| **Similarity** | "Find questions about optimizing database queries" |
| **Hybrid** | "Find questions about memory management tagged with 'c++'" |
| **Agent** | "Which users have the most answers, and what tags do they contribute to?" |
| **Follow-up** | "What tags do those questions have?" (after asking about viewed questions) |
| **Ambiguous** | "Show me some stuff about code" (triggers clarification) |

### Running Tests

```bash
python -m pytest tests/test_validation.py -v
```

---

## Project Structure

```
├── app.py                  # Flask web application
├── setup_embeddings.py     # One-time vector embedding setup
├── requirements.txt
├── .env.example
├── src/
│   ├── __init__.py
│   ├── config.py           # Configuration from environment
│   ├── database.py         # Neo4j connection & queries
│   ├── llm.py              # Gemini LLM & embedding setup
│   ├── query_classifier.py # Question type classification
│   ├── cypher_generator.py # NL → Cypher generation
│   ├── validator.py        # Cypher confidence scoring
│   ├── vector_search.py    # Vector similarity & hybrid search
│   ├── agent.py            # ReAct agent for complex queries
│   ├── memory.py           # Conversation history
│   ├── cache.py            # Query result caching
│   └── pipeline.py         # Main orchestration pipeline
├── templates/
│   └── index.html          # Dashboard HTML
├── static/
│   ├── css/style.css       # Dashboard styles
│   └── js/app.js           # Dashboard JavaScript
├── tests/
│   └── test_validation.py  # Validation test cases
└── context.md              # Assignment specification
```

---

## Validation Test Cases

The system includes 3 test cases demonstrating the validation layer:

1. **Reversed relationship direction** — `(Tag)-[:TAGGED]->(Question)` is valid Cypher but returns empty results. The validator catches the reversed direction and scores it below 0.7.

2. **Non-existent node label** — `MATCH (p:Post)` uses a label that doesn't exist in the schema (correct: `Question`). The validator detects the mismatch.

3. **Ambiguous query** — "Show me some stuff about code" is too vague. The classifier routes it to the ambiguous handler, and if it reaches validation, the validator flags it.

---

## How It Works

### Confidence Scoring

Every generated Cypher query is evaluated by Gemini before execution:

| Score | Action |
|-------|--------|
| ≥ 0.7 | Execute directly |
| 0.4–0.69 | Auto-correct issues and retry |
| < 0.4 | Reject — ask user to rephrase |

### Conversation Memory

The system uses `ConversationBufferMemory` to maintain chat history. Follow-up questions are resolved by rewriting them as standalone questions using the conversation context.

### Caching

Identical questions are served from an in-memory LRU cache, avoiding redundant Neo4j and Gemini API calls.