# RAG Chat Assistant 🤖

A production-ready **Retrieval-Augmented Generation (RAG)** chat application built with Streamlit and Azure AI services. This app enables intelligent document search and conversational AI with full observability through Azure AI Foundry tracing.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![Azure](https://img.shields.io/badge/Azure-AI%20Services-0078D4.svg)

## 📸 App Preview

![RAG Chat Assistant Interface](./StreamlitRAG-App/App%20Pic/AppPic.jpg)

*The RAG Chat Assistant provides an intuitive interface for document-based question answering with real-time response evaluation.*

## ✨ Features

### Core Capabilities
- **Intelligent Document Search** - Hybrid search combining text and vector similarity
- **Schema-Independent Index Support** - Works with any Azure AI Search index schema
- **Multi-Model Support** - Switch between GPT-4o, GPT-4o-mini, and other deployments
- **Document Ingestion** - Process PDFs and documents using Azure Document Intelligence
- **Response Evaluation** - Automatic scoring for relevance, coherence, and groundedness

### Advanced Features
- **Agentic Retrieval** - Query rewriting and iterative search for better results
- **Vector Embeddings** - Support for text-embedding-3-small/large models
- **Feedback Collection** - User feedback stored in Cosmos DB for continuous improvement
- **Full Observability** - AI Foundry tracing with gen_ai.* semantic attributes

### Performance Optimizations
- **Lazy Tracing** - Deferred tracing setup for faster startup
- **Field Caching** - Cached schema detection for quick index switching
- **Session Caching** - Streamlit resource caching for efficient reloads

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Streamlit UI (app.py)                       │
├─────────────────────────────────────────────────────────────────┤
│                    RAG Services Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ OpenAIClient │  │ SearchClient │  │EmbeddingClient│           │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
│         │                 │                 │                    │
├─────────┴─────────────────┴─────────────────┴────────────────────┤
│                      Azure Services                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ Azure OpenAI │  │  AI Search   │  │  Cosmos DB   │           │
│  │ (AI Foundry) │  │              │  │  (Feedback)  │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
│  ┌──────────────┐  ┌──────────────┐                              │
│  │Doc Intelligence│ │ App Insights │                             │
│  │              │  │  (Tracing)   │                              │
│  └──────────────┘  └──────────────┘                              │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Azure subscription with the following services:
  - Azure OpenAI or AI Foundry project
  - Azure AI Search
  - Azure Cosmos DB
  - Azure Document Intelligence (optional, for PDF ingestion)
  - Application Insights (for tracing)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd StreamlitApp-co-pilot
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/Mac
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your Azure credentials
   ```

5. **Run the application**
   ```bash
   # Windows
   start.bat
   # Linux/Mac
   ./start.sh
   # Or directly
   streamlit run app.py
   ```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file with the following configuration:

#### Azure AI Search
```env
AZURE_SEARCH_SERVICE_NAME=your-search-service
AZURE_SEARCH_API_KEY=your-api-key
AZURE_SEARCH_INDEX_NAME=your-index-name
```

#### Azure OpenAI / AI Foundry
```env
AZURE_OPENAI_ENDPOINT=https://your-endpoint.cognitiveservices.azure.com/
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_DEPLOYMENTS=gpt-4o,gpt-4o-mini
AZURE_OPENAI_DEFAULT_DEPLOYMENT=gpt-4o-mini
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
AZURE_OPENAI_EMBEDDING_DIMENSIONS=1536
```

#### Azure Cosmos DB
```env
COSMOS_DB_ENDPOINT=https://your-cosmos.documents.azure.com:443/
COSMOS_DB_KEY=your-cosmos-key
COSMOS_DB_DATABASE_NAME=RAGfeedback
COSMOS_DB_CONTAINER_NAME=RAGFeed
COSMOS_USE_AAD=true
```

#### Azure Document Intelligence
```env
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://your-docintel.cognitiveservices.azure.com/
AZURE_DOCUMENT_INTELLIGENCE_API_KEY=your-api-key
```

#### Index & Ingestion Settings
```env
USE_EXISTING_INDEX=true      # Use existing index without auto-ingestion
AUTO_INGEST=false            # Auto-ingest on startup (if USE_EXISTING_INDEX=false)
LOCAL_DATA_PATH=./data       # Path to documents for ingestion
CHUNK_SIZE=1000              # Document chunk size
CHUNK_OVERLAP=200            # Overlap between chunks
```

#### Tracing Settings
```env
ENABLE_TRACING=true          # Enable/disable tracing
LAZY_TRACING=true            # Defer tracing setup until first API call
AZURE_AI_FOUNDRY_ENDPOINT=https://your-foundry.services.ai.azure.com/api/projects/your-project
APPLICATIONINSIGHTS_CONNECTION_STRING=InstrumentationKey=...
```

#### Identity
```env
AZURE_TENANT_ID=your-tenant-id   # For multi-tenant scenarios
```

## 📖 Usage

### Chat Interface
1. Open the app in your browser (default: http://localhost:8501)
2. Type your question in the chat input
3. View AI response with source documents
4. Provide feedback using 👍/👎 buttons

### Admin Panel (Sidebar)

#### Index Configuration
- **Index Name**: Switch between different Azure AI Search indexes at runtime
- **Mode**: View current index mode (existing vs create/update)
- **Data Path**: Local path for document ingestion

#### Document Ingestion
- Click "Ingest Documents" to process files from `LOCAL_DATA_PATH`
- Supports PDF, DOCX, and text files
- Automatically creates embeddings and indexes documents

#### Settings
- **Model Selection**: Choose between available deployments
- **Evaluation Model**: Select model for response evaluation
- **Search Mode**: Toggle semantic ranking (requires Standard tier)
- **Agentic Retrieval**: Enable query rewriting for better results
- **Auto-Evaluate**: Automatically score responses

### System Information
View current configuration including:
- Active search index
- Tracing status
- Session ID
- Message count

## 🔍 Schema-Independent Search

The app automatically detects and maps fields from any Azure AI Search index:

| Detected As | Common Field Names |
|------------|-------------------|
| Content | `content`, `content_text`, `text`, `chunk`, `body` |
| Title | `title`, `document_title`, `name`, `filename` |
| Vector | `content_vector`, `content_embedding`, `embedding` |
| URL | `url`, `content_path`, `source_url`, `filepath` |
| ID | `id`, `content_id`, `chunk_id`, `doc_id` |

This allows you to switch between different indexes without code changes.

## 📊 Observability & Tracing

### AI Foundry Integration
The app integrates with Azure AI Foundry for comprehensive tracing:

- **gen_ai.* attributes**: Standard semantic conventions for AI operations
- **Evaluation metrics**: `gen_ai.evaluation.relevance`, `gen_ai.evaluation.coherence`, etc.
- **Request/Response tracking**: Full token usage and latency metrics

### Viewing Traces
1. Open Azure AI Foundry portal
2. Navigate to your project → Tracing
3. Filter by service name: `rag-chat-service`

### Kusto Queries
```kusto
// View all RAG operations
traces
| where customDimensions["service.name"] == "rag-chat-service"
| project timestamp, name, customDimensions

// View evaluation scores
traces
| where name startswith "gen_ai.evaluation"
| extend score = todouble(customDimensions["gen_ai.evaluation.score"])
| extend evaluator = tostring(customDimensions["gen_ai.evaluator.name"])
```

## 📁 Project Structure

```
StreamlitApp-co-pilot/
├── app.py                 # Main Streamlit application
├── rag_services.py        # Core service classes and utilities
├── requirements.txt       # Python dependencies
├── .env                   # Environment configuration (not in git)
├── .env.example           # Example configuration template
├── start.bat              # Windows startup script
├── start.sh               # Linux/Mac startup script
├── setup.py               # Azure Search index setup
├── data/                  # Document storage for ingestion
├── verify_traces.py       # Tracing verification script
└── README.md              # This file
```

## 🛠️ Core Components

### rag_services.py

| Class | Description |
|-------|-------------|
| `ConfigManager` | Centralized configuration with lazy tracing support |
| `AzureSearchClient` | Schema-independent document search and retrieval |
| `DocumentProcessor` | Text extraction and chunking with Document Intelligence |
| `CosmosDBClient` | Feedback and interaction storage |
| `OpenAIClient` | AI response generation with tracing |
| `EmbeddingClient` | Vector embedding generation |
| `AgenticRetrieval` | Advanced multi-step search with query rewriting |
| `EvaluationService` | Response quality assessment |

### Key Functions

| Function | Description |
|----------|-------------|
| `ingest_documents()` | Process and index documents with embeddings |
| `record_evaluation_metrics()` | Emit evaluation traces for AI Foundry |
| `rag_trace_span()` | Context manager for operation tracing |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Troubleshooting

### Common Issues

**400 Bad Request on Search**
- Check if index exists and has documents
- Verify API key permissions
- Schema detection will auto-map fields

**Tracing not appearing in AI Foundry**
- Ensure `ENABLE_TRACING=true`
- Check Application Insights connection string
- Wait 2-5 minutes for traces to appear

**Slow startup**
- Set `LAZY_TRACING=true` in `.env`
- Tracing setup is deferred until first API call

**Authentication errors**
- Verify `AZURE_TENANT_ID` if using multi-tenant
- Run `az login --tenant <tenant-id>` for CLI auth
- Check service principal permissions

**No Documents Retrieved**
- Verify index has documents: check Azure portal
- Try a simpler query to test connectivity
- Check field mappings in logs

## 📚 Resources

- [Azure AI Search Documentation](https://learn.microsoft.com/azure/search/)
- [Azure OpenAI Service](https://learn.microsoft.com/azure/ai-services/openai/)
- [Azure AI Foundry](https://learn.microsoft.com/azure/ai-foundry/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

Built with ❤️ using Azure AI services and Streamlit