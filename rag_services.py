import os
import warnings
from typing import List, Dict, Any, Optional, Tuple
import logging
from contextlib import contextmanager

# Suppress pkg_resources deprecation warning from opentelemetry
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.cosmos import CosmosClient, PartitionKey
from azure.identity import DefaultAzureCredential, AzureCliCredential
from openai import AzureOpenAI
from datetime import datetime
import uuid
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global tracer and clients for AI Foundry
_ai_foundry_tracer = None
_ai_project_client = None
_ai_foundry_openai_client = None
_trace_id_counter = 0


def _get_credential(tenant_id: Optional[str] = None):
    """Return an Azure credential, preferring Azure CLI when available.

    Falls back to DefaultAzureCredential if Azure CLI is not installed or not logged in.
    """
    try:
        if tenant_id:
            return AzureCliCredential(tenant_id=tenant_id)
        return AzureCliCredential()
    except Exception as cli_err:
        logger.warning(f"AzureCliCredential unavailable ({cli_err}); falling back to DefaultAzureCredential")
        try:
            return DefaultAzureCredential()
        except Exception as default_err:
            logger.error(f"DefaultAzureCredential failed: {default_err}")
            raise


# Application name for tracing - this appears in AI Foundry portal
RAG_APP_NAME = "RAG-Chat-Assistant"
SERVICE_NAME = "rag-chat-service"


@contextmanager
def rag_trace_span(operation_name: str, attributes: Dict[str, Any] = None):
    """Context manager to create a parent span for RAG operations.
    
    This wraps all child operations under a single trace name for better 
    visibility in AI Foundry portal.
    
    Usage:
        with rag_trace_span("chat_query", {"user_id": "123"}):
            # All operations here will be children of this span
            retrieve_docs()
            generate_response()
    """
    global _ai_foundry_tracer, _trace_id_counter
    
    if _ai_foundry_tracer:
        _trace_id_counter += 1
        span_name = f"{RAG_APP_NAME}.{operation_name}"
        
        with _ai_foundry_tracer.start_as_current_span(span_name) as span:
            # Set OpenTelemetry semantic convention attributes
            span.set_attribute("service.name", SERVICE_NAME)
            span.set_attribute("service.version", "1.0.0")
            span.set_attribute("ai.application.name", RAG_APP_NAME)
            span.set_attribute("gen_ai.operation.name", operation_name)
            span.set_attribute("gen_ai.system", "azure_openai")
            
            # Set custom attributes
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, str(value) if value is not None else "")
            
            yield span
    else:
        # No tracer available, just yield None
        yield None


def _setup_ai_foundry_tracing(endpoint: str, tenant_id: str = None, app_insights_conn_str: str = None):
    """Setup AI Foundry tracing using azure-ai-projects SDK with endpoint.
    
    This enables tracing/telemetry to AI Foundry portal via Application Insights.
    Per latest docs: https://learn.microsoft.com/en-us/azure/ai-foundry/observability/how-to/trace-agent-setup
    
    Key requirements for gen_ai.* attributes to appear in Foundry:
    1. Use AIProjectInstrumentor from azure.ai.projects.telemetry
    2. Set OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true for content
    
    Args:
        endpoint: AI Foundry project endpoint
        tenant_id: Azure tenant ID for authentication
        app_insights_conn_str: Application Insights connection string (from .env or AI Foundry)
    """
    global _ai_foundry_tracer, _ai_project_client, _ai_foundry_openai_client
    try:
        from azure.ai.projects import AIProjectClient
        from azure.monitor.opentelemetry import configure_azure_monitor
        from opentelemetry import trace as otel_trace
        
        # Set service name BEFORE configuring Azure Monitor
        os.environ["OTEL_SERVICE_NAME"] = SERVICE_NAME
        
        # Enable content recording to capture message content in traces
        # Per docs: OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT
        os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "true"
        
        # Prefer Azure CLI credential, fallback to DefaultAzureCredential
        credential = _get_credential(tenant_id)
        
        # Create AI Project client using endpoint (for telemetry)
        _ai_project_client = AIProjectClient(
            endpoint=endpoint,
            credential=credential
        )
        
        # Get the Application Insights connection string from project
        # This is the RECOMMENDED approach per latest docs
        try:
            ai_foundry_conn_str = _ai_project_client.telemetry.get_application_insights_connection_string()
            if ai_foundry_conn_str:
                app_insights_conn_str = ai_foundry_conn_str
                logger.info("Retrieved App Insights connection string from AI Foundry project")
        except Exception as e:
            logger.warning(f"Could not get App Insights from AI Foundry project: {e}")
            logger.info("Using App Insights connection string from .env")
        
        if app_insights_conn_str:
            # Configure Azure Monitor to export traces to App Insights
            configure_azure_monitor(connection_string=app_insights_conn_str)
            logger.info(f"Azure Monitor configured - Service: {SERVICE_NAME}")
            
            # Use AIProjectInstrumentor - this adds gen_ai.* attributes that Foundry expects
            # Per docs: https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/ai/azure-ai-projects#tracing
            try:
                from azure.ai.projects.telemetry import AIProjectInstrumentor
                AIProjectInstrumentor().instrument()
                logger.info("AIProjectInstrumentor enabled - gen_ai.* attributes will be traced")
            except ImportError:
                logger.warning("AIProjectInstrumentor not available, trying OpenAIInstrumentor")
                try:
                    from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor
                    OpenAIInstrumentor().instrument()
                    logger.info("OpenAI SDK instrumented for tracing (fallback)")
                except ImportError as e:
                    logger.warning(f"No instrumentor available: {e}")
        else:
            logger.warning("No App Insights connection string available - traces won't be exported")
        
        # Get tracer
        _ai_foundry_tracer = otel_trace.get_tracer(__name__)
        logger.info(f"AI Foundry tracing enabled for: {endpoint}")
        return True
        
    except ImportError as e:
        logger.warning(f"Required tracing packages not available: {e}")
        return False
    except Exception as e:
        logger.warning(f"AI Foundry tracing setup failed: {e}")
        import traceback
        logger.warning(traceback.format_exc())
        return False


def get_ai_foundry_openai_client():
    """Get the AI Foundry OpenAI client if available.
    
    Note: With AIProjectInstrumentor enabled, any AzureOpenAI client
    will automatically emit gen_ai.* attributes for Foundry tracing.
    """
    return None  # Use standard Azure OpenAI client - instrumentation handles tracing


def _setup_appinsights_tracing(connection_string: str):
    """Setup Application Insights tracing as fallback"""
    try:
        from azure.monitor.opentelemetry import configure_azure_monitor
        configure_azure_monitor(connection_string=connection_string)
        logger.info("Application Insights tracing configured")
        return True
    except Exception as e:
        logger.warning(f"Application Insights tracing failed: {e}")
        return False


def record_telemetry(event_name: str, properties: Dict[str, Any]):
    """Emit structured telemetry via logging and OpenTelemetry spans.
    
    This sends telemetry to Azure Monitor/App Insights which flows to AI Foundry portal.
    Uses customDimensions pattern for rich filtering in Application Analytics.
    """
    global _ai_foundry_tracer
    try:
        # Add application name prefix for better filtering
        full_event_name = f"{RAG_APP_NAME}.{event_name}"
        
        # Log with customDimensions format (picked up by Azure Monitor)
        # This follows the Microsoft recommended pattern for AI Foundry monitoring
        logger.info(
            full_event_name,
            extra={"customDimensions": {**properties, "app_name": RAG_APP_NAME, "service_name": SERVICE_NAME}}
        )
        
        # If AI Foundry tracer is available, also create a span
        if _ai_foundry_tracer:
            with _ai_foundry_tracer.start_as_current_span(full_event_name) as span:
                # Set semantic convention attributes
                span.set_attribute("service.name", SERVICE_NAME)
                span.set_attribute("ai.application.name", RAG_APP_NAME)
                span.set_attribute("gen_ai.operation.name", event_name)
                span.set_attribute("gen_ai.system", "azure_openai")
                
                for key, value in properties.items():
                    # Use proper semantic conventions where possible
                    if key == "model":
                        span.set_attribute("gen_ai.request.model", str(value))
                    elif key == "total_tokens":
                        span.set_attribute("gen_ai.usage.total_tokens", int(value) if value else 0)
                    elif key == "prompt_tokens":
                        span.set_attribute("gen_ai.usage.input_tokens", int(value) if value else 0)
                    elif key == "completion_tokens":
                        span.set_attribute("gen_ai.usage.output_tokens", int(value) if value else 0)
                    else:
                        span.set_attribute(key, str(value) if value is not None else "")
    except Exception as e:
        # Fallback to simple logging
        logger.info(f"telemetry:{event_name}:{properties}")


def record_evaluation_metrics(
    query: str,
    response: str, 
    context: str,
    scores: Dict[str, float],
    model: str,
    trace_id: str = None,
    response_id: str = None,
    method: str = "unknown"
):
    """Record evaluation metrics in the format expected by AI Foundry Monitoring.
    
    AI Foundry Monitoring expects EACH evaluation metric as a SEPARATE trace event with:
    - event.name = "gen_ai.evaluation.<metric_name>" (e.g., "gen_ai.evaluation.relevance")
    - gen_ai.evaluator.name = the metric name (e.g., "relevance")
    - gen_ai.evaluation.score = the numeric score value
    - gen_ai.response.id = to link evaluation to the original inference call
    
    This matches the Kusto query Foundry uses:
        traces | where event_name startswith "gen_ai.evaluation"
               | extend evaluator_name = customDimensions["gen_ai.evaluator.name"]
               | extend score = todouble(customDimensions["gen_ai.evaluation.score"])
    
    Args:
        query: The user's question
        response: The AI generated response
        context: The retrieved context documents
        scores: Dict with keys like 'relevance', 'coherence', 'groundedness' and numeric values
        model: The model used for evaluation
        trace_id: Optional trace ID to link evaluation to the original request
        response_id: Optional response ID to link evaluation to the inference call (gen_ai.response.id)
        method: Evaluation method used ('azure_ai_evaluation_sdk' or 'llm_prompting')
    """
    global _ai_foundry_tracer
    
    try:
        # Generate a response_id if not provided - this links evaluation to inference
        if not response_id:
            response_id = trace_id or str(uuid.uuid4())
        
        # Import OpenTelemetry trace API for emitting events
        from opentelemetry import trace as otel_trace
        
        # Get the current span to attach events to, or use the tracer to create one
        current_span = otel_trace.get_current_span()
        
        # Emit EACH evaluation metric as a SEPARATE trace event
        # This is the format AI Foundry Monitoring dashboard expects
        for metric_name, score_value in scores.items():
            # Event name must start with "gen_ai.evaluation" for Foundry to pick it up
            event_name = f"gen_ai.evaluation.{metric_name}"
            
            # Properties for the trace event - these become customDimensions in App Insights
            eval_attributes = {
                "event.name": event_name,
                "gen_ai.evaluator.name": metric_name,
                "gen_ai.evaluation.score": float(score_value) if score_value else 0.0,
                "gen_ai.response.id": response_id,
                "gen_ai.request.model": model,
                "gen_ai.system": "azure_openai",
                "gen_ai.operation.name": "evaluation",
                "evaluation.method": method,
                "app_name": RAG_APP_NAME,
                "service.name": SERVICE_NAME,
            }
            
            if trace_id:
                eval_attributes["gen_ai.trace.id"] = trace_id
            
            # Emit via OpenTelemetry tracer - this is the primary method for Foundry
            if _ai_foundry_tracer:
                # Create a span specifically for this evaluation metric
                # The span name should be the event_name so it appears correctly
                with _ai_foundry_tracer.start_as_current_span(event_name) as span:
                    # Set all attributes on the span - these become customDimensions
                    for attr_key, attr_value in eval_attributes.items():
                        span.set_attribute(attr_key, attr_value)
                    
                    # Also add as a span event for traces table
                    span.add_event(
                        event_name,
                        attributes=eval_attributes
                    )
            
            # Also add event to current span if one exists (for nested traces)
            if current_span and current_span.is_recording():
                current_span.add_event(event_name, attributes=eval_attributes)
            
            # Emit as a log record with structured data
            # This goes to the 'traces' table in App Insights where Foundry query looks
            # Use the special format that Azure Monitor understands
            log_extra = {
                "custom_dimensions": eval_attributes
            }
            
            # Create a logger specifically for this evaluation type
            eval_logger = logging.getLogger(f"gen_ai.evaluation.{metric_name}")
            eval_logger.info(
                event_name,  # This becomes the 'message' field
                extra=log_extra
            )
        
        logger.info(f"Recorded {len(scores)} evaluation metrics for response_id={response_id}")
                
    except Exception as e:
        logger.warning(f"Failed to record evaluation metrics: {e}")


class ConfigManager:
    """Centralized configuration management"""
    
    def __init__(self, env_file: str = ".env"):
        from dotenv import load_dotenv
        # override=True ensures .env values take precedence over system env vars
        load_dotenv(env_file, override=True)
        
        # Azure Search
        self.search_service_name = os.getenv("AZURE_SEARCH_SERVICE_NAME")
        self.search_api_key = os.getenv("AZURE_SEARCH_API_KEY")
        self.search_index_name = os.getenv("AZURE_SEARCH_INDEX_NAME", "documents-index")
        
        # Azure OpenAI
        self.openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
        self.openai_deployments = os.getenv("AZURE_OPENAI_DEPLOYMENTS", "").split(",")
        self.default_deployment = os.getenv("AZURE_OPENAI_DEFAULT_DEPLOYMENT", "gpt-4")
        
        # Embedding configuration
        self.embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
        self.embedding_dimensions = int(os.getenv("AZURE_OPENAI_EMBEDDING_DIMENSIONS", "1536"))
        
        # Document Intelligence
        self.doc_intel_endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        self.doc_intel_key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_API_KEY")
        
        # Cosmos DB
        self.cosmos_endpoint = os.getenv("COSMOS_DB_ENDPOINT")
        self.cosmos_key = os.getenv("COSMOS_DB_KEY")
        self.cosmos_database = os.getenv("COSMOS_DB_DATABASE_NAME", "ragfeedback")
        self.cosmos_container = os.getenv("COSMOS_DB_CONTAINER_NAME", "interactions")
        self.cosmos_use_aad = os.getenv("COSMOS_USE_AAD", "false").lower() == "true"
        
        # Azure Identity settings (for multi-account scenarios)
        self.azure_tenant_id = os.getenv("AZURE_TENANT_ID")
        
        # Data settings
        self.local_data_path = os.getenv("LOCAL_DATA_PATH", "./data")
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
        
        # Index and ingestion settings
        self.use_existing_index = os.getenv("USE_EXISTING_INDEX", "true").lower() == "true"
        self.auto_ingest = os.getenv("AUTO_INGEST", "false").lower() == "true"
        
        # Tracing settings
        self.enable_tracing = os.getenv("ENABLE_TRACING", "true").lower() == "true"
        self.lazy_tracing = os.getenv("LAZY_TRACING", "true").lower() == "true"
        
        # Application settings
        self.app_title = os.getenv("APP_TITLE", "RAG Chat Assistant")
        self.default_system_prompt = os.getenv("DEFAULT_SYSTEM_PROMPT", 
            "You are a helpful AI assistant that provides accurate information based on the retrieved documents.")
            
        # Setup tracing - prefer AI Foundry, fallback to Application Insights
        self.ai_foundry_endpoint = os.getenv("AZURE_AI_FOUNDRY_ENDPOINT")
        self.appinsights_connection_string = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
        
        self._tracing_initialized = False
        
        # Only setup tracing if enabled and not lazy
        if self.enable_tracing and not self.lazy_tracing:
            self._setup_tracing()
        elif self.enable_tracing and self.lazy_tracing:
            logger.info("Lazy tracing enabled - tracing will be initialized on first API call")
        else:
            logger.info("Tracing disabled via ENABLE_TRACING=false")
    
    def _setup_tracing(self):
        """Setup tracing (called immediately or lazily based on config)"""
        if self._tracing_initialized:
            return
            
        tracing_configured = False
        
        # Try AI Foundry tracing first (using endpoint)
        if self.ai_foundry_endpoint:
            tracing_configured = _setup_ai_foundry_tracing(
                self.ai_foundry_endpoint, 
                self.azure_tenant_id,
                self.appinsights_connection_string
            )
        
        # Fallback to Application Insights
        if not tracing_configured and self.appinsights_connection_string:
            tracing_configured = _setup_appinsights_tracing(self.appinsights_connection_string)
        
        if not tracing_configured:
            logger.info("No tracing configured - telemetry will only be logged locally")
        
        self._tracing_initialized = True
    
    def ensure_tracing(self):
        """Ensure tracing is initialized (call this before tracing-dependent operations)"""
        if self.enable_tracing and not self._tracing_initialized:
            self._setup_tracing()

class AzureSearchClient:
    """Azure AI Search client for document retrieval - Schema independent"""
    
    # Common field name patterns for auto-detection (order matters - more specific first)
    CONTENT_FIELD_PATTERNS = ['content_text', 'content', 'text', 'chunk', 'chunk_text', 'body', 'description', 'passage', 'document_content']
    TITLE_FIELD_PATTERNS = ['document_title', 'title', 'name', 'heading', 'subject', 'filename', 'file_name']
    VECTOR_FIELD_PATTERNS = ['content_vector', 'content_embedding', 'text_vector', 'vector', 'embedding', 'embeddings']
    URL_FIELD_PATTERNS = ['content_path', 'source_url', 'url', 'uri', 'link', 'path', 'filepath', 'file_path']
    ID_FIELD_PATTERNS = ['content_id', 'chunk_id', 'doc_id', 'document_id', 'id', 'key']
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.search_client = SearchClient(
            endpoint=f"https://{config.search_service_name}.search.windows.net",
            index_name=config.search_index_name,
            credential=AzureKeyCredential(config.search_api_key)
        )
        # Cache for detected field mappings per index
        self._field_cache = {}
        self._index_fields = None
        self._index_field_types = None
    
    def _get_index_fields(self) -> List[str]:
        """Get list of field names from the current index"""
        if self._index_fields is not None:
            return self._index_fields
        
        try:
            from azure.search.documents.indexes import SearchIndexClient
            index_client = SearchIndexClient(
                endpoint=f"https://{self.config.search_service_name}.search.windows.net",
                credential=AzureKeyCredential(self.config.search_api_key)
            )
            index = index_client.get_index(self.search_client._index_name)
            self._index_fields = [field.name for field in index.fields]
            # Store field types for smarter detection
            self._index_field_types = {field.name: str(field.type) for field in index.fields}
            logger.info(f"Index '{self.search_client._index_name}' fields: {self._index_fields}")
            return self._index_fields
        except Exception as e:
            logger.warning(f"Could not retrieve index schema: {e}")
            return []
    
    def _detect_field(self, patterns: List[str], fields: List[str], exclude_types: List[str] = None) -> Optional[str]:
        """Detect a field from the index that matches one of the patterns.
        
        Args:
            patterns: List of field name patterns to match (in priority order)
            fields: List of actual field names in the index
            exclude_types: Field types to exclude (e.g., vector fields for content)
        """
        fields_lower = {f.lower(): f for f in fields}
        
        # Filter out excluded types if we have type info
        if exclude_types and self._index_field_types:
            fields_lower = {
                f_lower: f_orig for f_lower, f_orig in fields_lower.items()
                if not any(excl in self._index_field_types.get(f_orig, '').lower() for excl in exclude_types)
            }
        
        # First pass: exact matches (case-insensitive)
        for pattern in patterns:
            if pattern.lower() in fields_lower:
                return fields_lower[pattern.lower()]
        
        # Second pass: partial matching (pattern is substring of field name)
        for pattern in patterns:
            for field_lower, field_orig in fields_lower.items():
                if pattern.lower() in field_lower:
                    return field_orig
        
        return None
    
    def _get_field_mappings(self) -> Dict[str, Optional[str]]:
        """Get field mappings for current index, auto-detecting if needed"""
        index_name = self.search_client._index_name
        if index_name in self._field_cache:
            return self._field_cache[index_name]
        
        fields = self._get_index_fields()
        
        # Detect content field - exclude vector types
        content_field = self._detect_field(
            self.CONTENT_FIELD_PATTERNS, fields, 
            exclude_types=['vector', 'collection']
        )
        
        mappings = {
            'content': content_field,
            'title': self._detect_field(self.TITLE_FIELD_PATTERNS, fields, exclude_types=['vector', 'collection']),
            'vector': self._detect_field(self.VECTOR_FIELD_PATTERNS, fields),
            'url': self._detect_field(self.URL_FIELD_PATTERNS, fields, exclude_types=['vector', 'collection']),
            'id': self._detect_field(self.ID_FIELD_PATTERNS, fields, exclude_types=['vector', 'collection']),
        }
        
        logger.info(f"Field mappings for '{index_name}': {mappings}")
        self._field_cache[index_name] = mappings
        return mappings
    
    def _extract_document(self, result: Dict[str, Any], mappings: Dict[str, Optional[str]]) -> Dict[str, Any]:
        """Extract document data using detected field mappings"""
        content_field = mappings.get('content')
        title_field = mappings.get('title')
        url_field = mappings.get('url')
        id_field = mappings.get('id')
        
        # Get content - try mapped field first, then fallback to finding any text
        content = ''
        if content_field and content_field in result:
            content = result.get(content_field, '')
        else:
            # Fallback: find the longest string field value
            for key, value in result.items():
                if isinstance(value, str) and len(value) > len(content) and not key.startswith('@'):
                    content = value
        
        # Get title
        title = ''
        if title_field and title_field in result:
            title = result.get(title_field, '')
        
        # Get URL/source
        url = ''
        if url_field and url_field in result:
            url = result.get(url_field, '')
        
        # Get ID
        doc_id = ''
        if id_field and id_field in result:
            doc_id = result.get(id_field, '')
        
        return {
            'content': content,
            'title': title,
            'score': result.get('@search.score', 0.0),
            'url': url,
            'metadata': {
                'chunk_id': doc_id,
                'source': url or result.get('source', ''),
                'raw_fields': {k: v for k, v in result.items() if not k.startswith('@') and not isinstance(v, list)}
            }
        }
    
    def search_documents(self, query: str, top_k: int = 5, use_semantic_ranking: bool = False) -> List[Dict[str, Any]]:
        """Search for relevant documents - schema independent
        
        Note: use_semantic_ranking defaults to False because free-tier Azure Search 
        doesn't support semantic search. Set to True only if you have Standard tier or higher.
        """
        try:
            # Get field mappings for current index
            mappings = self._get_field_mappings()
            
            search_params = {
                'search_text': query,
                'top': top_k,
                'include_total_count': True
            }
            
            if use_semantic_ranking:
                search_params.update({
                    'query_type': 'semantic',
                    'semantic_configuration_name': 'default',
                    'query_caption': 'extractive',
                    'query_answer': 'extractive'
                })
            
            results = list(self.search_client.search(**search_params))
            
            documents = []
            for result in results:
                doc = self._extract_document(result, mappings)
                if use_semantic_ranking and result.get('@search.captions'):
                    doc['captions'] = [caption.text for caption in result['@search.captions']]
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def vector_search(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search using vector similarity - schema independent"""
        from azure.search.documents.models import VectorizedQuery
        
        try:
            # Get field mappings and detect vector field
            mappings = self._get_field_mappings()
            vector_field = mappings.get('vector', 'content_vector')
            
            if not vector_field:
                logger.warning("No vector field detected in index - falling back to text search")
                return []
            
            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=top_k,
                fields=vector_field
            )
            
            results = list(self.search_client.search(
                search_text=None,
                vector_queries=[vector_query],
                top=top_k,
                include_total_count=True
            ))
            
            documents = []
            for result in results:
                doc = self._extract_document(result, mappings)
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []
    
    def hybrid_search(self, query: str, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Hybrid search combining text and vector search - schema independent"""
        from azure.search.documents.models import VectorizedQuery
        
        try:
            # Get field mappings
            mappings = self._get_field_mappings()
            vector_field = mappings.get('vector', 'content_vector')
            
            if not vector_field:
                logger.warning("No vector field detected - using text search only")
                return self.search_documents(query, top_k)
            
            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=top_k,
                fields=vector_field
            )
            
            results = list(self.search_client.search(
                search_text=query,
                vector_queries=[vector_query],
                top=top_k,
                include_total_count=True
            ))
            
            documents = []
            for result in results:
                doc = self._extract_document(result, mappings)
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return []
    
    def clear_field_cache(self):
        """Clear cached field mappings (useful when switching indexes)"""
        self._field_cache = {}
        self._index_fields = None


class EmbeddingClient:
    """Azure OpenAI client for generating embeddings"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        
        # Use AAD auth with credential fallback for AI Foundry
        # AIProjectInstrumentor (if enabled) will automatically add gen_ai.* attributes
        credential = _get_credential(config.azure_tenant_id)
        
        self.client = AzureOpenAI(
            azure_endpoint=config.openai_endpoint,
            azure_ad_token_provider=lambda: credential.get_token("https://cognitiveservices.azure.com/.default").token,
            api_version=config.openai_api_version
        )
        logger.info(f"EmbeddingClient initialized - endpoint: {config.openai_endpoint}")
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.config.embedding_deployment
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return []
    
    def get_embeddings(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        """Generate embeddings for multiple texts in batches"""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.config.embedding_deployment
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Error generating batch embeddings: {e}")
                # Return empty vectors for failed batch
                all_embeddings.extend([[] for _ in batch])
        
        return all_embeddings


class DocumentProcessor:
    """Document processing using Azure Document Intelligence"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.client = DocumentAnalysisClient(
            endpoint=config.doc_intel_endpoint,
            credential=AzureKeyCredential(config.doc_intel_key)
        )
    
    def extract_text_from_document(self, file_path: str) -> str:
        """Extract text from document using Azure Document Intelligence"""
        try:
            with open(file_path, "rb") as f:
                poller = self.client.begin_analyze_document("prebuilt-layout", document=f)
                result = poller.result()
            
            content = ""
            for page in result.pages:
                for line in page.lines:
                    content += line.content + "\n"
            
            return content
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            return ""
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks for indexing"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.config.chunk_size - self.config.chunk_overlap):
            chunk_words = words[i:i + self.config.chunk_size]
            chunk = ' '.join(chunk_words)
            chunks.append(chunk)
        
        return chunks

class CosmosDBClient:
    """Cosmos DB client for storing feedback and interactions"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        if config.cosmos_use_aad:
            # Use tenant_id if specified to target the correct Azure account with fallback
            credential = _get_credential(config.azure_tenant_id)
            self.client = CosmosClient(config.cosmos_endpoint, credential)
        else:
            self.client = CosmosClient(config.cosmos_endpoint, config.cosmos_key)
        self.database = self.client.get_database_client(config.cosmos_database)
        self.container = self.database.get_container_client(config.cosmos_container)
    
    def create_database_and_container(self):
        """Create database and container if they don't exist"""
        try:
            # Create database
            try:
                database = self.client.create_database_if_not_exists(id=self.config.cosmos_database)
            except Exception as e:
                logger.info(f"Database might already exist: {e}")
            
            # Create container
            try:
                container = self.database.create_container_if_not_exists(
                    id=self.config.cosmos_container,
                    partition_key=PartitionKey(path="/session_id"),
                    offer_throughput=400
                )
            except Exception as e:
                logger.info(f"Container might already exist: {e}")
                
        except Exception as e:
            logger.error(f"Error creating database/container: {e}")
    
    def store_interaction(self, session_id: str, user_query: str, retrieved_docs: List[Dict], 
                         ai_response: str, model_used: str, feedback: Optional[str] = None) -> str:
        """Store chat interaction in Cosmos DB"""
        try:
            # Create a lightweight version of retrieved docs for storage
            stored_docs = []
            for doc in retrieved_docs:
                stored_docs.append({
                    'title': doc.get('title'),
                    'url': doc.get('url'),
                    'score': doc.get('score'),
                    'chunk_id': doc.get('metadata', {}).get('chunk_id'),
                    # Truncate content to save space
                    'content_snippet': doc.get('content', '')[:200] + "..."
                })

            interaction_id = str(uuid.uuid4())
            item = {
                'id': interaction_id,
                'session_id': session_id,
                'timestamp': datetime.utcnow().isoformat(),
                'user_query': user_query,
                'retrieved_documents': stored_docs,
                'ai_response': ai_response,
                'model_used': model_used,
                'feedback': feedback,
                'created_at': datetime.utcnow().isoformat()
            }
            
            self.container.create_item(body=item)
            return interaction_id
            
        except Exception as e:
            logger.error(f"Error storing interaction: {e}")
            return ""
    
    def update_feedback(self, interaction_id: str, feedback: str):
        """Update feedback for an interaction"""
        try:
            # Read the item first
            items = list(self.container.query_items(
                query="SELECT * FROM c WHERE c.id = @interaction_id",
                parameters=[{"name": "@interaction_id", "value": interaction_id}]
            ))
            
            if items:
                item = items[0]
                item['feedback'] = feedback
                item['feedback_timestamp'] = datetime.utcnow().isoformat()
                self.container.replace_item(item=item['id'], body=item)
                
        except Exception as e:
            logger.error(f"Error updating feedback: {e}")

class OpenAIClient:
    """Azure OpenAI client for chat completions"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        
        # Use AAD auth with credential fallback for AI Foundry
        # AIProjectInstrumentor (if enabled) will automatically add gen_ai.* attributes
        credential = _get_credential(config.azure_tenant_id)
        
        self.client = AzureOpenAI(
            azure_endpoint=config.openai_endpoint,
            azure_ad_token_provider=lambda: credential.get_token("https://cognitiveservices.azure.com/.default").token,
            api_version=config.openai_api_version
        )
        logger.info(f"OpenAIClient initialized - endpoint: {config.openai_endpoint}")
    
    def generate_response(self, query: str, retrieved_docs: List[Dict], 
                         model: str, system_prompt: str = None) -> Tuple[str, Dict]:
        """Generate response using Azure OpenAI"""
        # Ensure tracing is initialized (lazy initialization)
        self.config.ensure_tracing()
        
        try:
            # Prepare context from retrieved documents
            context = "\n\n".join([
                f"Document {i+1}: {doc['content'][:500]}..." if len(doc['content']) > 500 
                else f"Document {i+1}: {doc['content']}"
                for i, doc in enumerate(retrieved_docs)
            ])
            
            # Use default system prompt if not provided
            if not system_prompt:
                system_prompt = self.config.default_system_prompt
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nPlease provide a comprehensive answer based on the context provided."}
            ]
            
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=1500
            )
            
            usage_info = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }

            record_telemetry("chat_completion", {
                "model": model,
                "total_tokens": usage_info.get('total_tokens'),
                "prompt_tokens": usage_info.get('prompt_tokens'),
                "completion_tokens": usage_info.get('completion_tokens')
            })
            
            return response.choices[0].message.content, usage_info
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}", {}

class AgenticRetrieval:
    """Agentic retrieval using multiple search strategies"""
    
    def __init__(self, search_client: AzureSearchClient, openai_client: OpenAIClient):
        self.search_client = search_client
        self.openai_client = openai_client
    
    def agentic_search(self, query: str, model: str) -> List[Dict[str, Any]]:
        """Perform agentic retrieval with query rewriting and multi-step search"""
        try:
            # Step 1: Generate alternative queries
            query_generation_prompt = f"""
            Given the user query: "{query}"
            
            Generate 3 alternative ways to search for this information:
            1. A more specific version
            2. A broader version  
            3. A version using different keywords
            
            Return only the alternative queries, one per line, without numbering or explanation.
            """
            
            messages = [{"role": "user", "content": query_generation_prompt}]
            response = self.openai_client.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
                max_tokens=300
            )
            
            alternative_queries = response.choices[0].message.content.strip().split('\n')
            all_queries = [query] + [q.strip() for q in alternative_queries if q.strip()]
            
            # Step 2: Search with all queries
            all_docs = []
            seen_content = set()
            
            for search_query in all_queries:
                docs = self.search_client.search_documents(search_query, top_k=3)
                for doc in docs:
                    content_hash = hash(doc['content'])
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        all_docs.append(doc)
            
            # Step 3: Rank and select top documents
            top_docs = sorted(all_docs, key=lambda x: x['score'], reverse=True)[:5]
            record_telemetry("agentic_retrieval", {
                "query": query,
                "generated_queries": len(all_queries),
                "retrieved_docs": len(top_docs)
            })
            return top_docs
            
        except Exception as e:
            logger.error(f"Error in agentic search: {e}")
            # Fallback to normal search
            return self.search_client.search_documents(query, top_k=5)

class EvaluationService:
    """Service for evaluating RAG responses using Azure AI Evaluation SDK.
    
    Uses built-in evaluators from azure-ai-evaluation for:
    - Relevance: How well the response addresses the query
    - Groundedness: How well the response is grounded in the provided context
    - Coherence: How logically coherent and well-structured the response is
    """
    
    def __init__(self, openai_client: OpenAIClient, config: ConfigManager = None):
        self.openai_client = openai_client
        self.config = config
        self._evaluators_available = False
        self._model_config = None
        self._relevance_evaluator = None
        self._groundedness_evaluator = None
        self._coherence_evaluator = None
        
        # Try to initialize Azure AI Evaluation SDK evaluators
        self._init_evaluators()
    
    def _init_evaluators(self):
        """Initialize Azure AI Evaluation SDK evaluators."""
        try:
            from azure.ai.evaluation import (
                RelevanceEvaluator,
                GroundednessEvaluator,
                CoherenceEvaluator,
                AzureOpenAIModelConfiguration,
            )
            
            if self.config:
                # Create model configuration for evaluators
                # Use the same Azure OpenAI endpoint for evaluation with credential fallback
                credential = _get_credential(self.config.azure_tenant_id)
                
                self._model_config = AzureOpenAIModelConfiguration(
                    azure_endpoint=self.config.openai_endpoint,
                    azure_deployment=self.config.default_deployment,
                    api_version=self.config.openai_api_version,
                    credential=credential,
                )
                
                # Initialize evaluators with model configuration
                self._relevance_evaluator = RelevanceEvaluator(model_config=self._model_config)
                self._groundedness_evaluator = GroundednessEvaluator(model_config=self._model_config)
                self._coherence_evaluator = CoherenceEvaluator(model_config=self._model_config)
                
                self._evaluators_available = True
                logger.info("Azure AI Evaluation SDK evaluators initialized successfully")
            else:
                logger.warning("No config provided, using LLM-based fallback evaluation")
                
        except ImportError as e:
            logger.warning(f"Azure AI Evaluation SDK not available: {e}")
            logger.info("Install with: pip install azure-ai-evaluation")
        except Exception as e:
            logger.warning(f"Failed to initialize evaluators: {e}")
            import traceback
            logger.warning(traceback.format_exc())
    
    def evaluate_response(self, query: str, retrieved_docs: List[Dict], 
                         response: str, model: str) -> Dict[str, float]:
        """Evaluate response for relevance, coherence, and groundedness.
        
        Uses Azure AI Evaluation SDK if available, otherwise falls back to LLM-based evaluation.
        """
        # Prepare context
        context = "\n".join([doc['content'] for doc in retrieved_docs])
        
        if self._evaluators_available:
            return self._evaluate_with_sdk(query, context, response, model)
        else:
            return self._evaluate_with_llm(query, context, response, model)
    
    def _evaluate_with_sdk(self, query: str, context: str, response: str, model: str) -> Dict[str, float]:
        """Evaluate using Azure AI Evaluation SDK built-in evaluators."""
        scores = {}
        
        try:
            # Run evaluators in a traced span
            with rag_trace_span("evaluation", {"model": model, "method": "azure_ai_evaluation_sdk"}):
                # Relevance evaluation
                try:
                    relevance_result = self._relevance_evaluator(
                        query=query,
                        response=response,
                        context=context
                    )
                    scores['relevance'] = float(relevance_result.get('relevance', 0))
                except Exception as e:
                    logger.warning(f"Relevance evaluation failed: {e}")
                    scores['relevance'] = 0.0
                
                # Groundedness evaluation
                try:
                    groundedness_result = self._groundedness_evaluator(
                        query=query,
                        response=response,
                        context=context
                    )
                    scores['groundedness'] = float(groundedness_result.get('groundedness', 0))
                except Exception as e:
                    logger.warning(f"Groundedness evaluation failed: {e}")
                    scores['groundedness'] = 0.0
                
                # Coherence evaluation
                try:
                    coherence_result = self._coherence_evaluator(
                        query=query,
                        response=response
                    )
                    scores['coherence'] = float(coherence_result.get('coherence', 0))
                except Exception as e:
                    logger.warning(f"Coherence evaluation failed: {e}")
                    scores['coherence'] = 0.0
                
                # Record evaluation metrics using the proper AI Foundry format
                record_evaluation_metrics(
                    query=query,
                    response=response,
                    context=context,
                    scores=scores,
                    model=model,
                    method="azure_ai_evaluation_sdk"
                )
                
            return scores
            
        except Exception as e:
            logger.error(f"SDK evaluation failed, falling back to LLM: {e}")
            return self._evaluate_with_llm(query, context, response, model)
    
    def _evaluate_with_llm(self, query: str, context: str, response: str, model: str) -> Dict[str, float]:
        """Fallback: Evaluate using LLM-based prompting."""
        try:
            with rag_trace_span("evaluation", {"model": model, "method": "llm_prompting"}):
                evaluation_prompt = f"""
                Evaluate the following AI response across three dimensions. Rate each on a scale of 1-5 where 5 is excellent and 1 is poor.

                USER QUERY: {query}
                
                RETRIEVED CONTEXT: {context}
                
                AI RESPONSE: {response}
                
                Please evaluate:
                1. RELEVANCE: How well does the response answer the user's question?
                2. COHERENCE: How clear, logical, and well-structured is the response?
                3. GROUNDEDNESS: How well is the response supported by the retrieved documents?
                
                Respond in this exact format:
                RELEVANCE: [score]
                COHERENCE: [score] 
                GROUNDEDNESS: [score]
                """
                
                messages = [{"role": "user", "content": evaluation_prompt}]
                response_eval = self.openai_client.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=300
                )
                
                eval_text = response_eval.choices[0].message.content
                
                # Parse scores
                scores = {}
                for line in eval_text.split('\n'):
                    if ':' in line:
                        metric, score_str = line.split(':', 1)
                        metric = metric.strip().lower()
                        try:
                            score = float(score_str.strip())
                            scores[metric] = score
                        except ValueError:
                            continue

                # Record evaluation metrics using the proper AI Foundry format
                record_evaluation_metrics(
                    query=query,
                    response=response,
                    context=context,
                    scores=scores,
                    model=model,
                    method="llm_prompting"
                )
                
                return scores
                
        except Exception as e:
            logger.error(f"Error evaluating response: {e}")
            return {'relevance': 0.0, 'coherence': 0.0, 'groundedness': 0.0}

# Data ingestion utilities
def ingest_documents(config: ConfigManager, search_client: AzureSearchClient, 
                    doc_processor: DocumentProcessor, embedding_client: 'EmbeddingClient' = None):
    """Ingest documents from local path to Azure Search with optional vector embeddings"""
    import os
    
    data_path = config.local_data_path
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path {data_path} does not exist")
    
    documents_to_index = []
    processed_files = []
    all_chunks = []
    chunk_metadata = []
    
    for filename in os.listdir(data_path):
        if filename.lower().endswith(('.pdf', '.docx', '.txt', '.jpg', '.png')):
            file_path = os.path.join(data_path, filename)
            logger.info(f"Processing {filename}")
            
            # Extract text
            if filename.lower().endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            else:
                content = doc_processor.extract_text_from_document(file_path)
            
            if not content or len(content.strip()) == 0:
                logger.warning(f"No content extracted from {filename}")
                continue
            
            # Chunk text
            chunks = doc_processor.chunk_text(content)
            logger.info(f"Created {len(chunks)} chunks from {filename}")
            
            # Store chunks and metadata for batch embedding
            for i, chunk in enumerate(chunks):
                doc_id = f"{filename.replace('.', '_')}_{i}"
                all_chunks.append(chunk)
                chunk_metadata.append({
                    'id': doc_id,
                    'title': filename,
                    'content': chunk,
                    'source': filename,
                    'chunk_id': str(i),
                    'url': file_path
                })
            
            processed_files.append(filename)
    
    # Generate embeddings if embedding client is provided
    if embedding_client and all_chunks:
        logger.info(f"Generating embeddings for {len(all_chunks)} chunks...")
        embeddings = embedding_client.get_embeddings(all_chunks)
        
        # Add embeddings to documents
        for i, metadata in enumerate(chunk_metadata):
            document = metadata.copy()
            if i < len(embeddings) and embeddings[i]:
                document['content_vector'] = embeddings[i]
            documents_to_index.append(document)
        
        logger.info(f"Generated {len(embeddings)} embeddings")
    else:
        # No embedding client, just use text
        documents_to_index = chunk_metadata
        if all_chunks:
            logger.warning("No embedding client provided - documents will be indexed without vectors")
    
    # Upload to Azure Search
    if documents_to_index:
        result = search_client.search_client.upload_documents(documents=documents_to_index)
        succeeded = sum(1 for r in result if r.succeeded)
        failed = len(result) - succeeded
        logger.info(f"Indexed {succeeded} chunks, {failed} failed")
        if failed > 0:
            logger.warning(f"Some documents failed to index")
        return {'processed_files': processed_files, 'chunks': len(documents_to_index), 'succeeded': succeeded, 'failed': failed}
    else:
        logger.warning("No documents found to index")
        return {'processed_files': [], 'chunks': 0, 'succeeded': 0, 'failed': 0}