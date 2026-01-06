import streamlit as st
import uuid
from typing import Dict, List, Any

from rag_services import (
    ConfigManager, 
    AzureSearchClient, 
    DocumentProcessor, 
    CosmosDBClient, 
    OpenAIClient,
    EmbeddingClient,
    AgenticRetrieval,
    EvaluationService,
    ingest_documents,
    rag_trace_span
)

# Page configuration
st.set_page_config(
    page_title="RAG Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def initialize_services():
    """Initialize all services with caching.
    
    Note: With LAZY_TRACING=true, tracing setup is deferred until first API call.
    This speeds up initial page load significantly.
    """
    try:
        config = ConfigManager()
        search_client = AzureSearchClient(config)
        doc_processor = DocumentProcessor(config)
        cosmos_client = CosmosDBClient(config)
        openai_client = OpenAIClient(config)
        embedding_client = EmbeddingClient(config)
        agentic_retrieval = AgenticRetrieval(search_client, openai_client)
        # Pass config to EvaluationService for Azure AI Evaluation SDK
        evaluation_service = EvaluationService(openai_client, config)
        
        # Create Cosmos DB database and container if they don't exist
        cosmos_client.create_database_and_container()
        
        # Auto-ingest if configured (USE_EXISTING_INDEX=false and AUTO_INGEST=true)
        if not config.use_existing_index and config.auto_ingest:
            try:
                from rag_services import ingest_documents
                result = ingest_documents(config, search_client, doc_processor, embedding_client)
                if result and result['chunks'] > 0:
                    st.info(f"Auto-ingested {result['chunks']} chunks from {len(result['processed_files'])} files")
            except Exception as e:
                st.warning(f"Auto-ingest failed: {e}")
        
        return {
            'config': config,
            'search_client': search_client,
            'doc_processor': doc_processor,
            'cosmos_client': cosmos_client,
            'openai_client': openai_client,
            'embedding_client': embedding_client,
            'agentic_retrieval': agentic_retrieval,
            'evaluation_service': evaluation_service
        }
    except Exception as e:
        st.error(f"Error initializing services: {e}")
        return None

def initialize_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if 'interaction_ids' not in st.session_state:
        st.session_state.interaction_ids = {}

def display_chat_history(services):
    """Display chat history"""
    for i, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])
                
                # Display retrieved documents if available
                if "retrieved_docs" in message:
                    with st.expander("📚 Source Documents"):
                        for j, doc in enumerate(message["retrieved_docs"]):
                            st.markdown(f"**Document {j+1}** (Score: {doc['score']:.3f})")
                            st.markdown(f"**Title:** {doc['title']}")
                            st.markdown(f"**Content:** {doc['content'][:300]}...")
                            if doc.get('captions'):
                                st.markdown(f"**Relevant excerpts:** {', '.join(doc['captions'])}")
                            st.markdown("---")
                
                # Display evaluation scores if available
                if "evaluation" in message:
                    with st.expander("📊 Response Evaluation"):
                        eval_scores = message["evaluation"]
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Relevance", f"{eval_scores.get('relevance', 0):.1f}/5")
                        with col2:
                            st.metric("Coherence", f"{eval_scores.get('coherence', 0):.1f}/5")
                        with col3:
                            st.metric("Groundedness", f"{eval_scores.get('groundedness', 0):.1f}/5")
                
                # Feedback buttons
                interaction_id = st.session_state.interaction_ids.get(i)
                if interaction_id:
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        if st.button("👍", key=f"thumbs_up_{i}", help="Thumbs up"):
                            services['cosmos_client'].update_feedback(interaction_id, "thumbs_up")
                            st.success("Thank you for your feedback!")
                    with col2:
                        if st.button("👎", key=f"thumbs_down_{i}", help="Thumbs down"):
                            services['cosmos_client'].update_feedback(interaction_id, "thumbs_down")
                            st.success("Thank you for your feedback!")

def sidebar_controls(services):
    """Sidebar controls for configuration"""
    st.sidebar.title("⚙️ Configuration")
    
    # Model selection
    available_models = [model.strip() for model in services['config'].openai_deployments if model.strip()]
    if not available_models:
        available_models = [services['config'].default_deployment]
    
    selected_model = st.sidebar.selectbox(
        "Select Model",
        options=available_models,
        index=0 if services['config'].default_deployment not in available_models 
        else available_models.index(services['config'].default_deployment)
    )

    rewrite_model = st.sidebar.selectbox(
        "Rewrite / Agentic Model",
        options=available_models,
        index=available_models.index(selected_model) if selected_model in available_models else 0,
        help="Use a lighter model for query rewriting/agentic steps"
    )

    evaluation_model = st.sidebar.selectbox(
        "Evaluation Model",
        options=available_models,
        index=available_models.index(selected_model) if selected_model in available_models else 0,
        help="Use a lighter model for evaluations to save cost"
    )
    
    # Retrieval settings
    st.sidebar.subheader("🔍 Retrieval Settings")
    use_agentic = st.sidebar.toggle("Enable Agentic Retrieval", value=False)
    
    search_type = st.sidebar.selectbox(
        "Search Type",
        options=["keyword", "vector", "hybrid"],
        index=2,  # Default to hybrid
        help="Keyword: text search, Vector: embedding similarity, Hybrid: combines both"
    )
    
    top_k = st.sidebar.slider("Number of documents to retrieve", min_value=1, max_value=10, value=5)
    
    # Evaluation settings
    st.sidebar.subheader("📊 Evaluation Settings")
    enable_evaluation = st.sidebar.toggle("Enable Real-time Evaluation", value=True)
    
    # System prompt customization
    st.sidebar.subheader("💬 System Prompt")
    custom_prompt = st.sidebar.text_area(
        "System Prompt",
        value=services['config'].default_system_prompt,
        height=100
    )
    
    return {
        'selected_model': selected_model,
        'rewrite_model': rewrite_model,
        'evaluation_model': evaluation_model,
        'use_agentic': use_agentic,
        'search_type': search_type,
        'top_k': top_k,
        'enable_evaluation': enable_evaluation,
        'custom_prompt': custom_prompt
    }

def admin_panel(services):
    """Admin panel for data management and index configuration"""
    st.sidebar.subheader("🛠️ Admin Panel")
    
    # Index Configuration Section
    with st.sidebar.expander("📑 Index Configuration", expanded=False):
        config = services['config']
        
        # Initialize session state for index name if not set
        if 'custom_index_name' not in st.session_state:
            st.session_state.custom_index_name = config.search_index_name
        
        # Editable index name
        new_index_name = st.text_input(
            "Index Name",
            value=st.session_state.custom_index_name,
            help="Enter the Azure AI Search index name to use for retrieval"
        )
        
        # Apply index name change
        if new_index_name != st.session_state.custom_index_name:
            st.session_state.custom_index_name = new_index_name
            # Update the search client with new index
            try:
                from azure.search.documents import SearchClient
                from azure.core.credentials import AzureKeyCredential
                services['search_client'].search_client = SearchClient(
                    endpoint=f"https://{config.search_service_name}.search.windows.net",
                    index_name=new_index_name,
                    credential=AzureKeyCredential(config.search_api_key)
                )
                services['config'].search_index_name = new_index_name
                # Clear field cache so schema is re-detected for new index
                services['search_client'].clear_field_cache()
                st.success(f"✅ Switched to index: **{new_index_name}**")
            except Exception as e:
                st.error(f"Failed to switch index: {e}")
        
        # Show current mode
        mode = "Using Existing Index" if config.use_existing_index else "Create/Update Index"
        st.info(f"**Mode:** {mode}")
        
        # Show data path for ingestion
        st.text_input(
            "Local Data Path",
            value=config.local_data_path,
            disabled=True,
            help="Set in .env as LOCAL_DATA_PATH"
        )
        
        st.caption("💡 Index name can be changed above. Data path requires .env edit.")
    
    # Ingestion Section
    with st.sidebar.expander("📁 Document Ingestion", expanded=False):
        st.caption("Ingest documents from local path to search index")
        
        if st.button("🔄 Ingest Documents", key="ingest_btn"):
            with st.spinner("Ingesting documents with embeddings..."):
                try:
                    result = ingest_documents(
                        services['config'],
                        services['search_client'],
                        services['doc_processor'],
                        services['embedding_client']
                    )
                    if result and result['chunks'] > 0:
                        st.success(f"✅ Ingested {result['chunks']} chunks from {len(result['processed_files'])} files!")
                        st.info(f"Files: {', '.join(result['processed_files'])}")
                    else:
                        st.warning("No documents found. Add files to the data folder.")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        # Show ingestion settings
        st.caption(f"Chunk size: {config.chunk_size}, Overlap: {config.chunk_overlap}")
    
    # Chat Management
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.interaction_ids = {}
        st.rerun()
    
    # Display system info
    with st.sidebar.expander("ℹ️ System Information"):
        st.write(f"**Session ID:** {st.session_state.session_id[:8]}...")
        # Show the active index (may differ from .env if changed at runtime)
        active_index = st.session_state.get('custom_index_name', services['config'].search_index_name)
        st.write(f"**Active Index:** {active_index}")
        st.write(f"**Default Index:** {services['config'].search_index_name}")
        st.write(f"**Index Mode:** {'Existing' if services['config'].use_existing_index else 'Create/Update'}")
        st.write(f"**Data Path:** {services['config'].local_data_path}")
        st.write(f"**Tracing:** {'Enabled' if services['config'].enable_tracing else 'Disabled'}")
        st.write(f"**Lazy Tracing:** {'Yes' if services['config'].lazy_tracing else 'No'}")
        st.write(f"**Total Messages:** {len(st.session_state.messages)}")

def main():
    """Main application"""
    st.title("🤖 RAG Chat Assistant")
    st.markdown("Welcome to the RAG-powered chat assistant with Azure AI services!")
    
    # Initialize services
    services = initialize_services()
    if not services:
        st.error("Failed to initialize services. Please check your configuration.")
        return
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar controls
    settings = sidebar_controls(services)
    
    # Admin panel
    admin_panel(services)
    
    # Main chat interface
    st.subheader("💬 Chat")
    
    # Display chat history
    display_chat_history(services)
    
    # Chat input
    if user_query := st.chat_input("Ask me anything..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_query)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Wrap entire RAG flow in a single trace span for visibility
                with rag_trace_span("chat_query", {
                    "session_id": st.session_state.session_id,
                    "search_type": settings['search_type'],
                    "model": settings['selected_model'],
                    "use_agentic": settings['use_agentic']
                }) as span:
                    try:
                        # Retrieve documents based on search type
                        if settings['use_agentic']:
                            retrieved_docs = services['agentic_retrieval'].agentic_search(
                                user_query, settings['rewrite_model']
                            )
                        else:
                            search_type = settings['search_type']
                            
                            if search_type == 'vector':
                                # Vector-only search
                                query_vector = services['embedding_client'].get_embedding(user_query)
                                if query_vector:
                                    retrieved_docs = services['search_client'].vector_search(
                                        query_vector, 
                                        top_k=settings['top_k']
                                    )
                                else:
                                    st.warning("Failed to generate embedding, falling back to keyword search")
                                    retrieved_docs = services['search_client'].search_documents(
                                        user_query, 
                                        top_k=settings['top_k'],
                                        use_semantic_ranking=False
                                    )
                            elif search_type == 'hybrid':
                                # Hybrid search (text + vector)
                                query_vector = services['embedding_client'].get_embedding(user_query)
                                if query_vector:
                                    retrieved_docs = services['search_client'].hybrid_search(
                                        user_query,
                                        query_vector, 
                                        top_k=settings['top_k']
                                    )
                                else:
                                    st.warning("Failed to generate embedding, falling back to keyword search")
                                    retrieved_docs = services['search_client'].search_documents(
                                        user_query, 
                                        top_k=settings['top_k'],
                                        use_semantic_ranking=False
                                    )
                            else:
                                # Keyword search
                                retrieved_docs = services['search_client'].search_documents(
                                    user_query, 
                                    top_k=settings['top_k'],
                                    use_semantic_ranking=False
                                )
                        
                        # Add retrieval info to span
                        if span:
                            span.set_attribute("docs_retrieved", len(retrieved_docs) if retrieved_docs else 0)
                        
                        if not retrieved_docs:
                            st.warning("No relevant documents found. Please try a different query.")
                            response = "I couldn't find any relevant documents to answer your question. Please try rephrasing your query or check if documents have been ingested."
                            evaluation_scores = {}
                        else:
                            # Generate response
                            response, usage_info = services['openai_client'].generate_response(
                                user_query,
                                retrieved_docs,
                                settings['selected_model'],
                                settings['custom_prompt']
                            )
                            
                            # Add token usage to span
                            if span and usage_info:
                                span.set_attribute("total_tokens", usage_info.get('total_tokens', 0))
                            
                            # Evaluate response
                            if settings['enable_evaluation']:
                                evaluation_scores = services['evaluation_service'].evaluate_response(
                                    user_query,
                                    retrieved_docs,
                                    response,
                                    settings['evaluation_model']
                                )
                                # Add evaluation scores to span
                                if span and evaluation_scores:
                                    span.set_attribute("eval.relevance", evaluation_scores.get('relevance', 0))
                                    span.set_attribute("eval.coherence", evaluation_scores.get('coherence', 0))
                                    span.set_attribute("eval.groundedness", evaluation_scores.get('groundedness', 0))
                            else:
                                evaluation_scores = {}
                        
                        # Display response
                        st.write(response)
                        
                        # Store interaction in Cosmos DB
                        interaction_id = services['cosmos_client'].store_interaction(
                            st.session_state.session_id,
                            user_query,
                            retrieved_docs,
                            response,
                            settings['selected_model']
                        )
                        
                        # Add assistant message to chat history
                        assistant_message = {
                            "role": "assistant",
                            "content": response,
                            "retrieved_docs": retrieved_docs,
                            "evaluation": evaluation_scores,
                            "model": settings['selected_model']
                        }
                        st.session_state.messages.append(assistant_message)
                        
                        # Store interaction ID for feedback
                        st.session_state.interaction_ids[len(st.session_state.messages) - 1] = interaction_id
                        
                    except Exception as e:
                        st.error(f"Error generating response: {e}")
                        response = f"I encountered an error while processing your request: {str(e)}"
                        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Rerun to update the interface
        st.rerun()

if __name__ == "__main__":
    main()