"""
Setup script for creating Azure Search index and initial configuration
"""
import os
import json
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SearchField, SearchFieldDataType, SimpleField, SearchableField,
    VectorSearch, HnswAlgorithmConfiguration, VectorSearchProfile,
    SemanticConfiguration, SemanticPrioritizedFields, SemanticField, SemanticSearch
)
from azure.core.credentials import AzureKeyCredential
from rag_services import ConfigManager

def create_search_index(config: ConfigManager):
    """Create Azure Search index with vector search support"""
    try:
        # Initialize the index client
        admin_client = SearchIndexClient(
            endpoint=f"https://{config.search_service_name}.search.windows.net",
            credential=AzureKeyCredential(config.search_api_key)
        )
        
        # Configure vector search
        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="hnsw-config",
                    parameters={
                        "m": 4,
                        "efConstruction": 400,
                        "efSearch": 500,
                        "metric": "cosine"
                    }
                )
            ],
            profiles=[
                VectorSearchProfile(
                    name="vector-profile",
                    algorithm_configuration_name="hnsw-config"
                )
            ]
        )
        
        # Define the index schema with vector field
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True, sortable=True, filterable=True),
            SearchableField(name="title", type=SearchFieldDataType.String, searchable=True, sortable=True),
            SearchableField(name="content", type=SearchFieldDataType.String, searchable=True, analyzer_name="en.lucene"),
            SimpleField(name="source", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="chunk_id", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="url", type=SearchFieldDataType.String, filterable=True),
            # Vector field for embeddings
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=config.embedding_dimensions,
                vector_search_profile_name="vector-profile"
            )
        ]
        
        # Configure semantic search (optional - only works on paid tiers)
        semantic_config = SemanticConfiguration(
            name="default",
            prioritized_fields=SemanticPrioritizedFields(
                title_field=SemanticField(field_name="title"),
                content_fields=[SemanticField(field_name="content")]
            )
        )
        
        semantic_search = SemanticSearch(configurations=[semantic_config])
        
        # Create the index with vector search
        index = SearchIndex(
            name=config.search_index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_search=semantic_search
        )
        
        result = admin_client.create_or_update_index(index)
        print(f"Index '{config.search_index_name}' created successfully with vector search!")
        print(f"  - Vector dimensions: {config.embedding_dimensions}")
        print(f"  - Embedding model: {config.embedding_deployment}")
        return True
        
    except Exception as e:
        print(f"Error creating index: {e}")
        return False

def validate_configuration(config: ConfigManager):
    """Validate that all required configurations are set"""
    required_settings = {
        'Azure Search Service Name': config.search_service_name,
        'Azure Search API Key': config.search_api_key,
        'Azure OpenAI Endpoint': config.openai_endpoint,
        'Azure OpenAI API Key': config.openai_api_key,
        'Azure Document Intelligence Endpoint': config.doc_intel_endpoint,
        'Azure Document Intelligence API Key': config.doc_intel_key,
        'Cosmos DB Endpoint': config.cosmos_endpoint,
        'Cosmos DB Key': config.cosmos_key
    }
    
    missing_settings = []
    for setting_name, setting_value in required_settings.items():
        if not setting_value or setting_value.startswith('your-'):
            missing_settings.append(setting_name)
    
    if missing_settings:
        print("❌ Missing required configuration:")
        for setting in missing_settings:
            print(f"  - {setting}")
        print("\nPlease update your .env file with the correct values.")
        return False
    else:
        print("✅ All required configurations are set!")
        return True

def create_data_directory(config: ConfigManager):
    """Create data directory if it doesn't exist"""
    if not os.path.exists(config.local_data_path):
        os.makedirs(config.local_data_path)
        print(f"Created data directory: {config.local_data_path}")
        
        # Create a sample document
        sample_content = """
        Welcome to the RAG Chat Assistant!
        
        This is a sample document to demonstrate the document ingestion and retrieval capabilities.
        
        The system supports:
        - PDF documents
        - Word documents (.docx)
        - Text files (.txt)
        - Image files with text (JPG, PNG)
        
        Features:
        - Semantic search and ranking
        - Agentic retrieval with query rewriting
        - Real-time evaluation of responses
        - User feedback collection
        - Azure AI services integration
        
        To get started:
        1. Place your documents in the data folder
        2. Use the "Ingest Documents" button in the sidebar
        3. Start asking questions!
        """
        
        sample_file = os.path.join(config.local_data_path, "sample_document.txt")
        with open(sample_file, 'w', encoding='utf-8') as f:
            f.write(sample_content)
        print(f"Created sample document: {sample_file}")
    else:
        print(f"Data directory already exists: {config.local_data_path}")

def main():
    """Main setup function"""
    print("🔧 RAG Chat Assistant Setup")
    print("=" * 50)
    
    try:
        # Load configuration
        config = ConfigManager()
        
        # Validate configuration
        print("\n📋 Validating configuration...")
        if not validate_configuration(config):
            return
        
        # Create data directory
        print("\n📁 Setting up data directory...")
        create_data_directory(config)
        
        # Create search index
        print("\n🔍 Creating Azure Search index...")
        if create_search_index(config):
            print("✅ Setup completed successfully!")
            print("\nNext steps:")
            print("1. Place your documents in the data folder")
            print("2. Run: streamlit run app.py")
            print("3. Use the 'Ingest Documents' button to index your files")
            print("4. Start chatting!")
        else:
            print("❌ Setup failed. Please check your Azure Search configuration.")
    
    except Exception as e:
        print(f"❌ Setup failed with error: {e}")

if __name__ == "__main__":
    main()