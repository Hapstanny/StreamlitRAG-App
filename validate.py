"""
Validation script to test the RAG Chat Assistant functionality
"""
import os
import sys
from dotenv import load_dotenv

def check_imports():
    """Test all required imports"""
    print("🔍 Checking imports...")
    try:
        import streamlit
        print("  ✅ Streamlit")
        
        import azure.search.documents
        print("  ✅ Azure Search Documents")
        
        import azure.ai.formrecognizer
        print("  ✅ Azure Form Recognizer")
        
        import azure.cosmos
        print("  ✅ Azure Cosmos DB")
        
        import openai
        print("  ✅ OpenAI")
        
        print("  ✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        print("  Run: pip install -r requirements.txt")
        return False

def check_environment():
    """Check environment configuration"""
    print("\n📋 Checking environment configuration...")
    
    # Load environment variables
    load_dotenv()
    
    required_vars = [
        'AZURE_SEARCH_SERVICE_NAME',
        'AZURE_SEARCH_API_KEY',
        'AZURE_SEARCH_INDEX_NAME',
        'AZURE_OPENAI_ENDPOINT',
        'AZURE_OPENAI_API_KEY',
        'AZURE_OPENAI_API_VERSION',
        'AZURE_OPENAI_DEPLOYMENTS',
        'AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT',
        'AZURE_DOCUMENT_INTELLIGENCE_API_KEY',
        'COSMOS_DB_ENDPOINT',
        'COSMOS_DB_KEY',
        'COSMOS_DB_DATABASE_NAME',
        'COSMOS_DB_CONTAINER_NAME'
    ]
    
    missing_vars = []
    placeholder_vars = []
    
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            missing_vars.append(var)
        elif value.startswith('your-'):
            placeholder_vars.append(var)
        else:
            print(f"  ✅ {var}")
    
    if missing_vars:
        print("\n  ❌ Missing environment variables:")
        for var in missing_vars:
            print(f"    - {var}")
    
    if placeholder_vars:
        print("\n  ⚠️  Environment variables with placeholder values:")
        for var in placeholder_vars:
            print(f"    - {var}")
    
    if not missing_vars and not placeholder_vars:
        print("  ✅ All environment variables are configured!")
        return True
    else:
        print("\n  Please update your .env file with actual Azure service credentials.")
        return False

def check_data_directory():
    """Check data directory setup"""
    print("\n📁 Checking data directory...")
    
    data_path = os.getenv('LOCAL_DATA_PATH', './data')
    
    if os.path.exists(data_path):
        print(f"  ✅ Data directory exists: {data_path}")
        
        files = [f for f in os.listdir(data_path) 
                if f.lower().endswith(('.pdf', '.docx', '.txt', '.jpg', '.png'))]
        
        if files:
            print(f"  ✅ Found {len(files)} document(s) ready for ingestion:")
            for file in files[:5]:  # Show first 5 files
                print(f"    - {file}")
            if len(files) > 5:
                print(f"    ... and {len(files) - 5} more")
        else:
            print("  ⚠️  No documents found. Add files to the data directory for testing.")
        
        return True
    else:
        print(f"  ❌ Data directory not found: {data_path}")
        print("  Run: python setup.py to create the directory")
        return False

def test_services():
    """Test service initialization"""
    print("\n🔧 Testing service initialization...")
    
    try:
        from rag_services import ConfigManager
        config = ConfigManager()
        print("  ✅ ConfigManager initialized")
        
        # Test configuration values
        if not config.search_service_name or config.search_service_name.startswith('your-'):
            print("  ⚠️  Azure Search service not configured")
            return False
        
        if not config.openai_endpoint or config.openai_endpoint.startswith('https://your-'):
            print("  ⚠️  Azure OpenAI endpoint not configured")
            return False
        
        print("  ✅ Core configuration validated")
        return True
        
    except Exception as e:
        print(f"  ❌ Service initialization failed: {e}")
        return False

def check_streamlit_config():
    """Check if Streamlit can run"""
    print("\n🌐 Checking Streamlit configuration...")
    
    if os.path.exists('app.py'):
        print("  ✅ app.py found")
        
        # Basic syntax check
        try:
            with open('app.py', 'r', encoding='utf-8') as f:
                content = f.read()
                compile(content, 'app.py', 'exec')
            print("  ✅ app.py syntax is valid")
            return True
        except SyntaxError as e:
            print(f"  ❌ Syntax error in app.py: {e}")
            return False
    else:
        print("  ❌ app.py not found")
        return False

def main():
    """Run all validation checks"""
    print("🤖 RAG Chat Assistant Validation")
    print("=" * 50)
    
    checks = [
        check_imports,
        check_environment,
        check_data_directory,
        test_services,
        check_streamlit_config
    ]
    
    results = []
    for check in checks:
        results.append(check())
    
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)
    
    if all(results):
        print("✅ All checks passed! Your RAG Chat Assistant is ready to run.")
        print("\nTo start the application:")
        print("1. Ensure your Azure services are properly configured")
        print("2. Run: python setup.py (if you haven't already)")
        print("3. Run: streamlit run app.py")
        print("4. Open your browser and start chatting!")
    else:
        print("❌ Some checks failed. Please address the issues above.")
        print("\nCommon solutions:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Configure .env file with your Azure credentials")
        print("3. Run setup script: python setup.py")
        
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)