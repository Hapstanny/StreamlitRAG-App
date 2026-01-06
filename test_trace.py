"""Test script to verify AI Foundry tracing with gen_ai.* attributes.

Per docs: https://learn.microsoft.com/en-us/azure/ai-foundry/observability/how-to/trace-agent-setup
Key requirements:
1. Use AIProjectInstrumentor from azure.ai.projects.telemetry (adds gen_ai.* attributes)
2. Set OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true for content recording
3. configure_azure_monitor() exports to Application Insights
"""
import os
os.environ['OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT'] = 'true'
os.environ['OTEL_SERVICE_NAME'] = 'rag-chat-service'

from dotenv import load_dotenv
load_dotenv(override=True)

from azure.identity import AzureCliCredential
from azure.ai.projects import AIProjectClient
from azure.monitor.opentelemetry import configure_azure_monitor
from opentelemetry import trace
from openai import AzureOpenAI
import time

print("=== AI Foundry Tracing Test (with gen_ai.* attributes) ===\n")

# Setup credentials
tenant_id = os.getenv('AZURE_TENANT_ID')
credential = AzureCliCredential(tenant_id=tenant_id)

# Get App Insights connection string from AI Foundry
project_endpoint = os.getenv('AZURE_AI_FOUNDRY_ENDPOINT')
print(f"Project endpoint: {project_endpoint}")

project_client = AIProjectClient(endpoint=project_endpoint, credential=credential)
conn_str = project_client.telemetry.get_application_insights_connection_string()
print(f"App Insights: {conn_str[:70]}...")

# Configure Azure Monitor FIRST
configure_azure_monitor(connection_string=conn_str)
print("Azure Monitor configured!")

# Use AIProjectInstrumentor - this adds gen_ai.system, gen_ai.provider.name attributes
from azure.ai.projects.telemetry import AIProjectInstrumentor
AIProjectInstrumentor().instrument()
print("AIProjectInstrumentor enabled - gen_ai.* attributes will be traced!")

# Get tracer
tracer = trace.get_tracer(__name__)

# Setup OpenAI client (AIProjectInstrumentor will instrument this automatically)
openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
print(f"OpenAI endpoint: {openai_endpoint}")

client = AzureOpenAI(
    azure_endpoint=openai_endpoint,
    azure_ad_token_provider=lambda: credential.get_token("https://cognitiveservices.azure.com/.default").token,
    api_version='2024-12-01-preview'
)
print("AzureOpenAI client created!")

# Make an instrumented OpenAI call within a custom span
print("\n--- Making OpenAI call (should have gen_ai.* attributes) ---")
with tracer.start_as_current_span('rag-query-test') as span:
    span.set_attribute('user.query', 'What is artificial intelligence?')
    
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'What is AI? Answer in one sentence.'}
        ],
        max_tokens=50
    )
    
    response_id = response.id  # Capture response ID to link evaluation
    print(f"Response: {response.choices[0].message.content}")
    print(f"Response ID: {response_id}")
    print(f"Model: {response.model}")
    print(f"Usage: input_tokens={response.usage.prompt_tokens}, output_tokens={response.usage.completion_tokens}")

# Test evaluation metrics in AI Foundry format
print("\n--- Recording Evaluation Metrics (gen_ai.evaluation.* format) ---")
from rag_services import record_evaluation_metrics

# Record evaluation scores - each metric becomes a separate trace event
test_scores = {
    'relevance': 4.5,
    'coherence': 4.0,
    'groundedness': 3.8,
    'fluency': 4.2
}

record_evaluation_metrics(
    query="What is AI?",
    response=response.choices[0].message.content,
    context="AI is artificial intelligence...",
    scores=test_scores,
    model='gpt-4o-mini',
    response_id=response_id,  # Link evaluation to the inference call
    method="test_evaluation"
)
print(f"Recorded {len(test_scores)} evaluation metrics linked to response_id={response_id}")

print("\n--- Waiting for traces to flush (5 seconds) ---")
time.sleep(5)
print("\nDone! Check AI Foundry portal Tracing tab AND Monitoring tab.")
print("The trace should now have gen_ai.system and gen_ai.provider.name attributes.")
print("Evaluations should appear with gen_ai.evaluation.score and gen_ai.evaluator.name.")

