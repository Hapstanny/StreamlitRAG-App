"""Verify traces in Application Insights for AI Foundry Monitoring.

This script queries Application Insights to verify:
1. Inference traces with gen_ai.* attributes
2. Evaluation traces with gen_ai.evaluation.* format

Uses the same query pattern as Foundry Monitoring dashboard.
"""
import os
from datetime import timedelta
from dotenv import load_dotenv
from pprint import pprint

load_dotenv(override=True)

from azure.identity import AzureCliCredential
from azure.monitor.query import LogsQueryClient, LogsQueryStatus
from azure.ai.projects import AIProjectClient

# Get credentials
tenant_id = os.getenv('AZURE_TENANT_ID')
credential = AzureCliCredential(tenant_id=tenant_id)

# Get workspace ID from AI Foundry project
project_endpoint = os.getenv('AZURE_AI_FOUNDRY_ENDPOINT')
project_client = AIProjectClient(endpoint=project_endpoint, credential=credential)

# Get the Log Analytics workspace ID from the connection string
conn_str = project_client.telemetry.get_application_insights_connection_string()
print(f"App Insights Connection: {conn_str[:80]}...")

# Parse the workspace ID from the connection string - we need the resource ID
# For now, let's check if we can get it from environment
workspace_id = os.getenv('LOGS_WORKSPACE_ID')

if not workspace_id:
    # Try to extract from connection string or use the App Insights resource directly
    print("\nNote: LOGS_WORKSPACE_ID not set in .env")
    print("You can find it in Azure Portal -> Application Insights -> Overview -> Workspace ID")
    print("Or from Azure Portal -> Log Analytics workspace -> Properties -> Resource ID\n")
    
    # Try using the instrumentation key to identify the resource
    if "InstrumentationKey=" in conn_str:
        ikey = conn_str.split("InstrumentationKey=")[1].split(";")[0]
        print(f"Instrumentation Key: {ikey}")
        print("\nTo verify traces, run this KQL query in Azure Portal -> Application Insights -> Logs:")
else:
    # Query the traces table
    client = LogsQueryClient(credential)
    
    # Query for inference calls (gen_ai.* traces)
    inference_query = """
    dependencies
    | where timestamp > ago(1h)
    | where isnotnull(customDimensions["gen_ai.system"]) or isnotnull(customDimensions["gen_ai.provider.name"])
    | project timestamp, name, customDimensions
    | order by timestamp desc
    | take 5
    """
    
    # Query for evaluation traces
    eval_query = """
    traces
    | where timestamp > ago(1h)
    | where message startswith "gen_ai.evaluation" or customDimensions["event.name"] startswith "gen_ai.evaluation"
    | project timestamp, message, 
              evaluator=customDimensions["gen_ai.evaluator.name"],
              score=customDimensions["gen_ai.evaluation.score"],
              response_id=customDimensions["gen_ai.response.id"]
    | order by timestamp desc
    | take 10
    """
    
    print("\n=== Querying for Inference Traces ===")
    try:
        response = client.query_workspace(workspace_id, inference_query, timespan=timedelta(hours=1))
        if response.status == LogsQueryStatus.SUCCESS:
            for table in response.tables:
                for row in table.rows:
                    pprint(dict(zip(table.columns, row)))
        else:
            print(f"Query error: {response.partial_error}")
    except Exception as e:
        print(f"Error querying inference traces: {e}")
    
    print("\n=== Querying for Evaluation Traces ===")
    try:
        response = client.query_workspace(workspace_id, eval_query, timespan=timedelta(hours=1))
        if response.status == LogsQueryStatus.SUCCESS:
            for table in response.tables:
                for row in table.rows:
                    pprint(dict(zip(table.columns, row)))
        else:
            print(f"Query error: {response.partial_error}")
    except Exception as e:
        print(f"Error querying evaluation traces: {e}")

# Print sample KQL queries to run manually
print("\n" + "="*60)
print("MANUAL VERIFICATION - Run these queries in Azure Portal")
print("Go to: Application Insights -> Logs")
print("="*60)

print("""
// 1. Check for inference traces with gen_ai.* attributes (dependencies table)
dependencies
| where timestamp > ago(1h)
| where isnotnull(customDimensions["gen_ai.system"]) 
     or isnotnull(customDimensions["gen_ai.provider.name"])
| project timestamp, name, 
          system=customDimensions["gen_ai.system"],
          model=customDimensions["gen_ai.request.model"],
          response_id=customDimensions["gen_ai.response.id"]
| order by timestamp desc
| take 10
""")

print("""
// 2. Check for evaluation traces (traces table - where Foundry Monitoring looks)
traces
| where timestamp > ago(1h)
| where message startswith "gen_ai.evaluation" 
     or customDimensions["event.name"] startswith "gen_ai.evaluation"
| project timestamp, message,
          event_name=customDimensions["event.name"],
          evaluator=customDimensions["gen_ai.evaluator.name"],
          score=customDimensions["gen_ai.evaluation.score"],
          response_id=customDimensions["gen_ai.response.id"]
| order by timestamp desc
| take 20
""")

print("""
// 3. Full Foundry Monitoring query (simplified)
let get_event_name = (customDimensions: dynamic, message: string) { 
    iff(customDimensions["event.name"] == "", message, customDimensions["event.name"]) 
};
traces
| where timestamp > ago(1h)
| extend event_name = get_event_name(customDimensions, message)
| where event_name startswith "gen_ai.evaluation"
| extend
    evaluator_name = coalesce(tostring(customDimensions["gen_ai.evaluator.name"]), split(event_name, ".")[2]),
    score = todouble(customDimensions["gen_ai.evaluation.score"]),
    response_id = tostring(customDimensions["gen_ai.response.id"])
| project timestamp, evaluator_name, score, response_id
| order by timestamp desc
""")
