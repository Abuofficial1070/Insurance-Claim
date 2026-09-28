from google.adk.agents import Agent

from .tools import fraud_prediction_tool
from .tools import policy_search_tool


root_agent = Agent(

    name="InsuranceDecisionAgent",

    model="gemini-2.5-flash",

    description="Insurance Claim Verification Agent",

    instruction="""
You are an insurance verification agent.

Always use the available tools.

Tool 1

Predict fraud risk.

Tool 2

Search the insurance policy.

After calling both tools,
analyse the results.

Return

Disease Coverage

Hospital Verification

Claim Amount Analysis

Fraud Analysis

Policy Status

Final Decision

Reason
""",

    tools=[
        fraud_prediction_tool,
        policy_search_tool
    ]

)