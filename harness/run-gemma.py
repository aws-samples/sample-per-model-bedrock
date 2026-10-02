import json

from aws_bedrock_token_generator import provide_token
from openai import OpenAI

REGION = "us-east-1"
MODEL = "google.gemma-4-31b"

client = OpenAI(
    api_key=provide_token(region=REGION),
    base_url=f"https://bedrock-mantle.{REGION}.api.aws/openai/v1",
)

answer_tool = {
    "type": "function",
    "name": "answer",
    "description": "Return the company profile.",
    "parameters": {
        "type": "object",
        "properties": {
            "business": {"type": "string"},
        },
        "required": [
            "business"
        ],
    },
}

response = client.responses.create(
    model=MODEL,
    input=[{"role": "user", "content": "Tell me about Parcel Perform"}],
    tools=[answer_tool],
    tool_choice={"type": "function", "name": "answer"},  # must call it
    temperature=1.0,  # Gemma 4 repeats itself at 0
    store=False,  # don't retain the conversation for 30 days
)

call = next(item for item in response.output if item.type == "function_call")
print(json.dumps(json.loads(call.arguments), indent=2))
