"""Small Gradio app that sends a user question to the OpenAI client and
returns an optimized search query and a justification.

This replaces the previous Streamlit-based UI with a Gradio interface.
"""

from dotenv import load_dotenv
import json
import re
from pydantic import BaseModel
from openai import OpenAI
import gradio as gr

# Load environment variables (e.g. OPENAI_API_KEY) from a .env file
load_dotenv(override=True)

client = OpenAI()


class WebSearchPrompt(BaseModel):
    search_query: str
    justification: str


def _strip_code_fences(text: str) -> str:
    """Remove triple-backtick code fences and surrounding language markers."""
    # Remove ```...``` blocks
    text = re.sub(r"```[\s\S]*?```", lambda m: m.group(0).strip('`'), text)
    return text.strip()


def _extract_json(text: str):
    """Try to extract a JSON object from model text.

    Returns parsed JSON dict or raises ValueError if not found/parseable.
    """
    text = _strip_code_fences(text)
    # Find the first '{' and the last '}' and try to parse that substring.
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in text")
    candidate = text[start:end+1]
    return json.loads(candidate)


def generate_search(user_query: str):
    """Call the OpenAI chat completion to generate an optimized search query.

    Returns a tuple: (search_query:str, justification:str)
    """
    if not user_query or not user_query.strip():
        return "", ""

    system_prompt = (
        "You are an assistant that rewrites a user's question into an optimized web "
        "search query and provides a brief justification. Respond in JSON only with "
        "two keys: `search_query` and `justification`. Example: {\"search_query\": \"...\", "
        "\"justification\": \"...\"}"
    )

    # Call the OpenAI chat completions API
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        temperature=0.3,
    )

    response_content = response.choices[0].message.content

    # Attempt to parse JSON from the model's response
    try:
        parsed = _extract_json(response_content)
        obj = WebSearchPrompt(**parsed)
        return obj.search_query, obj.justification
    except Exception:
        # Fallback: return the raw content as justification and echo the user query
        return user_query, response_content


title = "Web Search Optimization with LLM"
description = "Enter a question to receive an optimized web search query and reasoning."


with gr.Blocks(title=title) as demo:
    gr.Markdown(f"## {title}")
    gr.Markdown(description)

    with gr.Row():
        inp = gr.Textbox(label="Enter your question:", lines=2, placeholder="Ask something like: 'How do I optimize a SQL query for large tables'?")
        out_query = gr.Textbox(label="Optimized Search Query")
        out_reason = gr.Textbox(label="Reasoning")

    def _wrap_and_return(q: str):
        search, reason = generate_search(q)
        return search, reason

    submit_btn = gr.Button("Generate")
    submit_btn.click(fn=_wrap_and_return, inputs=inp, outputs=[out_query, out_reason])


if __name__ == "__main__":
    demo.launch()