from __future__ import annotations

from typing import Any, Dict

from autogen import AssistantAgent


# HuggingFace Inference Providers — new unified router (replaces api-inference.huggingface.co).
# The old endpoint no longer handles large LLMs.
HF_BASE_URL = "https://router.huggingface.co/v1"

# Default model + provider.
# The ":novita" suffix tells HF which inference provider to route through.
# Use ":auto" to let HF pick the fastest available provider automatically.
# Other free-tier options: ":together", ":sambanova", ":fireworks-ai"
# Browse supported providers per model at: https://huggingface.co/models -> Inference Providers
DEFAULT_MODEL = "Qwen/Qwen2.5-72B-Instruct:novita"


def make_llm_config(
    api_key: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    return {
        "config_list": [
            {
                "api_type": "openai",   # router.huggingface.co is OpenAI-compatible
                "model": model,         # must include ":provider" suffix, e.g. ":novita"
                "api_key": api_key,     # your HuggingFace access token (hf_...)
                "base_url": HF_BASE_URL,
            }
        ],
        "temperature": temperature,
    }


def build_agents(api_key: str, model: str = DEFAULT_MODEL) -> Dict[str, AssistantAgent]:
    llm_config = make_llm_config(api_key=api_key, model=model)

    triage_agent = AssistantAgent(
        name="triage_agent",
        llm_config=llm_config,
        system_message=(
            "You are a triage agent for a research team. "
            "Classify whether the question can be answered from local documents or needs web research. "
            "Respond ONLY with JSON."
        ),
    )

    local_research_agent = AssistantAgent(
        name="local_research_agent",
        llm_config=llm_config,
        system_message=(
            "You are a local research agent. Use only the provided document excerpts. "
            "Return JSON with evidence and a draft answer."
        ),
    )

    web_research_agent = AssistantAgent(
        name="web_research_agent",
        llm_config=llm_config,
        system_message=(
            "You are a web research agent. Use the provided web search results only. "
            "Return JSON with evidence and a draft answer."
        ),
    )

    verifier_agent = AssistantAgent(
        name="verifier_agent",
        llm_config=llm_config,
        system_message=(
            "You are a verifier. Check evidence sufficiency and identify gaps. "
            "Return JSON verdict and gaps."
        ),
    )

    synthesizer_agent = AssistantAgent(
        name="synthesizer_agent",
        llm_config=llm_config,
        system_message=(
            "You are the final synthesizer. Produce a clear answer with citations to the evidence."
        ),
    )

    return {
        "triage": triage_agent,
        "local": local_research_agent,
        "web": web_research_agent,
        "verifier": verifier_agent,
        "synthesizer": synthesizer_agent,
    }


def run_agent(agent: AssistantAgent, prompt: str) -> str:
    reply = agent.generate_reply(messages=[{"role": "user", "content": prompt}])
    if isinstance(reply, dict):
        return reply.get("content", "") or ""
    return str(reply)