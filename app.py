import os
import streamlit as st

from agents import DEFAULT_MODEL
from router import run_pipeline
from tools import build_local_index, load_documents

SEARXNG_BASE_URL = "https://searxng.site/search"


st.set_page_config(page_title="AG2 Adaptive Research Team", layout="wide")

st.title("AG2 Adaptive Research Team")
st.caption("Agent teamwork + agent-enabled routing, built with AG2 × HuggingFace")

with st.sidebar:
    st.header("API Configuration")
    api_key = st.text_input(
        "HuggingFace Access Token",
        type="password",
        help="Create a free token at https://huggingface.co/settings/tokens (read access is enough)",
    )
    model = st.text_input(
        "Model (HuggingFace model ID + provider suffix)",
        value=DEFAULT_MODEL,
        help=(
            "Format: <org>/<model>:<provider>  e.g. Qwen/Qwen2.5-72B-Instruct:novita\n"
            "Use ':auto' to let HuggingFace pick the fastest available provider.\n"
            "Browse providers at: https://huggingface.co/models (Inference Providers filter)"
        ),
    )
    web_enabled = st.toggle("Enable Web Fallback", value=True)
    st.markdown(
        "Web fallback uses a public SearxNG instance, which may be rate-limited."
    )
    st.info(
        "**Getting a free token**\n\n"
        "1. Sign up at [huggingface.co](https://huggingface.co)\n"
        "2. Go to **Settings → Access Tokens**\n"
        "3. Create a token with **Read** scope\n"
        "4. Paste it above\n\n"
        "**Model format**\n\n"
        "Always append a provider suffix, e.g.:\n"
        "- `Qwen/Qwen2.5-72B-Instruct:novita`\n"
        "- `Qwen/Qwen2.5-72B-Instruct:auto`\n"
        "- `meta-llama/Llama-3.3-70B-Instruct:sambanova`"
    )

st.subheader("1. Upload Local Documents")
files = st.file_uploader(
    "Upload PDFs or text files",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True,
)

st.subheader("2. Ask a Question")
question = st.text_area("Research question")

run_clicked = st.button("Run Research")

if run_clicked:
    if not api_key:
        st.error("Please provide your HuggingFace access token.")
        st.stop()

    if not question.strip():
        st.error("Please enter a research question.")
        st.stop()

    # Validate that the model string includes a provider suffix
    if ":" not in model:
        st.error(
            "Model must include a provider suffix, e.g. `Qwen/Qwen2.5-72B-Instruct:novita`. "
            "Append `:auto` to let HuggingFace choose automatically."
        )
        st.stop()

    # Expose the token as an env var so any downstream library picks it up too
    os.environ["HF_TOKEN"] = api_key

    documents = load_documents(files or [])
    local_index = build_local_index(documents)

    with st.spinner("Running the AG2 team..."):
        result = run_pipeline(
            question=question,
            local_chunks=local_index,
            api_key=api_key,
            model=model,
            web_enabled=web_enabled,
            searxng_base_url=SEARXNG_BASE_URL,
        )

    st.subheader("Routing Decision")
    st.json(result.get("triage", {}))

    st.subheader("Evidence")
    st.json(result.get("evidence", []))

    st.subheader("Verifier")
    st.json(result.get("verifier", {}))

    st.subheader("Final Answer")
    st.markdown(result.get("final_answer", ""))