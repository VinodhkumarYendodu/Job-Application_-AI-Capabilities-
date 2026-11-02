from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path

import streamlit as st
from groq import APIConnectionError, APIStatusError, RateLimitError

from agents import AGENTS, MODEL_DEFAULT, run_agent
from job_finder import JOB_FINDER, find_matching_jobs
from providers import GROQ_FALLBACK_MODELS, list_provider_models

ROOT = Path(__file__).parent
st.set_page_config(page_title="ApplyFlow · Job Application Studio", page_icon="✦", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { --ink:#172923; --muted:#6e7c73; --green:#176b52; --lime:#d8f26b; --paper:#f5f7f2; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:var(--paper); color:var(--ink); }
.block-container { padding-top:2rem; max-width:1280px; }
h1,h2,h3 { font-family:'Manrope',sans-serif; color:var(--ink); letter-spacing:-.035em; }
.hero { background:#143f32; color:#f7fbf5; border-radius:24px; padding:34px 38px; margin-bottom:24px; position:relative; overflow:hidden; }
.hero:after { content:'✦'; position:absolute; right:38px; top:14px; color:#d8f26b; font-size:94px; opacity:.9; }
.hero h1 { color:white; font-size:2.55rem; margin:7px 0 8px; }
.hero p { color:#d2e0d6; max-width:670px; margin:0; }
.eyebrow { color:#d8f26b; font-weight:700; font-size:.76rem; letter-spacing:.13em; text-transform:uppercase; }
.agent-card { background:white; border:1px solid #e5eae2; border-radius:16px; padding:16px 18px; min-height:100px; }
.agent-card b { color:#173d31; font-family:'Manrope',sans-serif; }
.agent-card small { color:#78867e; line-height:1.45; display:block; margin-top:5px; }
[data-testid="stSidebar"] { background:#edf2eb; }
.stButton button[kind="primary"] { background:#176b52; border:0; border-radius:11px; font-weight:700; }
.stButton button { border-radius:10px; }
.stTabs [data-baseweb="tab-list"] { gap:8px; }
.stTabs [data-baseweb="tab"] { border-radius:10px 10px 0 0; }
div[data-testid="stMetric"] { background:white; border:1px solid #e7ece4; padding:14px 16px; border-radius:14px; }
</style>
""", unsafe_allow_html=True)


def read_sample(filename: str) -> str:
    path = ROOT / filename
    return path.read_text(encoding="utf-8") if path.exists() else ""


def load_uploaded_text(upload_key: str, target_key: str) -> None:
    upload = st.session_state.get(upload_key)
    if upload is None:
        return
    try:
        suffix = Path(upload.name).suffix.lower()
        raw = upload.getvalue()
        if suffix in (".md", ".txt"):
            text = raw.decode("utf-8-sig")
        elif suffix == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(BytesIO(raw))
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        elif suffix == ".docx":
            from docx import Document
            document = Document(BytesIO(raw))
            parts = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
            for table in document.tables:
                parts.extend(" | ".join(cell.text.strip() for cell in row.cells) for row in table.rows)
            text = "\n".join(parts)
        else:
            st.error("Supported resume formats are PDF, DOCX, Markdown, and TXT.")
            return
        if not text.strip():
            st.error("No selectable text was found. If this is a scanned PDF, use a text-based PDF or paste the resume text into the editor.")
            return
        st.session_state[target_key] = text
    except UnicodeDecodeError:
        st.error("Could not decode this file as UTF-8. Save it as UTF-8 text, DOCX, or PDF and upload it again.")
    except Exception as exc:
        st.error(f"Could not extract text from {upload.name}: {exc}")


st.markdown("""
<section class="hero"><div class="eyebrow">Your job search, with a little more clarity</div>
<h1>Make every application<br>feel intentional.</h1>
<p>Seven focused AI agents turn your resume and a job description into a grounded application kit—from fit analysis to live job matches and interview practice.</p></section>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ✦ ApplyFlow")
    st.caption("A thoughtful workspace for your next application.")
    provider_label = st.selectbox(
        "AI provider",
        ["Groq", "OpenAI API (ChatGPT)", "Anthropic API (Claude)", "Google AI API (Gemini)", "Ollama"],
        key="ai_provider",
        help="Provider API access and billing are managed separately from consumer chat subscriptions.",
    )
    provider = {
        "Groq": "Groq",
        "OpenAI API (ChatGPT)": "OpenAI",
        "Anthropic API (Claude)": "Anthropic",
        "Google AI API (Gemini)": "Gemini",
        "Ollama": "Ollama",
    }[provider_label]
    if provider == "Ollama":
        api_key = ""
        base_url = st.text_input("Ollama server URL", value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"), help="For a local Ollama install, keep the default address.")
        if "ollama_models" not in st.session_state:
            st.session_state.ollama_models = []
            st.session_state.ollama_models_url = ""
        refresh_ollama = st.button("↻ Refresh local models", key="refresh_ollama_models")
        if refresh_ollama or st.session_state.ollama_models_url != base_url.rstrip("/"):
            try:
                st.session_state.ollama_models = list_provider_models("Ollama", "", base_url)
                st.session_state.ollama_models_error = "" if st.session_state.ollama_models else "Ollama is running, but no models are installed."
            except Exception as exc:
                st.session_state.ollama_models = []
                st.session_state.ollama_models_error = f"Cannot reach Ollama at {base_url}: {exc}"
            st.session_state.ollama_models_url = base_url.rstrip("/")
        local_models = st.session_state.ollama_models
        if st.session_state.get("ollama_model") not in local_models:
            st.session_state.ollama_model = local_models[0] if local_models else "No models detected"
        selected_local = st.selectbox("Installed Ollama model", local_models or ["No models detected"], disabled=not bool(local_models), key="ollama_model")
        model = selected_local if local_models else ""
        if st.session_state.get("ollama_models_error"):
            st.caption(st.session_state.ollama_models_error)
    else:
        base_url = ""
        key_env = {"Groq": "GROQ_API_KEY", "OpenAI": "OPENAI_API_KEY", "Anthropic": "ANTHROPIC_API_KEY", "Gemini": "GEMINI_API_KEY"}[provider]
        api_key = st.text_input(f"{provider_label} API key", type="password", value=os.getenv(key_env, ""), help=f"Used for this session only. Or set {key_env} in your environment.")
        model_state_key = f"models_{provider.lower()}"
        error_state_key = f"models_error_{provider.lower()}"
        if model_state_key not in st.session_state:
            st.session_state[model_state_key] = GROQ_FALLBACK_MODELS.copy() if provider == "Groq" else []
        if st.button(f"↻ Load available {provider_label} models", disabled=not bool(api_key.strip()), key=f"load_models_{provider.lower()}"):
            try:
                st.session_state[model_state_key] = list_provider_models(provider, api_key.strip())
                st.session_state[error_state_key] = ""
            except Exception as exc:
                st.session_state[error_state_key] = str(exc)
        if st.session_state.get(error_state_key):
            st.caption(f"Could not load models: {st.session_state[error_state_key]}")
        available_models = st.session_state[model_state_key]
        if available_models:
            default_model = MODEL_DEFAULT if provider == "Groq" and MODEL_DEFAULT in available_models else available_models[0]
            model_key = f"selected_model_{provider.lower()}"
            current_model = st.session_state.get(model_key)
            if current_model not in available_models:
                st.session_state[model_key] = default_model
                current_model = default_model
            index = available_models.index(current_model) if current_model in available_models else available_models.index(default_model)
            model = st.selectbox(f"{provider_label} model", available_models, index=index, key=model_key)
        else:
            st.selectbox(f"{provider_label} model", ["Enter API key and load models"], disabled=True, key=f"empty_model_{provider.lower()}")
            model = ""
            st.caption("Enter your provider API key, then load the models available to that account.")
    st.divider()
    st.markdown("**Your information stays yours**")
    if provider == "Ollama":
        st.caption("Prompts are sent to your Ollama server. Job search terms go to public web search; matching runs on your selected local model. This app does not save uploads to disk.")
    else:
        st.caption(f"Resume and job description text are sent to {provider_label}. Job Finder searches public listings and ranks them with your selected provider. This app does not save uploads to disk.")

left, right = st.columns(2, gap="large")
if "resume_text" not in st.session_state:
    st.session_state.resume_text = read_sample("ravi_resume.md")
if "jd_text" not in st.session_state:
    st.session_state.jd_text = read_sample("jd.md")
with left:
    st.markdown("### 01 · Your experience")
    st.file_uploader("Upload a resume (PDF, DOCX, MD, or TXT)", type=["pdf", "docx", "md", "txt"], key="resume_file", on_change=load_uploaded_text, args=("resume_file", "resume_text"))
    resume = st.text_area("Resume text", height=340, key="resume_text", label_visibility="collapsed", placeholder="Paste your resume here…")
with right:
    st.markdown("### 02 · The opportunity")
    st.file_uploader("Upload a job description (PDF, DOCX, MD, or TXT)", type=["pdf", "docx", "md", "txt"], key="jd_file", on_change=load_uploaded_text, args=("jd_file", "jd_text"))
    job_description = st.text_area("Job description", height=340, key="jd_text", label_visibility="collapsed", placeholder="Paste the job description here…")

st.markdown("### Your application team")
cols = st.columns(3)
for i, agent in enumerate([*AGENTS.values(), JOB_FINDER]):
    with cols[i % 3]:
        st.markdown(f'<div class="agent-card"><b>{agent.name}</b><small>{agent.purpose}</small></div>', unsafe_allow_html=True)

if "outputs" not in st.session_state:
    st.session_state.outputs = {}

st.markdown("### Build your application kit")
run_all = st.button("✦  Run full application workflow", type="primary", use_container_width=True)
tabs = st.tabs(["JD Analysis", "Match Report", "Tailored Resume", "Cover Letter", "Accuracy Review", "Interview Prep", "Job Finder"])
keys = list(AGENTS.keys())


def generate(key: str) -> str | None:
    if not resume.strip() or not job_description.strip():
        st.error("Add both a resume and a job description to continue.")
        return None
    if provider != "Ollama" and not api_key.strip():
        st.error(f"Add your {provider_label} API key in the sidebar to run an agent.")
        return None
    if provider == "Ollama" and not model:
        st.error("No Ollama model is available. Start Ollama, install a model, and refresh the local model list.")
        return None
    context = ""
    if key == "review":
        context = "\n\n".join(f"{AGENTS[k].name}:\n{v}" for k, v in st.session_state.outputs.items() if k in ("tailor", "cover"))
    elif key in ("tailor", "cover", "interview") and "match" in st.session_state.outputs:
        context = st.session_state.outputs["match"]
    try:
        with st.spinner(f"{AGENTS[key].name} is working…"):
            result = run_agent(key, resume, job_description, context=context, provider=provider, api_key=api_key.strip(), base_url=base_url, model=model.strip())
        st.session_state.outputs[key] = result
        return result
    except RateLimitError:
        st.error("Groq rate limit reached. Wait a moment, then try again.")
    except APIConnectionError:
        st.error("Could not connect to Groq. Check your connection and try again.")
    except APIStatusError as exc:
        if exc.status_code == 404 and "model" in str(exc).lower():
            st.error(f"Groq could not find model `{model.strip()}` for this account. The model may be retired or unavailable. Set the sidebar model to `openai/gpt-oss-120b` or choose an active model from Groq's supported model list.")
        else:
            st.error(f"Groq API returned {exc.status_code}: {exc.message}")
    except Exception as exc:
        st.error(f"Could not run this agent: {exc}")
    return None


if run_all:
    if resume.strip() and job_description.strip() and (provider == "Ollama" or api_key.strip()):
        progress = st.progress(0, text="Preparing your application team…")
        for index, key in enumerate(keys):
            result = generate(key)
            progress.progress((index + 1) / len(keys), text=f"{AGENTS[key].name} {'finished' if result else 'needs attention'}")
            if result is None:
                break
        else:
            st.success("Your application kit is ready. Review every claim before using it.")
    else:
        generate("analysis")

for tab, key in zip(tabs, keys):
    with tab:
        st.caption(AGENTS[key].purpose)
        if st.button(f"Run {AGENTS[key].name}", key=f"run_{key}"):
            generate(key)
        content = st.session_state.outputs.get(key)
        if content:
            st.markdown(content)
            st.download_button("Download as Markdown", content, file_name=f"{key.replace('_', '-')}.md", mime="text/markdown", key=f"download_{key}")
        else:
            st.info("This section is ready when you are. Run this agent on its own or build the full kit above.")

with tabs[6]:
    st.caption("Search current listings, then rank them against the evidence in your resume.")
    st.markdown("Searches public LinkedIn and Naukri listings, then ranks the results with your selected provider. Groq GPT-OSS uses built-in browser search; Ollama uses public web-search results and ranks them locally.")
    finder_left, finder_right = st.columns([2, 1])
    with finder_left:
        target_roles = st.text_input("Roles to search", value="AI Engineer, Python Backend Engineer", help="Separate job titles with commas.")
        target_location = st.text_input("Preferred location", value="Hyderabad, India", help="Enter a city/country or Remote.")
    with finder_right:
        target_sources = st.multiselect("Job sites", ["LinkedIn", "Naukri"], default=["LinkedIn", "Naukri"])
        st.caption("Groq GPT-OSS uses built-in search; other providers use public web-search results.")
    if st.button("⌕  Find my best matches", type="primary", key="find_jobs"):
        st.session_state.job_matches = ""
        st.session_state.job_search_error = ""
        if provider != "Ollama" and not api_key.strip():
            st.error(f"Add your {provider_label} API key in the sidebar to rank the listings.")
        elif provider == "Ollama" and not model:
            st.error("No Ollama model is available. Start Ollama, install a model, and refresh the local model list.")
        else:
            try:
                with st.spinner("Searching live listings and comparing them with your resume…"):
                    st.session_state.job_matches = find_matching_jobs(
                        resume, target_roles, target_location,
                        ["linkedin.com" if source == "LinkedIn" else "naukri.com" for source in target_sources],
                        provider=provider, api_key=api_key.strip(), base_url=base_url, model=model.strip(),
                    )
                st.session_state.job_search_error = ""
            except Exception as exc:
                st.session_state.job_search_error = str(exc)
    if st.session_state.get("job_search_error"):
        st.error(st.session_state.job_search_error)
    result_markdown = st.session_state.get("job_matches", "")
    if result_markdown:
        st.markdown(result_markdown)
        st.download_button("Download job matches as Markdown", result_markdown, file_name="job-matches.md", mime="text/markdown", key="download_job_matches")

st.markdown("---")
st.caption("AI-generated drafts can be wrong. Confirm dates, metrics, skills, project claims, and every personal detail before applying.")
