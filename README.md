# ApplyFlow — Job Application Studio

A Python + Streamlit application with seven focused agents and selectable Groq, OpenAI, Anthropic, Google Gemini, or local Ollama models.

## What it does

- **Job Description Analyst** extracts role requirements and keywords.
- **Candidate Match Evaluator** compares documented resume evidence with the role.
- **Resume Tailor** rewrites bullets without adding unsupported experience.
- **Cover Letter Writer** drafts a role-specific letter and marks missing personal context.
- **Accuracy and Quality Reviewer** flags claims that need evidence or confirmation.
- **Interview Coach** generates role-specific questions and study exercises.
- **Job Finder** searches public LinkedIn and Naukri listings and ranks them against your resume evidence.

The project Markdown files are loaded as example inputs on first launch. Replace them with your own resume and job description, or upload `.pdf`, `.docx`, `.md`, or `.txt` files. Uploaded content appears in the editable text areas. Scanned PDFs without selectable text need OCR or a pasted text version. Outputs are available in the app and can be downloaded as Markdown.

## Run locally

1. Use Python 3.10 or newer.
2. Install dependencies: `python -m pip install -r requirements.txt`
3. Set the API key for your chosen cloud provider in the environment, or enter it in the app sidebar. Ollama does not require a key for local use.
4. Choose a provider in the sidebar. Enter that provider's API key and press **Load available models** to populate its model selector. For Ollama, start its local server; ApplyFlow reads installed models from `http://localhost:11434/api/tags` and lists them automatically. Change the server URL if Ollama runs elsewhere.
5. Start the app: `streamlit run app.py`

Create keys for the APIs you use: [Groq](https://console.groq.com/keys), [OpenAI](https://platform.openai.com/api-keys), [Anthropic](https://console.anthropic.com/settings/keys), and [Google Gemini](https://aistudio.google.com/app/apikey). ChatGPT, Claude, and Gemini consumer subscriptions are separate from API access and billing. Ollama runs locally and does not need an API key; its [local API](https://docs.ollama.com/api/tags) provides the installed model list. Groq's built-in browser search in Job Finder supports `openai/gpt-oss-120b` and `openai/gpt-oss-20b`; other providers use public web search snippets, then rank them with the selected model. [Groq browser search docs](https://console.groq.com/docs/tool-use/built-in-tools/browser-search)

## Privacy and accuracy

With a cloud provider selected, resume text and prompts are sent to that provider. With Ollama selected, prompts go to the configured Ollama server; for Job Finder, search terms go to public web search and result snippets are ranked by the local model. Public listings can be partial or stale, and some pages require a LinkedIn or Naukri sign-in. This app does not persist uploaded files or API keys to disk. Keep API keys in environment variables or the sidebar, never commit live keys to the repository, and verify availability and application details on the job board.
