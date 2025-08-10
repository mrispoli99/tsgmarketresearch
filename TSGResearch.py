#!/usr/bin/env python3
"""
Market Research Assistant — Streamlit + Gemini 2.5 (clean UX)

Updates in this version (per feedback):
- Light theme (no dark background)
- Simple web search (removed include/exclude domain filters)
- Removed the sidebar pip-install reminder section
- More robust Gemini error handling with helpful messages & an automatic retry

Features
- Upload documents (PDF, DOCX, PPTX, TXT, CSV, XLSX)
- Search the web (SerpAPI Google search) with simple recency & count controls
- Do either or both in one run
- Generates a comprehensive summary with inline citations like [D1], [S2]
- Shows a bibliography with links and file names
- Lets you preview extracted text per source
- Download the report as Markdown

Setup
1) Python 3.10+
2) pip install -r requirements.txt  (see REQUIRED_PACKAGES below)
3) Set API keys via Streamlit Secrets or environment variables:
   - GEMINI_API_KEY
   - SERPAPI_API_KEY (for web search)

Run
   streamlit run market_research_assistant.py
"""

import os
import io
import re
import time
import json
import textwrap
import datetime as dt
from typing import List, Dict, Optional, Tuple

# --- UI / App ---
try:
    import streamlit as st
except Exception as e:
    raise SystemExit("Streamlit is required. Install with: pip install streamlit")

# --- External deps with graceful fallbacks ---
MISSING_DEPS = []

def _soft_import(pkg: str, pip_name: Optional[str] = None):
    try:
        return __import__(pkg)
    except Exception:
        MISSING_DEPS.append(pip_name or pkg)
        return None

requests = _soft_import("requests")
bs4 = _soft_import("bs4")
pypdf = _soft_import("pypdf")
docx = _soft_import("docx")  # python-docx
pptx = _soft_import("pptx")  # python-pptx
pandas = _soft_import("pandas")

# Google Gemini SDK
try:
    import google.generativeai as genai
except Exception:
    MISSING_DEPS.append("google-generativeai")
    genai = None

# Constants
APP_TITLE = "TSG Market Research Assistant"
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
REQUIRED_PACKAGES = [
    "streamlit",
    "google-generativeai",
    "requests",
    "beautifulsoup4",
    "pypdf",
    "python-docx",
    "python-pptx",
    "pandas",
]

# Keep prompt sizes sane
MAX_CHARS_PER_SOURCE = 20_000
MAX_SOURCES = 20

# ---------- Utilities ----------

def get_secret(name: str, prompt_label: str, help_text: str = "") -> Optional[str]:
    """Retrieve secret from st.secrets or env; fall back to a password input box."""
    if name in st.secrets:
        return st.secrets[name]
    if os.environ.get(name):
        return os.environ.get(name)
    return st.sidebar.text_input(prompt_label, type="password", help=help_text)


# ---------- Web Search (SerpAPI) ----------

def serpapi_search(api_key: str, query: str, num_results: int = 8, recency_days: Optional[int] = None,
                   backoff_retries: int = 3) -> List[Dict]:
    if requests is None:
        st.error("The 'requests' package is required for web search. Install it and retry.")
        return []
    params = {
        "engine": "google",
        "q": query,
        "num": max(1, min(num_results, 10)),
        "hl": "en",
        "gl": "us",
        "api_key": api_key,
    }
    if recency_days:
        if recency_days <= 1:
            params["tbs"] = "qdr:d"
        elif recency_days <= 7:
            params["tbs"] = "qdr:w"
        elif recency_days <= 31:
            params["tbs"] = "qdr:m"
        else:
            params["tbs"] = "qdr:y"

    url = "https://serpapi.com/search.json"
    results = []

    for attempt in range(backoff_retries):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                data = r.json()
                news = data.get("news_results", []) or []
                organic = data.get("organic_results", []) or []
                cards = []
                for item in news:
                    cards.append({
                        "title": item.get("title"),
                        "link": item.get("link") or item.get("url"),
                        "snippet": item.get("snippet"),
                        "source": item.get("source"),
                        "date": item.get("date"),
                        "type": "news",
                    })
                for item in organic:
                    cards.append({
                        "title": item.get("title"),
                        "link": item.get("link") or item.get("url"),
                        "snippet": item.get("snippet"),
                        "source": item.get("source") or item.get("displayed_link"),
                        "date": item.get("date"),
                        "type": "web",
                    })
                for c in cards:
                    if c.get("link"):
                        results.append(c)
                    if len(results) >= num_results:
                        break
                break
            else:
                time.sleep(1 + attempt)
        except Exception:
            time.sleep(1 + attempt)
    return results


def fetch_and_extract(url: str, timeout: int = 30) -> Tuple[Optional[str], Optional[str]]:
    if requests is None or bs4 is None:
        return None, None
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return None, None
        html = r.text
        soup = bs4.BeautifulSoup(html, "html.parser")
        title = None
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "form", "aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        text = re.sub(r"\s+", " ", text).strip()
        return text, title
    except Exception:
        return None, None


# ---------- Document Parsing ----------

def read_pdf(file: io.BytesIO) -> str:
    if pypdf is None:
        raise RuntimeError("Install 'pypdf' to read PDFs: pip install pypdf")
    try:
        reader = pypdf.PdfReader(file)
        parts = []
        for page in reader.pages:
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                continue
        return "\n".join(parts)
    except Exception as e:
        return f"[PDF extraction error] {e}"


def read_docx(file: io.BytesIO) -> str:
    if docx is None:
        raise RuntimeError("Install 'python-docx' to read DOCX: pip install python-docx")
    try:
        document = docx.Document(file)
        return "\n".join(p.text for p in document.paragraphs)
    except Exception as e:
        return f"[DOCX extraction error] {e}"


def read_pptx(file: io.BytesIO) -> str:
    if pptx is None:
        raise RuntimeError("Install 'python-pptx' to read PPTX: pip install python-pptx")
    try:
        prs = pptx.Presentation(file)
        texts = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "has_text_frame") and shape.has_text_frame:
                    texts.append(shape.text)
        return "\n".join(texts)
    except Exception as e:
        return f"[PPTX extraction error] {e}"


def read_text(file: io.BytesIO) -> str:
    try:
        return file.read().decode("utf-8", errors="ignore")
    except Exception:
        try:
            return file.read().decode("latin-1", errors="ignore")
        except Exception as e:
            return f"[TXT read error] {e}"


def read_tabular(file: io.BytesIO, filename: str) -> str:
    if pandas is None:
        raise RuntimeError("Install 'pandas' to read CSV/XLSX: pip install pandas openpyxl")
    try:
        if filename.lower().endswith(".csv"):
            df = pandas.read_csv(file)
        else:
            df = pandas.read_excel(file)
        head = df.head(15).to_markdown(index=False)
        desc = df.describe(include="all", datetime_is_numeric=True).to_markdown()
        return f"Table Preview (first 15 rows)\n\n{head}\n\nStats\n\n{desc}"
    except Exception as e:
        return f"[Tabular read error] {e}"


def extract_from_upload(file_obj) -> str:
    name = file_obj.name.lower()
    data = io.BytesIO(file_obj.read())
    if name.endswith(".pdf"):
        return read_pdf(data)
    elif name.endswith(".docx"):
        return read_docx(data)
    elif name.endswith(".pptx"):
        return read_pptx(data)
    elif name.endswith(".txt"):
        return read_text(data)
    elif name.endswith(".csv") or name.endswith(".xlsx"):
        return read_tabular(data, name)
    else:
        return "[Unsupported file type]"


# ---------- Gemini ----------

def configure_gemini(api_key: str):
    if genai is None:
        raise RuntimeError("google-generativeai SDK not installed. pip install google-generativeai")
    genai.configure(api_key=api_key)


def build_source_map(doc_blobs: List[Dict], web_blobs: List[Dict]) -> Tuple[List[Dict], Dict[str, Dict]]:
    sources = []
    id_map = {}

    for i, d in enumerate(doc_blobs, start=1):
        sid = f"D{i}"
        src = {
            "id": sid,
            "kind": "doc",
            "title": d.get("title") or d.get("name") or f"Document {i}",
            "name": d.get("name"),
            "url": None,
            "date": None,
            "text": (d.get("text") or "")[:MAX_CHARS_PER_SOURCE],
        }
        sources.append(src)
        id_map[sid] = src

    for j, w in enumerate(web_blobs, start=1):
        sid = f"S{j}"
        src = {
            "id": sid,
            "kind": "web",
            "title": w.get("title") or w.get("name") or f"Web Result {j}",
            "name": w.get("title") or f"Web Result {j}",
            "url": w.get("link"),
            "date": w.get("date"),
            "text": (w.get("text") or "")[:MAX_CHARS_PER_SOURCE],
        }
        sources.append(src)
        id_map[sid] = src

    return sources[:MAX_SOURCES], id_map


def build_prompt(topic: str, sources: List[Dict], report_style: str = "balanced") -> List[Dict]:
    style_note = {
        "balanced": "balanced tone (concise but thorough)",
        "executive": "executive brief (bulleted, crisp)",
        "analyst": "analyst deep-dive (denser detail)",
    }.get(report_style, "balanced tone")

    header = textwrap.dedent(f"""
    You are a meticulous market research analyst. Produce a single cohesive report on: "{topic}".

    RULES
    - Only use the SOURCES provided below. Do not invent facts.
    - Insert inline citations using the source IDs like [D1] or [S2] immediately after the sentences they support.
    - If multiple sources support a claim, you may include multiple IDs like [D1; S2].
    - If a claim is uncertain or contradictory across sources, call it out and cite the differing sources.
    - If information is missing, say so explicitly.

    OUTPUT FORMAT (Markdown)
    # Title
    **Date:** {dt.date.today().isoformat()}
    **Scope:** {style_note}

    ## Key Takeaways (5–8 bullets)
    - ...

    ## Market Overview

    ## Trends & Drivers

    ## Competitive Landscape

    ## Risks & Unknowns

    ## Signals & Notable Quotes

    ## Appendix: Source Notes

    Then, include a **Bibliography** mapping each ID to its title and URL or filename.
    """)

    catalog_lines = []
    for s in sources:
        title = s.get("title") or s.get("name") or s["id"]
        url = s.get("url") or s.get("name") or "(local document)"
        date = s.get("date") or ""
        catalog_lines.append(f"- [{s['id']}] {title} — {url} {('('+date+')') if date else ''}")

    catalog = "\n".join(catalog_lines)

    system = (
        header
        + "\n\nSOURCES CATALOG\n" + catalog
        + "\n\nNow you will receive the full texts in a series of blocks. After all blocks, produce the report."
    )

    prompt = [{"role": "user", "parts": [system]}]

    for s in sources:
        block_header = f"BEGIN SOURCE {s['id']} — {s.get('title') or s.get('name') or s['id']}\n"
        block_footer = f"\nEND SOURCE {s['id']}\n"
        content = block_header + (s.get("text") or "[empty]") + block_footer
        prompt.append({"role": "user", "parts": [content]})

    prompt.append({"role": "user", "parts": ["When ready, write the report now following the OUTPUT FORMAT strictly and include citations."]})
    return prompt


def _extract_text_from_response(resp) -> Tuple[Optional[str], Optional[str]]:
    """Return (text, finish_reason) or (None, reason) if not available."""
    try:
        if hasattr(resp, "text") and resp.text:
            return resp.text, "STOP"
        # SDK variants
        if getattr(resp, "candidates", None):
            for c in resp.candidates:
                # Prefer candidate with parts/text
                try:
                    parts = getattr(c, "content", None).parts if getattr(c, "content", None) else []
                    if parts:
                        # Some SDK versions expose .text on each part
                        texts = []
                        for p in parts:
                            t = getattr(p, "text", None)
                            if t:
                                texts.append(t)
                        if texts:
                            return "\n".join(texts), getattr(c, "finish_reason", None) or getattr(c, "finishReason", None)
                except Exception:
                    continue
            # If we got here, no candidate had parts
            reason = getattr(resp.candidates[0], "finish_reason", None) or getattr(resp.candidates[0], "finishReason", None)
            return None, reason or "UNKNOWN"
    except Exception:
        return None, "EXCEPTION"
    return None, "EMPTY"


def call_gemini(prompt_messages: List[Dict], model_name: str, temperature: float = 0.2,
                max_output_tokens: int = 8192) -> str:
    model = genai.GenerativeModel(model_name)
    # Attempt 1
    try:
        resp = model.generate_content(
            prompt_messages,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
            },
        )
    except Exception as e:
        return f"[Gemini error] {e}"

    text, reason = _extract_text_from_response(resp)
    if text:
        return text

    # Attempt 2 (fallback): reduce tokens & temperature, try again
    try:
        resp2 = model.generate_content(
            prompt_messages,
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": max(4096, int(max_output_tokens * 0.6)),
            },
        )
        text2, reason2 = _extract_text_from_response(resp2)
        if text2:
            return text2
        return f"[Gemini error] No content returned. finish_reason={reason2}"
    except Exception as e:
        return f"[Gemini error] {e} (initial finish_reason={reason})"


# ---------- Streamlit UI ----------

st.set_page_config(page_title=APP_TITLE, page_icon="🕵️", layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.header("🔧 Settings")
    st.caption("Provide keys here or via Streamlit Secrets / environment variables.")
    gemini_key = get_secret("GEMINI_API_KEY", "Gemini API Key", "Stored only in session state.")
    serp_key = get_secret("SERPAPI_API_KEY", "SerpAPI Key (for web search)")

    model_name = st.text_input("Gemini model", value=DEFAULT_MODEL, help="e.g., gemini-2.5-flash")
    report_style = st.selectbox("Report style", ["balanced", "executive", "analyst"], index=0)

# Topic & mode
colA, colB = st.columns([2, 1])
with colA:
    topic = st.text_input("Research topic / question", placeholder="e.g., US skincare DTC market trends in 2025")
with colB:
    mode = st.radio("Data sources", ["Uploaded documents", "Web articles", "Both"], index=2)

# Uploads
uploads = st.file_uploader(
    "Upload documents (PDF, DOCX, PPTX, TXT, CSV, XLSX)",
    type=["pdf", "docx", "pptx", "txt", "csv", "xlsx"],
    accept_multiple_files=True,
)

# Web search controls (simple)
with st.expander("Web search options"):
    col1, col2 = st.columns([2,1])
    with col1:
        query = st.text_input("Search query", placeholder="Enter search terms…")
    with col2:
        num_results = st.number_input("Max results", 1, 10, 6)
        recency_days = st.selectbox("Recency filter", [None, 1, 7, 30, 365], index=2)

# Run button
go = st.button("🚀 Run Research", type="primary")

# --- Main pipeline ---
if go:
    if not topic:
        st.error("Please provide a topic/question.")
        st.stop()

    # Configure Gemini
    if gemini_key:
        try:
            configure_gemini(gemini_key)
        except Exception as e:
            st.error(f"Problem configuring Gemini: {e}")
            st.stop()
    else:
        st.error("Gemini API key is required.")
        st.stop()

    want_docs = mode in ("Uploaded documents", "Both")
    want_web = mode in ("Web articles", "Both")

    doc_blobs: List[Dict] = []
    web_blobs: List[Dict] = []

    # --- Documents ---
    if want_docs:
        if not uploads:
            st.info("No documents uploaded.")
        else:
            st.subheader("📄 Uploaded Documents")
            doc_rows = []
            for f in uploads:
                with st.spinner(f"Extracting: {f.name}"):
                    text = extract_from_upload(f)
                doc_blobs.append({
                    "name": f.name,
                    "title": f.name,
                    "text": text,
                })
                doc_rows.append({"File": f.name, "Chars": len(text)})
            if pandas is not None and doc_rows:
                st.dataframe(pandas.DataFrame(doc_rows))
            else:
                for r in doc_rows:
                    st.write(r)

    # --- Web search ---
    if want_web:
        if not serp_key:
            st.error("Web mode requires a SerpAPI key. Provide it in the sidebar.")
        elif not query:
            st.error("Enter a search query in 'Web search options'.")
        else:
            st.subheader("🌐 Web Results")
            with st.spinner("Searching the web…"):
                cards = serpapi_search(serp_key, query, num_results=int(num_results), recency_days=recency_days)

            if not cards:
                st.warning("No web results returned.")
            else:
                rows = []
                for c in cards:
                    url = c.get("link")
                    title = c.get("title") or url
                    with st.spinner(f"Fetching: {title}"):
                        text, page_title = fetch_and_extract(url)
                    if page_title and not title:
                        title = page_title
                    web_blobs.append({
                        "title": title,
                        "link": url,
                        "date": c.get("date"),
                        "text": text or "",
                    })
                    rows.append({"Title": title, "URL": url, "Type": c.get("type"), "Chars": len(text or "")})

                if pandas is not None and rows:
                    st.dataframe(pandas.DataFrame(rows))
                else:
                    for r in rows:
                        st.write(r)

    # --- Build sources & prompt ---
    sources, id_map = build_source_map(doc_blobs, web_blobs)

    if not sources:
        st.error("No sources available. Upload docs and/or run a web search.")
        st.stop()

    # Source previews
    st.markdown("---")
    st.subheader("🔎 Source Previews")
    for s in sources:
        label = f"{s['id']}: {s.get('title') or s.get('name')}"
        with st.expander(label):
            if s.get("url"):
                st.markdown(f"**URL:** {s['url']}")
            st.text(s.get("text", "")[:4000] or "[No text extracted]")

    # Generate report with Gemini
    st.markdown("---")
    st.subheader("📝 Research Report")
    prompt = build_prompt(topic, sources, report_style=report_style)

    with st.spinner("Summarizing with Gemini…"):
        report_md = call_gemini(prompt, model_name=model_name)

    if report_md.startswith("[Gemini error]"):
        st.error(report_md)
        st.info("Tip: If this says finish_reason=MAX_TOKENS or similar, try reducing sources, shortening long PDFs, or switch report style to 'executive'.")
    else:
        st.markdown(report_md)

        # Bibliography
        st.markdown("---")
        st.subheader("📚 Bibliography")
        for s in sources:
            title = s.get("title") or s.get("name") or s["id"]
            url = s.get("url")
            if url:
                st.markdown(f"**[{s['id']}]** {title} — {url}")
            else:
                st.markdown(f"**[{s['id']}]** {title} — (uploaded document)")

        # Download
        st.markdown("---")
        st.download_button(
            label="💾 Download report (Markdown)",
            data=report_md.encode("utf-8"),
            file_name=f"research_report_{dt.date.today().isoformat()}.md",
            mime="text/markdown",
        )

# (No custom CSS — default Streamlit light theme)
