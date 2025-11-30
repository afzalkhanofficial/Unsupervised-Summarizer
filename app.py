"""
app.py

Flask app for an unsupervised extractive summarizer using:
  - Sentence-BERT (sentence-transformers)
  - Semantic TextRank (PageRank on SBERT similarity graph)

Features:
  - Accepts PDF uploads or plain text
  - Extracts text from PDFs using pdfplumber
  - Splits into sentences (NLTK)
  - Builds SBERT sentence embeddings
  - Builds similarity graph and runs PageRank -> sentence ranking
  - Returns extractive summary (keeps original order for coherence)
  - Minimal, responsive single-file web UI suitable for deployment on Render.com
"""

import os
import io
import math
import tempfile
from typing import List, Tuple

from flask import Flask, request, render_template_string, abort, send_file, jsonify
import pdfplumber
import nltk
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# ---------- Configuration ----------
MAX_CONTENT_LENGTH = 25 * 1024 * 1024  # 25 MB max upload
ALLOWED_EXTENSIONS = {"pdf", "txt"}
DEFAULT_SUMMARY_RATIO = 0.25  # default fraction of sentences to keep
MIN_SENTENCE_WORDS = 3        # minimum words for a sentence to be considered
SIMILARITY_THRESHOLD = 0.1    # threshold to create graph edges (helps sparsity)
MODEL_NAME = os.environ.get("SBERT_MODEL", "all-MiniLM-L6-v2")  # small & fast model

# ---------- Flask app ----------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# ---------- Ensure NLTK punkt is available ----------
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# ---------- Load SBERT model once at startup ----------
try:
    sbert_model = SentenceTransformer(MODEL_NAME)
except Exception as e:
    # If model download fails, raise early so Render logs show error.
    raise RuntimeError(f"Failed to load SBERT model '{MODEL_NAME}': {e}")

# ---------- Utility functions ----------

def allowed_file(filename: str) -> bool:
    if not filename:
        return False
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """
    Extract text from PDF bytes using pdfplumber.
    Falls back to empty string on error.
    """
    text_parts = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
    except Exception as e:
        # If pdfplumber fails (corrupt PDF, scanned images-only doc), return what we have.
        app.logger.warning(f"pdfplumber extraction failed: {e}")
    return "\n\n".join(text_parts).strip()

def split_into_sentences(text: str) -> List[str]:
    """
    Split using NLTK's sentence tokenizer but also do basic cleanup.
    """
    raw_sentences = nltk.tokenize.sent_tokenize(text)
    sentences = []
    for s in raw_sentences:
        s = s.strip().replace("\n", " ").replace("  ", " ")
        # filter out extremely short sentences
        if len(s.split()) >= MIN_SENTENCE_WORDS:
            sentences.append(s)
    return sentences

def build_similarity_graph(embeddings: np.ndarray, threshold: float = SIMILARITY_THRESHOLD) -> nx.Graph:
    """
    Build an undirected weighted graph where nodes are sentence indices.
    Edge weights are cosine similarity (only if > threshold).
    """
    sim_matrix = cosine_similarity(embeddings)
    n = sim_matrix.shape[0]
    G = nx.Graph()
    for i in range(n):
        G.add_node(i)
    for i in range(n):
        for j in range(i + 1, n):
            score = float(sim_matrix[i, j])
            if score > threshold:
                G.add_edge(i, j, weight=score)
    # If graph has isolated nodes (no edges), connect them weakly to ensure Pagerank runs smoothly
    if G.number_of_edges() == 0:
        # fully connect with small weights (fallback)
        for i in range(n):
            for j in range(i + 1, n):
                G.add_edge(i, j, weight=float(sim_matrix[i, j]) + 1e-6)
    return G

def run_textrank_on_sentences(sentences: List[str], ratio: float = DEFAULT_SUMMARY_RATIO) -> Tuple[List[str], List[int]]:
    """
    Core summarization routine:
      - Encode sentences with SBERT
      - Build similarity graph and run PageRank
      - Select top-k sentences where k = max(1, int(len*summ_ratio))
      - Return selected sentences in their original order and their indices
    """
    if not sentences:
        return [], []

    # Encode sentences (normalize for better cosine behavior)
    embeddings = sbert_model.encode(sentences, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
    # Build similarity graph
    G = build_similarity_graph(embeddings)
    # Run PageRank (weighted)
    try:
        scores = nx.pagerank(G, weight="weight")
    except Exception:
        # Fallback to numpy-based pagerank if networkx fails
        scores = nx.pagerank_numpy(G, weight="weight")

    # Sort sentences by score descending
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    n_sentences = len(sentences)
    k = max(1, int(math.ceil(n_sentences * ratio)))
    top_indices = [idx for idx, _ in ranked[:k]]
    # Keep original order for coherence
    top_indices_sorted = sorted(top_indices)
    summary_sentences = [sentences[i] for i in top_indices_sorted]
    return summary_sentences, top_indices_sorted

# ---------- Routes & UI ----------

INDEX_HTML = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <title>SBERT + Semantic TextRank — Policy Brief Summarizer</title>
    <style>
      body{font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,"Helvetica Neue",Arial;padding:2rem;background:#f3f6fb;color:#111}
      .wrap{max-width:960px;margin:0 auto;background:white;border-radius:12px;padding:1.6rem;box-shadow:0 8px 30px rgba(35,40,50,0.07)}
      header{display:flex;align-items:center;gap:1rem}
      h1{margin:0;font-size:1.3rem}
      p.lead{margin:0;color:#556}
      form{margin-top:1rem;display:grid;gap:.75rem}
      .row{display:flex;gap:.5rem;flex-wrap:wrap}
      label{display:inline-block;padding:.6rem .8rem;background:#f1f5f9;border:1px dashed #e2e8f0;border-radius:8px;cursor:pointer}
      input[type=file]{display:none}
      button{padding:.6rem .9rem;border:0;background:#2563eb;color:white;border-radius:8px;cursor:pointer}
      textarea{width:100%;min-height:160px;padding:.6rem;border-radius:8px;border:1px solid #e6eef8}
      .small{font-size:.9rem;color:#445}
      .result{margin-top:1rem;padding:1rem;border-radius:8px;background:#fcfeff;border:1px solid #e6eef8}
      mark{background: #fff8b0}
      footer{margin-top:1rem;color:#657}
      .controls{display:flex;gap:.5rem;align-items:center}
      input[type=number]{width:80px;padding:.4rem;border-radius:8px;border:1px solid #e6eef8}
    </style>
  </head>
  <body>
    <div class="wrap">
      <header>
        <div>
          <h1>SBERT + Semantic TextRank</h1>
          <p class="lead">Extractive summarizer for policy briefs & healthcare documents (PDF or text)</p>
        </div>
      </header>

      <form id="upload-form" method="post" action="/summarize" enctype="multipart/form-data">
        <div class="row">
          <label for="file">Select PDF / TXT
            <input id="file" name="file" type="file" accept=".pdf,.txt">
          </label>

          <label for="text_input">Or paste text</label>
        </div>

        <textarea id="text_input" name="text" placeholder="Paste policy brief or healthcare document text here (optional)"></textarea>

        <div class="row controls">
          <label class="small">Summary ratio (fraction of sentences to keep):</label>
          <input type="number" name="ratio" step="0.05" min="0.05" max="1" value="0.25">
          <button type="submit">Summarize</button>
        </div>

        <p class="small">Tip: For PDFs, text extraction works best on digital PDFs (not scanned images).</p>
      </form>

      <div id="result"></div>

      <footer>
        <p class="small">Server model: <strong>{{ model_name }}</strong>. This service runs an unsupervised extractive algorithm — no document leaves the server unless you download it.</p>
      </footer>
    </div>

    <script>
      // progressively handle form submission to display result without a full redirect if API returns JSON.
      const form = document.getElementById('upload-form');
      const resultDiv = document.getElementById('result');

      form.addEventListener('submit', async (e) => {
        e.preventDefault();
        resultDiv.innerHTML = "<div class='result small'>Working... this may take a few seconds for large documents.</div>";

        const formData = new FormData(form);
        try {
          const resp = await fetch('/summarize', { method: 'POST', body: formData });
          if (!resp.ok) {
            const txt = await resp.text();
            throw new Error(txt || resp.statusText);
          }
          const data = await resp.json();
          // build html
          let html = `<div class='result'><h3>Extractive Summary</h3>`;
          html += `<p class='small'><strong>Selected ${data.selected_count} of ${data.total_sentences} sentences (ratio=${data.ratio})</strong></p>`;
          html += `<div>`;
          if (data.highlighted_html) {
            html += `<div style="line-height:1.6">${data.highlighted_html}</div>`;
          } else {
            html += `<pre style="white-space:pre-wrap">${data.summary_text}</pre>`;
          }
          html += `</div>`;
          if (data.download_url) {
            html += `<p><a href="${data.download_url}" download="summary.txt">Download summary.txt</a></p>`;
          }
          html += `</div>`;
          resultDiv.innerHTML = html;
        } catch (err) {
          resultDiv.innerHTML = `<div class="result small" style="color:maroon">Error: ${err.message}</div>`;
        }
      });
    </script>
  </body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML, model_name=MODEL_NAME)

@app.route("/summarize", methods=["POST"])
def summarize_route():
    """
    Handle uploads and text submission:
      - Accepts 'file' (pdf or txt) OR 'text' (raw text)
      - parameter 'ratio' controls summary length (0.05 - 1.0)
      - Returns JSON with summary and highlighted HTML
    """
    # Get ratio param
    try:
        ratio = float(request.form.get("ratio", DEFAULT_SUMMARY_RATIO))
        if ratio <= 0 or ratio > 1:
            ratio = DEFAULT_SUMMARY_RATIO
    except Exception:
        ratio = DEFAULT_SUMMARY_RATIO

    uploaded_file = request.files.get("file", None)
    raw_text = (request.form.get("text") or "").strip()

    text = ""
    if uploaded_file and uploaded_file.filename:
        filename = uploaded_file.filename
        if not allowed_file(filename):
            return abort(400, "Only PDF and TXT files are supported.")
        try:
            content = uploaded_file.read()
            ext = filename.rsplit(".", 1)[1].lower()
            if ext == "pdf":
                text = extract_text_from_pdf_bytes(content)
            elif ext == "txt":
                # attempt decode as utf-8, fallback to latin1
                try:
                    text = content.decode("utf-8", errors="replace")
                except Exception:
                    text = content.decode("latin1", errors="replace")
        except Exception as e:
            app.logger.error(f"Error reading uploaded file: {e}")
            return abort(400, "Failed to read uploaded file.")
    elif raw_text:
        text = raw_text
    else:
        return abort(400, "No file or text provided. Please upload a PDF/TXT or paste text.")

    if not text.strip():
        return abort(400, "No text could be extracted from the provided input.")

    # Split into sentences
    sentences = split_into_sentences(text)
    if not sentences:
        return abort(400, "Could not split the document into sentences. Try providing plain text or a digital PDF.")

    # Run SBERT + TextRank
    summary_sentences, selected_indices = run_textrank_on_sentences(sentences, ratio=ratio)

    # Build highlighted HTML: highlight chosen sentences in original full text
    # To produce a readable highlight, we'll reconstruct a "document view" from the list of sentences,
    # placing <mark> tags around selected ones.
    highlighted_parts = []
    selected_set = set(selected_indices)
    for idx, sent in enumerate(sentences):
        safe_sent = escape_html(sent)
        if idx in selected_set:
            highlighted_parts.append(f"<mark>{safe_sent}</mark>")
        else:
            highlighted_parts.append(safe_sent)
    highlighted_html = " ".join(highlighted_parts)

    summary_text = "\n\n".join(summary_sentences)

    # Save summary to a temporary file so the UI can link to it for download
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", prefix="summary_")
    try:
        tmp.write(summary_text.encode("utf-8"))
        tmp.flush()
        tmp_path = tmp.name
    finally:
        tmp.close()

    # Provide a simple download endpoint path (one-shot)
    download_url = f"/download_summary/{os.path.basename(tmp_path)}"

    # Store path to file in a global-safe place (filesystem). The /download_summary route will serve it.
    # Note: in production you'd want a better cleanup policy (e.g., TTL-based worker). For Render testing this is sufficient.
    # We'll keep the file on disk for a short while.
    DOWNLOAD_REGISTRY[os.path.basename(tmp_path)] = tmp_path

    response = {
        "summary_text": summary_text,
        "highlighted_html": highlighted_html,
        "selected_count": len(summary_sentences),
        "total_sentences": len(sentences),
        "ratio": ratio,
        "download_url": download_url,
    }
    return jsonify(response)

# Simple in-memory registry of temp summary files (filename -> path)
DOWNLOAD_REGISTRY = {}

@app.route("/download_summary/<fname>", methods=["GET"])
def download_summary(fname):
    path = DOWNLOAD_REGISTRY.get(fname)
    if not path or not os.path.exists(path):
        return abort(404, "File not found or expired.")
    return send_file(path, as_attachment=True, download_name="summary.txt")

# ---------- Helpers ----------
def escape_html(s: str) -> str:
    """Minimal HTML escaping for markup safety in the simple UI."""
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
