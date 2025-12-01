import io
import os
import re

import numpy as np
import networkx as nx
from flask import Flask, request, render_template_string, abort
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader

# -------------------- Flask App -------------------- #

app = Flask(__name__)

# -------------------- HTML Templates -------------------- #

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Policy Brief Summarizer - Primary Healthcare</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
      :root {
        --primary-color: #2563eb;
        --secondary-color: #1d4ed8;
        --bg-color: #0f172a;
        --card-bg: #ffffff;
        --accent: #22c55e;
        --danger: #ef4444;
      }

      * {
        box-sizing: border-box;
        margin: 0;
        padding: 0;
      }

      body {
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background: radial-gradient(circle at top left,#2563eb,#0f172a 40%,#020617);
        min-height: 100vh;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 1.5rem;
        color: #0f172a;
      }

      .container {
        width: 100%;
        max-width: 900px;
        background: rgba(255,255,255,0.98);
        border-radius: 1.5rem;
        padding: 2rem 2.5rem;
        box-shadow: 0 25px 80px rgba(15,23,42,0.45);
        position: relative;
        overflow: hidden;
      }

      .badge {
        position: absolute;
        top: 1.25rem;
        right: 1.5rem;
        font-size: 0.75rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        border: 1px solid rgba(37,99,235,0.25);
        background: rgba(239,246,255,0.9);
        color: #1d4ed8;
      }

      h1 {
        font-size: 1.9rem;
        margin-bottom: 0.4rem;
        color: #0f172a;
      }

      .subtitle {
        font-size: 0.95rem;
        color: #64748b;
        margin-bottom: 1.5rem;
      }

      .subtitle span {
        background: #eff6ff;
        color: #1d4ed8;
        padding: 0.15rem 0.5rem;
        border-radius: 999px;
        font-size: 0.8rem;
        margin-left: 0.4rem;
      }

      form {
        margin-top: 0.75rem;
      }

      .grid {
        display: grid;
        grid-template-columns: 1.5fr 1fr;
        gap: 1.5rem;
      }

      .upload-area {
        border: 2px dashed rgba(148,163,184,0.9);
        border-radius: 1rem;
        padding: 1.25rem;
        text-align: center;
        background: #f8fafc;
        transition: border-color 0.2s ease, background 0.2s ease, transform 0.15s ease;
        cursor: pointer;
      }

      .upload-area.dragover {
        border-color: var(--primary-color);
        background: #eff6ff;
        transform: translateY(-1px);
      }

      .upload-area h2 {
        font-size: 1rem;
        margin-bottom: 0.25rem;
      }

      .upload-area p {
        font-size: 0.85rem;
        color: #64748b;
      }

      .upload-area span.browse {
        color: var(--primary-color);
        font-weight: 600;
        cursor: pointer;
      }

      .file-meta {
        margin-top: 0.75rem;
        font-size: 0.85rem;
        color: #0f172a;
        background: #e5f0ff;
        padding: 0.4rem 0.7rem;
        border-radius: 999px;
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
      }

      .file-meta strong {
        font-weight: 600;
      }

      .file-meta small {
        color: #4b5563;
      }

      .options-card {
        background: #f9fafb;
        border-radius: 1rem;
        padding: 1rem 1.25rem;
        border: 1px solid #e5e7eb;
      }

      .options-card h3 {
        font-size: 0.95rem;
        margin-bottom: 0.75rem;
        color: #111827;
      }

      .radio-group {
        display: flex;
        flex-direction: column;
        gap: 0.35rem;
        font-size: 0.87rem;
        color: #4b5563;
      }

      .radio-item {
        display: flex;
        align-items: center;
        gap: 0.4rem;
      }

      .radio-item input {
        accent-color: var(--primary-color);
      }

      .radio-item span {
        font-weight: 500;
        color: #111827;
      }

      .helper {
        font-size: 0.78rem;
        color: #6b7280;
        margin-top: 0.6rem;
      }

      .submit-row {
        display: flex;
        justify-content: flex-end;
        margin-top: 1.5rem;
      }

      button[type="submit"] {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        border: none;
        color: white;
        font-weight: 600;
        font-size: 0.96rem;
        padding: 0.75rem 1.4rem;
        border-radius: 999px;
        cursor: pointer;
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        box-shadow: 0 12px 30px rgba(37,99,235,0.35);
        transition: transform 0.15s ease, box-shadow 0.15s ease, filter 0.15s ease;
      }

      button[type="submit"]:hover {
        transform: translateY(-1px);
        filter: brightness(1.05);
        box-shadow: 0 16px 40px rgba(37,99,235,0.45);
      }

      button[type="submit"]:active {
        transform: translateY(0);
        box-shadow: 0 10px 25px rgba(37,99,235,0.35);
      }

      .footer-note {
        margin-top: 1rem;
        font-size: 0.78rem;
        color: #6b7280;
      }

      .footer-note code {
        background: #f3f4f6;
        padding: 0.15rem 0.35rem;
        border-radius: 0.35rem;
        font-size: 0.78rem;
      }

      @media (max-width: 768px) {
        .container {
          padding: 1.5rem 1.25rem;
          border-radius: 1.25rem;
        }

        .grid {
          grid-template-columns: 1fr;
        }

        .badge {
          position: static;
          margin-bottom: 0.75rem;
        }
      }
    </style>
</head>
<body>
  <div class="container">
    <div class="badge">Unsupervised • Extractive</div>
    <h1>Policy Brief Summarizer</h1>
    <p class="subtitle">
      Automatic extractive summarization tailored for primary healthcare policy briefs.
      <span>TF-IDF · TextRank · MMR</span>
    </p>

    <form id="upload-form" action="{{ url_for('summarize') }}" method="post" enctype="multipart/form-data">
      <div class="grid">
        <div>
          <label for="file-input">
            <div id="upload-area" class="upload-area">
              <h2>Upload Policy Brief</h2>
              <p>
                Drag & drop a <strong>PDF</strong> or <strong>.txt</strong> file here<br>
                or <span class="browse">browse from your system</span>
              </p>
              <input id="file-input" type="file" name="file" accept=".pdf,.txt" hidden required>
              <div id="file-meta" class="file-meta" style="display:none;">
                <strong id="file-name"></strong>
                <small id="file-size"></small>
              </div>
            </div>
          </label>
        </div>

        <div>
          <div class="options-card">
            <h3>Summary Length</h3>
            <div class="radio-group">
              <label class="radio-item">
                <input type="radio" name="length" value="short">
                <span>Short</span> (≈ 10–15% of sentences, max ~5)
              </label>
              <label class="radio-item">
                <input type="radio" name="length" value="medium" checked>
                <span>Medium</span> (≈ 20–25% of sentences, max ~10)
              </label>
              <label class="radio-item">
                <input type="radio" name="length" value="long">
                <span>Long</span> (≈ 30–35% of sentences, max ~15)
              </label>
            </div>
            <p class="helper">
              The system uses TF-IDF embeddings, a cosine-similarity graph + TextRank to score sentences,
              and MMR to reduce redundancy before reordering selected sentences.
            </p>
          </div>
        </div>
      </div>

      <div class="submit-row">
        <button type="submit">
          🔍 Generate Summary
        </button>
      </div>
    </form>

    <p class="footer-note">
      Note: Works best with structured policy briefs and reports (primary healthcare, guidelines, frameworks, etc.).
      Large PDFs are automatically converted to text using <code>PyPDF2</code>.
    </p>
  </div>

  <script>
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const fileMeta = document.getElementById('file-meta');
    const fileNameEl = document.getElementById('file-name');
    const fileSizeEl = document.getElementById('file-size');

    uploadArea.addEventListener('click', function() {
      fileInput.click();
    });

    uploadArea.addEventListener('dragover', function(e) {
      e.preventDefault();
      uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', function(e) {
      e.preventDefault();
      uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', function(e) {
      e.preventDefault();
      uploadArea.classList.remove('dragover');
      const files = e.dataTransfer.files;
      if (files && files.length > 0) {
        fileInput.files = files;
        updateFileMeta(files[0]);
      }
    });

    fileInput.addEventListener('change', function(e) {
      if (e.target.files && e.target.files.length > 0) {
        updateFileMeta(e.target.files[0]);
      }
    });

    function updateFileMeta(file) {
      fileNameEl.textContent = file.name;
      const sizeKB = file.size / 1024;
      if (sizeKB < 1024) {
        fileSizeEl.textContent = "(" + sizeKB.toFixed(1) + " KB)";
      } else {
        fileSizeEl.textContent = "(" + (sizeKB / 1024).toFixed(2) + " MB)";
      }
      fileMeta.style.display = 'inline-flex';
    }
  </script>
</body>
</html>
"""

RESULT_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Summary • Policy Brief Summarizer</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
      :root {
        --primary-color: #2563eb;
        --secondary-color: #1d4ed8;
        --bg-color: #0f172a;
        --accent: #22c55e;
        --muted: #6b7280;
      }

      * {
        box-sizing: border-box;
        margin: 0;
        padding: 0;
      }

      body {
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background: radial-gradient(circle at top left,#2563eb,#0f172a 40%,#020617);
        min-height: 100vh;
        padding: 1.5rem;
        display: flex;
        align-items: center;
        justify-content: center;
      }

      .container {
        width: 100%;
        max-width: 1000px;
        background: rgba(255,255,255,0.98);
        border-radius: 1.5rem;
        padding: 2rem 2.5rem;
        box-shadow: 0 25px 80px rgba(15,23,42,0.5);
      }

      .header {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        align-items: flex-start;
        margin-bottom: 1.5rem;
      }

      h1 {
        font-size: 1.6rem;
        color: #0f172a;
      }

      .pill {
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        background: #eff6ff;
        color: #1d4ed8;
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        font-size: 0.78rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
      }

      .stats {
        display: flex;
        flex-wrap: wrap;
        gap: 0.75rem;
        margin-bottom: 1.25rem;
        font-size: 0.82rem;
        color: #4b5563;
      }

      .stat-chip {
        background: #f3f4f6;
        border-radius: 999px;
        padding: 0.25rem 0.65rem;
      }

      .stat-chip strong {
        color: #111827;
      }

      .summary-card {
        background: #f9fafb;
        border-radius: 1rem;
        border: 1px solid #e5e7eb;
        padding: 1rem 1.25rem;
        max-height: 60vh;
        overflow-y: auto;
      }

      .summary-card h2 {
        font-size: 0.98rem;
        margin-bottom: 0.65rem;
        color: #111827;
      }

      .summary-text {
        font-size: 0.95rem;
        line-height: 1.6;
        color: #111827;
        white-space: pre-wrap;
      }

      .note {
        margin-top: 0.9rem;
        font-size: 0.78rem;
        color: #6b7280;
      }

      .note strong {
        color: #111827;
      }

      .actions {
        margin-top: 1.75rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1rem;
        flex-wrap: wrap;
      }

      .back-btn {
        background: #0f172a;
        color: white;
        border-radius: 999px;
        padding: 0.6rem 1.4rem;
        font-size: 0.9rem;
        font-weight: 500;
        text-decoration: none;
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        box-shadow: 0 10px 25px rgba(15,23,42,0.4);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
      }

      .back-btn:hover {
        transform: translateY(-1px);
        box-shadow: 0 14px 30px rgba(15,23,42,0.5);
      }

      .algo-pill {
        font-size: 0.78rem;
        color: var(--muted);
      }

      .algo-pill code {
        background: #f3f4f6;
        padding: 0.1rem 0.35rem;
        border-radius: 0.35rem;
        font-size: 0.78rem;
      }

      @media(max-width: 768px) {
        .container {
          padding: 1.5rem 1.25rem;
        }

        .header {
          flex-direction: column;
        }
      }
    </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div>
        <h1>Extractive Summary</h1>
        <div class="pill">Primary Healthcare • Policy Brief</div>
      </div>
    </div>

    <div class="stats">
      <div class="stat-chip">
        <strong>{{ stats.summary_sentences }}</strong> sentences in summary
      </div>
      <div class="stat-chip">
        <strong>{{ stats.original_sentences }}</strong> sentences in original
      </div>
      <div class="stat-chip">
        Compression: <strong>{{ stats.compression_ratio }}%</strong>
      </div>
      <div class="stat-chip">
        Characters: <strong>{{ stats.summary_chars }}</strong> / {{ stats.original_chars }}
      </div>
    </div>

    <div class="summary-card">
      <h2>Summary</h2>
      <div class="summary-text">{{ summary }}</div>
    </div>

    <p class="note">
      <strong>How this was generated:</strong> Sentences were vectorized using TF-IDF.
      A cosine-similarity graph was constructed, and TextRank (PageRank) was applied
      to estimate sentence importance. Maximal Marginal Relevance (MMR) was then used
      to remove redundancy and increase coverage, before restoring chronological order.
    </p>

    <div class="actions">
      <a href="{{ url_for('index') }}" class="back-btn">
        ← Summarize another document
      </a>
      <div class="algo-pill">
        Using <code>TF-IDF</code> + <code>TextRank</code> + <code>MMR</code> (Unsupervised Extractive)
      </div>
    </div>
  </div>
</body>
</html>
"""

# -------------------- Text Utilities -------------------- #


def normalize_whitespace(text: str) -> str:
    """Basic whitespace normalization."""
    text = text.replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_into_sentences(text: str):
    """
    Lightweight rule-based sentence splitter.
    Designed to avoid external model downloads (no NLTK data).
    """
    # Ensure newline-separated paragraphs don't break logic
    text = re.sub(r"\n+", " ", text)

    # Split on sentence-ending punctuation followed by space + capital letter or end of string
    sentence_endings = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
    rough_sentences = sentence_endings.split(text)

    sentences = []
    for s in rough_sentences:
        s = s.strip()
        if len(s) < 2:
            continue
        # Remove stray bullet characters commonly seen in policy briefs
        s = re.sub(r"^[•\-\–\*]+\s*", "", s)
        if s:
            sentences.append(s)
    return sentences


# -------------------- PDF Extraction -------------------- #


def extract_text_from_pdf(file_storage) -> str:
    """
    Extracts text from an uploaded PDF using PyPDF2.
    """
    try:
        raw_bytes = file_storage.read()
        reader = PdfReader(io.BytesIO(raw_bytes))
        texts = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            texts.append(page_text)
        return "\n".join(texts)
    except Exception as e:
        raise RuntimeError(f"Error reading PDF: {e}")


# -------------------- Summarization Core (TF-IDF + TextRank + MMR) -------------------- #


def build_tfidf_matrix(sentences):
    """
    Build TF-IDF matrix for sentences.
    For policy briefs, we allow unigrams + bigrams and remove English stopwords.
    """
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_df=0.9,
        min_df=1
    )
    tfidf_matrix = vectorizer.fit_transform(sentences)
    return tfidf_matrix


def compute_textrank_scores(sim_matrix: np.ndarray):
    """
    Build a graph from cosine similarity matrix and run TextRank (PageRank).
    """
    np.fill_diagonal(sim_matrix, 0.0)  # Avoid self-loops biasing PageRank
    graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(graph, max_iter=200, tol=1e-6)
    return scores


def mmr_selection(scores_dict, sim_matrix, summary_size, lambda_param=0.7):
    """
    Maximal Marginal Relevance (MMR) for redundancy reduction.
    - scores_dict: TextRank scores (importance) for each sentence index.
    - sim_matrix: cosine similarity matrix between sentences.
    - summary_size: number of sentences to select.
    - lambda_param: trade-off between relevance and diversity (0–1).
    """
    n_sentences = sim_matrix.shape[0]
    indices = np.arange(n_sentences)

    # Convert scores dict to ordered numpy array
    scores = np.array([scores_dict.get(i, 0.0) for i in range(n_sentences)], dtype=float)
    # Normalize scores to [0, 1]
    if scores.max() > 0:
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
    else:
        scores[:] = 1.0 / n_sentences

    selected = []
    candidate_idxs = set(indices.tolist())

    if summary_size >= n_sentences:
        return list(indices)

    # Precompute to be safe
    sim_matrix = sim_matrix.copy()
    np.fill_diagonal(sim_matrix, 0.0)

    while len(selected) < summary_size and candidate_idxs:
        mmr_scores = {}
        for i in candidate_idxs:
            if not selected:
                diversity_penalty = 0.0
            else:
                sim_to_selected = max(sim_matrix[i][j] for j in selected)
                diversity_penalty = sim_to_selected

            mmr_score = lambda_param * scores[i] - (1 - lambda_param) * diversity_penalty
            mmr_scores[i] = mmr_score

        # Choose candidate with max MMR
        best_idx = max(mmr_scores, key=mmr_scores.get)
        selected.append(best_idx)
        candidate_idxs.remove(best_idx)

    return selected


def summarize_document(text: str, length_choice: str = "medium"):
    """
    High-level pipeline:
      1. Normalize and split into sentences
      2. TF-IDF embeddings
      3. Cosine similarity graph + TextRank
      4. MMR for redundancy reduction
      5. Reorder selected sentences to original order
    """
    cleaned = normalize_whitespace(text)
    sentences = split_into_sentences(cleaned)

    if not sentences:
        raise ValueError("No valid sentences found in the document.")

    n_sent = len(sentences)

    # If very short document, just return as-is
    if n_sent <= 3:
        summary_text = " ".join(sentences)
        stats = {
            "original_sentences": n_sent,
            "summary_sentences": n_sent,
            "original_chars": len(cleaned),
            "summary_chars": len(summary_text),
            "compression_ratio": 100,
        }
        return summary_text, stats

    # Determine summary length based on user choice
    length_choice = (length_choice or "medium").lower()
    if length_choice == "short":
        ratio = 0.15
        max_sents = 5
    elif length_choice == "long":
        ratio = 0.35
        max_sents = 15
    else:
        ratio = 0.25
        max_sents = 10

    target_count = max(1, int(round(n_sent * ratio)))
    target_count = min(target_count, max_sents, n_sent)

    # TF-IDF embeddings
    tfidf_matrix = build_tfidf_matrix(sentences)

    # Cosine similarity matrix
    sim_matrix = cosine_similarity(tfidf_matrix)

    # TextRank sentence importance
    textrank_scores = compute_textrank_scores(sim_matrix)

    # MMR selection for redundancy reduction
    selected_indices = mmr_selection(
        scores_dict=textrank_scores,
        sim_matrix=sim_matrix,
        summary_size=target_count,
        lambda_param=0.7,
    )

    # Reorder selected sentences according to original order
    selected_indices_sorted = sorted(selected_indices)
    summary_sentences = [sentences[i].strip() for i in selected_indices_sorted]
    summary_text = " ".join(summary_sentences)

    compression_ratio = int(round(100.0 * len(summary_sentences) / max(n_sent, 1)))

    stats = {
        "original_sentences": n_sent,
        "summary_sentences": len(summary_sentences),
        "original_chars": len(cleaned),
        "summary_chars": len(summary_text),
        "compression_ratio": compression_ratio,
    }

    return summary_text, stats


# -------------------- Flask Routes -------------------- #


@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML)


@app.route("/summarize", methods=["POST"])
def summarize():
    if "file" not in request.files:
        abort(400, description="No file part in the request.")

    uploaded_file = request.files["file"]
    if uploaded_file.filename == "":
        abort(400, description="No file selected.")

    filename = uploaded_file.filename.lower()
    if filename.endswith(".pdf"):
        raw_text = extract_text_from_pdf(uploaded_file)
    elif filename.endswith(".txt"):
        try:
            raw_text = uploaded_file.read().decode("utf-8", errors="ignore")
        except Exception as e:
            raise RuntimeError(f"Error reading text file: {e}")
    else:
        abort(400, description="Unsupported file format. Please upload a PDF or .txt file.")

    if not raw_text or len(raw_text.strip()) == 0:
        abort(400, description="Uploaded file appears to be empty or unreadable.")

    length_choice = request.form.get("length", "medium")

    try:
        summary_text, stats = summarize_document(raw_text, length_choice=length_choice)
    except Exception as e:
        abort(500, description=f"Error during summarization: {e}")

    return render_template_string(RESULT_HTML, summary=summary_text, stats=stats)


# -------------------- Entry Point -------------------- #

if __name__ == "__main__":
    # For local testing; on Render, you will typically use: gunicorn app:app
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
