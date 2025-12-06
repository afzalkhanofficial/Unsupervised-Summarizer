import io
import os
import re
import uuid
from collections import defaultdict
from typing import List, Tuple, Dict

import numpy as np
import networkx as nx
from flask import (
    Flask,
    request,
    render_template_string,
    abort,
    send_from_directory,
    jsonify,
    url_for,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
from werkzeug.utils import secure_filename

from PIL import Image
# Removed pytesseract import

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import simpleSplit

import google.generativeai as genai

# ---------------------- CONFIG ---------------------- #

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
SUMMARY_FOLDER = os.path.join(BASE_DIR, "summaries")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SUMMARY_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SUMMARY_FOLDER"] = SUMMARY_FOLDER

# API Key Check
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception:
        GEMINI_API_KEY = None


# ---------------------- HTML TEMPLATES ---------------------- #

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en" class="scroll-smooth">
<head>
  <meta charset="UTF-8">
  <title>Med | Policy Brief Summarizer</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <script>
    tailwind.config = {
      theme: {
        extend: {
          fontFamily: {
            sans: ['Inter', 'sans-serif'],
            mono: ['JetBrains Mono', 'monospace'],
          },
          colors: {
            brand: {
              50: '#f0fdfa', 100: '#ccfbf1', 200: '#99f6e4', 300: '#5eead4',
              400: '#2dd4bf', 500: '#14b8a6', 600: '#0d9488', 700: '#0f766e',
              800: '#115e59', 900: '#134e4a'
            },
          },
          animation: {
            'progress': 'progress 2s ease-in-out infinite',
          },
          keyframes: {
            progress: {
              '0%': { width: '0%' },
              '50%': { width: '70%' },
              '100%': { width: '100%' },
            }
          }
        }
      }
    }
  </script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&display=swap');
    
    .glass-panel {
      background: rgba(255, 255, 255, 0.95);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(226, 232, 240, 0.8);
      box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.05);
    }
    
    /* Loading Overlay Styles */
    #loading-overlay {
      display: none; /* Hidden by default */
      position: fixed;
      top: 0; left: 0; width: 100%; height: 100%;
      background: rgba(255, 255, 255, 0.95);
      z-index: 9999;
      justify-content: center;
      align-items: center;
      flex-direction: column;
    }
    
    .loader {
      width: 48px;
      height: 48px;
      border: 5px solid #0d9488;
      border-bottom-color: transparent;
      border-radius: 50%;
      display: inline-block;
      box-sizing: border-box;
      animation: rotation 1s linear infinite;
    }

    @keyframes rotation {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }
  </style>
</head>
<body class="bg-slate-50 text-slate-800 relative selection:bg-brand-200 selection:text-brand-900">

  <div id="loading-overlay">
    <div class="loader mb-6"></div>
    <h2 class="text-2xl font-bold text-slate-800 animate-pulse">Analyzing Document...</h2>
    <p class="text-slate-500 mt-2 text-sm font-medium">Running Extraction & TF-IDF Algorithms</p>
    
    <div class="w-64 h-2 bg-slate-200 rounded-full mt-6 overflow-hidden">
      <div class="h-full bg-brand-600 rounded-full animate-[progress_3s_ease-in-out_infinite]"></div>
    </div>
  </div>

  <nav class="fixed w-full z-40 bg-white/80 backdrop-blur-md border-b border-slate-200">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="flex justify-between h-16 items-center">
        <div class="flex items-center gap-3">
          <div class="w-8 h-8 bg-brand-600 rounded-lg flex items-center justify-center text-white shadow-lg shadow-brand-500/30">
            <i class="fa-solid fa-staff-snake"></i>
          </div>
          <span class="font-bold text-xl tracking-tight text-slate-900">Med.AI</span>
        </div>
        <div class="hidden md:flex gap-6 text-sm font-medium text-slate-500">
          <span class="hover:text-brand-600 cursor-default">Extractive Summarization</span>
          <span class="hover:text-brand-600 cursor-default">Policy Analysis</span>
        </div>
      </div>
    </div>
  </nav>

  <main class="pt-28 pb-20 px-4">
    <div class="max-w-6xl mx-auto">
      
      <div class="text-center mb-12 space-y-4">
        <div class="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-brand-50 text-brand-700 border border-brand-200 text-xs font-bold uppercase tracking-wide">
          <span class="w-2 h-2 rounded-full bg-brand-500 animate-pulse"></span>
          AI Powered Policy Briefs
        </div>
        <h1 class="text-4xl md:text-5xl font-extrabold text-slate-900 tracking-tight">
          Transform Healthcare Policy<br>
          <span class="text-transparent bg-clip-text bg-gradient-to-r from-brand-600 to-teal-400">into Actionable Insights</span>
        </h1>
        <p class="max-w-2xl mx-auto text-slate-600 text-lg">
          Upload PDFs, text files, or capture document images. Our intelligent engine extracts abstract summaries, goals, and financing structures instantly.
        </p>
      </div>

      <div class="glass-panel rounded-3xl p-1 shadow-2xl">
        <div class="bg-white rounded-[1.3rem] border border-slate-100 p-6 md:p-10">
          
          <form id="upload-form" action="{{ url_for('summarize') }}" method="post" enctype="multipart/form-data" class="grid lg:grid-cols-2 gap-10">
            
            <div class="space-y-6">
              <div>
                <h3 class="text-lg font-bold text-slate-900 mb-1">Document Input</h3>
                <p class="text-sm text-slate-500">Supported: PDF, TXT, Images</p>
              </div>

              <div class="group relative border-2 border-dashed border-slate-300 hover:border-brand-400 rounded-2xl bg-slate-50 hover:bg-brand-50/30 transition-all duration-300 p-8 text-center cursor-pointer">
                
                <div id="upload-placeholder" class="space-y-4 pointer-events-none">
                  <div class="w-16 h-16 bg-white rounded-full shadow-sm border border-slate-200 flex items-center justify-center mx-auto group-hover:scale-110 transition-transform">
                    <i class="fa-solid fa-cloud-arrow-up text-2xl text-brand-500"></i>
                  </div>
                  <div>
                    <p class="text-sm font-semibold text-slate-900">Click to browse or drop file</p>
                    <p class="text-xs text-slate-500 mt-1">PDF, TXT, JPG, PNG</p>
                  </div>
                </div>

                <div id="file-success" class="hidden space-y-3 pointer-events-none">
                   <div class="w-16 h-16 bg-emerald-50 rounded-full border border-emerald-200 flex items-center justify-center mx-auto">
                    <i class="fa-solid fa-check text-2xl text-emerald-500"></i>
                  </div>
                  <div>
                    <p class="text-sm font-bold text-slate-900" id="filename-display">filename.pdf</p>
                    <p class="text-xs text-emerald-600 font-medium">Ready for analysis</p>
                  </div>
                </div>

                <input id="file-upload" type="file" name="file" accept=".pdf,.txt,image/*" 
                       class="absolute inset-0 w-full h-full opacity-0 cursor-pointer">
              </div>

              <div class="relative">
                <div class="absolute inset-0 flex items-center">
                  <div class="w-full border-t border-slate-200"></div>
                </div>
                <div class="relative flex justify-center text-xs uppercase">
                  <span class="bg-white px-2 text-slate-500">Or use camera</span>
                </div>
              </div>

              <label class="flex items-center justify-center w-full py-3 px-4 rounded-xl border border-slate-200 shadow-sm bg-white text-slate-700 font-semibold cursor-pointer hover:bg-slate-50 hover:border-brand-300 transition gap-2">
                <i class="fa-solid fa-camera text-brand-600"></i>
                <span>Capture Document</span>
                <input id="camera-upload" type="file" name="file_camera" accept="image/*" capture="environment" class="hidden">
              </label>

            </div>

            <div class="flex flex-col justify-between space-y-8 border-l border-slate-100 lg:pl-10">
              
              <div class="space-y-6">
                <div>
                  <h3 class="text-lg font-bold text-slate-900 mb-1">Configuration</h3>
                  <p class="text-sm text-slate-500">Tailor the summary output</p>
                </div>

                <div class="space-y-4">
                  <div>
                    <span class="text-xs font-bold text-slate-400 uppercase tracking-wider">Summary Length</span>
                    <div class="grid grid-cols-3 gap-2 mt-2">
                      <label class="cursor-pointer">
                        <input type="radio" name="length" value="short" class="peer hidden">
                        <div class="py-2 text-center text-xs font-semibold rounded-lg border border-slate-200 text-slate-600 peer-checked:bg-slate-800 peer-checked:text-white peer-checked:border-slate-800 transition">
                          Short
                        </div>
                      </label>
                      <label class="cursor-pointer">
                        <input type="radio" name="length" value="medium" checked class="peer hidden">
                        <div class="py-2 text-center text-xs font-semibold rounded-lg border border-slate-200 text-slate-600 peer-checked:bg-slate-800 peer-checked:text-white peer-checked:border-slate-800 transition">
                          Medium
                        </div>
                      </label>
                      <label class="cursor-pointer">
                        <input type="radio" name="length" value="long" class="peer hidden">
                        <div class="py-2 text-center text-xs font-semibold rounded-lg border border-slate-200 text-slate-600 peer-checked:bg-slate-800 peer-checked:text-white peer-checked:border-slate-800 transition">
                          Detailed
                        </div>
                      </label>
                    </div>
                  </div>

                  <div>
                    <span class="text-xs font-bold text-slate-400 uppercase tracking-wider">Tone Style</span>
                    <div class="grid grid-cols-2 gap-2 mt-2">
                      <label class="cursor-pointer">
                        <input type="radio" name="tone" value="academic" checked class="peer hidden">
                        <div class="flex items-center justify-center gap-2 py-2 text-xs font-semibold rounded-lg border border-slate-200 text-slate-600 peer-checked:bg-brand-50 peer-checked:text-brand-700 peer-checked:border-brand-500 transition">
                          <i class="fa-solid fa-graduation-cap"></i> Academic
                        </div>
                      </label>
                      <label class="cursor-pointer">
                        <input type="radio" name="tone" value="easy" class="peer hidden">
                        <div class="flex items-center justify-center gap-2 py-2 text-xs font-semibold rounded-lg border border-slate-200 text-slate-600 peer-checked:bg-brand-50 peer-checked:text-brand-700 peer-checked:border-brand-500 transition">
                          <i class="fa-solid fa-child-reaching"></i> Simple English
                        </div>
                      </label>
                    </div>
                  </div>
                </div>
              </div>

              <button type="submit" 
                class="group w-full py-4 rounded-xl bg-gradient-to-r from-brand-600 to-teal-500 text-white font-bold shadow-lg shadow-brand-500/30 hover:shadow-brand-500/50 hover:scale-[1.02] transition-all duration-200 flex items-center justify-center gap-2">
                <span>Generate Summary</span>
                <i class="fa-solid fa-wand-magic-sparkles group-hover:rotate-12 transition-transform"></i>
              </button>

            </div>
          </form>

        </div>
      </div>
      
      <p class="text-center text-xs text-slate-400 mt-8">
        Privacy Note: Documents are processed securely and temporarily. No data is used for model training.
      </p>

    </div>
  </main>

  <script>
    // Elements
    const form = document.getElementById('upload-form');
    const fileInput = document.getElementById('file-upload');
    const cameraInput = document.getElementById('camera-upload');
    const placeholderDiv = document.getElementById('upload-placeholder');
    const successDiv = document.getElementById('file-success');
    const filenameDisplay = document.getElementById('filename-display');
    const loadingOverlay = document.getElementById('loading-overlay');

    // Helper to update UI
    function updateFileDisplay(name) {
      if (!name) return;
      placeholderDiv.classList.add('hidden');
      successDiv.classList.remove('hidden');
      filenameDisplay.textContent = name;
    }

    // Listeners for file selection
    fileInput.addEventListener('change', (e) => {
      if (e.target.files.length > 0) {
        updateFileDisplay(e.target.files[0].name);
        cameraInput.value = '';
      }
    });

    cameraInput.addEventListener('change', (e) => {
      if (e.target.files.length > 0) {
        updateFileDisplay("Captured Image");
        fileInput.value = '';
      }
    });

    // Form Submit Handler -> Show Loading
    form.addEventListener('submit', (e) => {
      if (!fileInput.value && !cameraInput.value) {
        e.preventDefault();
        alert("Please select a file or take a photo first.");
        return;
      }
      loadingOverlay.style.display = 'flex';
    });
  </script>
</body>
</html>
"""

RESULT_HTML = """
<!DOCTYPE html>
<html lang="en" class="scroll-smooth">
<head>
  <meta charset="UTF-8">
  <title>{{ title }}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <script>
    tailwind.config = {
      theme: {
        extend: {
          fontFamily: {
            sans: ['Inter', 'sans-serif'],
            mono: ['JetBrains Mono', 'monospace'],
          },
          colors: {
            brand: {
              50: '#f0fdfa', 100: '#ccfbf1', 200: '#99f6e4', 300: '#5eead4',
              400: '#2dd4bf', 500: '#14b8a6', 600: '#0d9488', 700: '#0f766e',
              800: '#115e59', 900: '#134e4a'
            },
          }
        }
      }
    }
  </script>
</head>
<body class="bg-slate-50 text-slate-800">

  <nav class="fixed w-full z-40 bg-white/80 backdrop-blur-lg border-b border-slate-200">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="flex justify-between h-16 items-center">
        <div class="flex items-center gap-3">
          <div class="w-8 h-8 bg-brand-600 rounded-lg flex items-center justify-center text-white">
            <i class="fa-solid fa-staff-snake"></i>
          </div>
          <span class="font-bold text-xl tracking-tight text-slate-900">Med.AI</span>
        </div>
        <a href="{{ url_for('index') }}"
           class="inline-flex items-center px-4 py-2 text-xs font-bold rounded-lg border border-slate-200 hover:border-brand-300 hover:text-brand-700 bg-white transition shadow-sm">
          <i class="fa-solid fa-arrow-left mr-2"></i>
          Summarize Another
        </a>
      </div>
    </div>
  </nav>

  <main class="pt-24 pb-12 px-4">
    <div class="max-w-7xl mx-auto grid lg:grid-cols-12 gap-8">
      
      <section class="lg:col-span-7 space-y-6">
        
        <div class="bg-white rounded-3xl shadow-xl border border-slate-100 overflow-hidden">
          <div class="p-6 md:p-8 border-b border-slate-100 bg-slate-50/50">
            <h1 class="text-2xl font-extrabold text-slate-900 leading-tight">{{ title }}</h1>
            <div class="flex gap-4 mt-3 text-xs font-semibold text-slate-500">
               <span class="flex items-center gap-1"><i class="fa-solid fa-compress text-brand-500"></i> Reduced by {{ 100 - stats.compression_ratio }}%</span>
               <span class="flex items-center gap-1"><i class="fa-solid fa-clock text-brand-500"></i> {{ stats.summary_sentences }} sentences</span>
            </div>
          </div>

          <div class="p-6 md:p-8 space-y-8">
            <div>
              <h2 class="text-sm font-bold text-slate-400 uppercase tracking-wider mb-3">Abstract</h2>
              <div class="bg-brand-50/50 p-4 rounded-xl border border-brand-100 text-slate-800 text-sm leading-relaxed">
                {{ abstract }}
              </div>
            </div>

            {% if sections %}
            <div>
              <h2 class="text-sm font-bold text-slate-400 uppercase tracking-wider mb-3">Key Findings</h2>
              <div class="space-y-4">
                {% for sec in sections %}
                <div class="group">
                  <h3 class="flex items-center gap-2 font-bold text-slate-900 text-sm mb-2">
                    <span class="w-2 h-2 rounded-full bg-brand-500"></span>
                    {{ sec.title }}
                  </h3>
                  <ul class="ml-2 pl-4 border-l-2 border-slate-100 space-y-2">
                    {% for bullet in sec.bullets %}
                      <li class="text-xs text-slate-600 leading-relaxed">{{ bullet }}</li>
                    {% endfor %}
                  </ul>
                </div>
                {% endfor %}
              </div>
            </div>
            {% endif %}
          </div>

          {% if summary_pdf_url %}
          <div class="bg-slate-50 p-4 border-t border-slate-100 flex justify-between items-center">
            <span class="text-xs text-slate-500 font-medium">Get this summary to go</span>
            <a href="{{ summary_pdf_url }}" class="inline-flex items-center px-4 py-2 rounded-lg bg-slate-900 text-white text-xs font-bold hover:bg-slate-800 transition shadow-lg shadow-slate-900/20">
              <i class="fa-solid fa-download mr-2"></i> Download PDF
            </a>
          </div>
          {% endif %}
        </div>

        <div class="bg-white rounded-3xl shadow-xl border border-slate-100 p-6 md:p-8">
          <h2 class="text-sm font-bold text-slate-900 mb-4 flex items-center gap-2">
            <i class="fa-solid fa-chart-pie text-brand-500"></i> Topic Coverage
          </h2>
          {% if category_labels %}
            <canvas id="catChart" height="120"></canvas>
          {% else %}
            <p class="text-xs text-slate-400">Not enough data for analytics.</p>
          {% endif %}
        </div>
      </section>

      <section class="lg:col-span-5 space-y-6">
        
        <div class="bg-white rounded-3xl shadow-xl border border-slate-100 overflow-hidden flex flex-col h-[500px]">
          <div class="p-4 border-b border-slate-100 bg-slate-50 flex justify-between items-center">
            <h2 class="text-sm font-bold text-slate-900">Original Document</h2>
            <span class="text-[10px] uppercase font-bold text-slate-400">{{ orig_type }}</span>
          </div>
          <div class="flex-1 bg-slate-100 overflow-hidden relative">
            {% if orig_type == 'pdf' %}
              <iframe src="{{ orig_url }}" class="w-full h-full border-none"></iframe>
            {% elif orig_type == 'text' %}
              <div class="p-4 h-full overflow-y-auto text-xs font-mono text-slate-600 whitespace-pre-wrap">{{ orig_text }}</div>
            {% elif orig_type == 'image' %}
              <div class="flex items-center justify-center h-full p-4">
                <img src="{{ orig_url }}" class="max-w-full max-h-full object-contain rounded-lg shadow-sm">
              </div>
            {% endif %}
          </div>
        </div>

        <div class="bg-white rounded-3xl shadow-xl border border-slate-100 p-6 flex flex-col h-[400px]">
          <div class="flex items-center justify-between mb-4">
            <h2 class="text-sm font-bold text-slate-900 flex items-center gap-2">
              <i class="fa-solid fa-sparkles text-violet-500"></i> AI Assistant
            </h2>
            <span class="text-[10px] font-bold px-2 py-1 bg-violet-50 text-violet-600 rounded-md">Gemini Flash</span>
          </div>
          
          <div id="chat-panel" class="flex-1 overflow-y-auto space-y-3 pr-2 mb-4 scrollbar-hide">
            <div class="flex gap-3">
              <div class="w-8 h-8 rounded-full bg-violet-100 flex-shrink-0 flex items-center justify-center text-violet-600 text-xs">
                <i class="fa-solid fa-robot"></i>
              </div>
              <div class="bg-slate-50 border border-slate-100 rounded-2xl rounded-tl-none p-3 text-xs text-slate-600 shadow-sm">
                Hello! I've read the document. Ask me about specific policies, numbers, or goals.
              </div>
            </div>
          </div>

          <div class="relative">
            <input id="chat-input" type="text" placeholder="Ask a question..." 
                   class="w-full bg-slate-50 border border-slate-200 rounded-xl py-3 pl-4 pr-12 text-xs focus:outline-none focus:border-brand-500 focus:ring-1 focus:ring-brand-500 transition">
            <button id="chat-send" class="absolute right-2 top-2 p-1.5 bg-brand-600 text-white rounded-lg hover:bg-brand-700 transition">
              <i class="fa-solid fa-paper-plane text-xs"></i>
            </button>
          </div>
          
          <textarea id="doc-context" class="hidden">{{ doc_context }}</textarea>
        </div>

      </section>
    </div>
  </main>

  {% if category_labels %}
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script>
    const ctx = document.getElementById('catChart').getContext('2d');
    new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: {{ category_labels|tojson }},
        datasets: [{
          data: {{ category_values|tojson }},
          backgroundColor: [
            '#0d9488', '#14b8a6', '#5eead4', '#99f6e4', '#ccfbf1', '#f0fdfa'
          ],
          borderWidth: 0
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { 
            legend: { position: 'right', labels: { boxWidth: 10, font: { size: 10 } } } 
        },
        cutout: '70%'
      }
    });
  </script>
  {% endif %}

  <script>
    const panel = document.getElementById('chat-panel');
    const input = document.getElementById('chat-input');
    const sendBtn = document.getElementById('chat-send');
    const ctxArea = document.getElementById('doc-context');
    const docText = ctxArea ? ctxArea.value : "";

    function addMsg(role, text) {
      const div = document.createElement('div');
      div.className = role === 'user' ? 'flex justify-end mb-3' : 'flex gap-3 mb-3';
      
      let html = '';
      if(role === 'assistant') {
        html += `<div class="w-8 h-8 rounded-full bg-violet-100 flex-shrink-0 flex items-center justify-center text-violet-600 text-xs"><i class="fa-solid fa-robot"></i></div>`;
        html += `<div class="bg-slate-50 border border-slate-100 rounded-2xl rounded-tl-none p-3 text-xs text-slate-600 shadow-sm max-w-[85%]">${text}</div>`;
      } else {
        html += `<div class="bg-brand-600 text-white rounded-2xl rounded-tr-none p-3 text-xs shadow-md max-w-[85%]">${text}</div>`;
      }
      div.innerHTML = html;
      panel.appendChild(div);
      panel.scrollTop = panel.scrollHeight;
    }

    async function send() {
      const txt = input.value.trim();
      if(!txt) return;
      addMsg('user', txt);
      input.value = '';
      
      try {
        const res = await fetch('{{ url_for("chat") }}', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ message: txt, doc_text: docText })
        });
        const data = await res.json();
        addMsg('assistant', data.reply);
      } catch(e) {
        addMsg('assistant', "Error connecting to AI.");
      }
    }

    sendBtn.addEventListener('click', send);
    input.addEventListener('keydown', (e) => { if(e.key === 'Enter') send(); });
  </script>
</body>
</html>
"""

# ---------------------- TEXT UTILITIES ---------------------- #

def normalize_whitespace(text: str) -> str:
    text = text.replace("\r", " ").replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_leading_numbering(s: str) -> str:
    return re.sub(r"^\s*\d+(\.\d+)*\s*[:\-\)]?\s*", "", s).strip()


def is_toc_like(s: str) -> bool:
    s_lower = s.lower()
    digits = sum(c.isdigit() for c in s)
    if digits >= 10 and len(s) > 80 and not re.search(
        r"\b(reduce|increase|improve|achieve|eliminate|raise|reach|decrease|enhance)\b",
        s_lower,
    ):
        return True
    if re.search(r"\bcontents\b", s_lower):
        return True
    return False


def sentence_split(text: str) -> List[str]:
    text = re.sub(r"\n+", " ", text)
    bulleted = re.split(r"\s+[•o]\s+", text)
    sentences = []
    for chunk in bulleted:
        parts = re.split(r"(?<=[\.\?\!])\s+(?=[A-Z0-9“'\"-])", chunk)
        for p in parts:
            p = p.strip()
            if not p:
                continue
            p = re.sub(r"^[\-\–\•\*]+\s*", "", p)
            p = strip_leading_numbering(p)
            if len(p) < 20:
                continue
            if is_toc_like(p):
                continue
            sentences.append(p)
    return sentences


def extract_text_from_pdf_bytes(raw: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(raw))
        pages = []
        for pg in reader.pages:
            pages.append(pg.extract_text() or "")
        return "\n".join(pages)
    except Exception:
        return ""


def extract_sections(raw_text: str) -> List[Tuple[str, str]]:
    lines = raw_text.splitlines()
    sections: List[Tuple[str, str]] = []
    current_title = "Front"
    buffer: List[str] = []

    heading_re = re.compile(r"^\s*\d+(\.\d+)*\s+[A-Za-z].{0,120}$")
    short_upper_re = re.compile(r"^[A-Z][A-Z\s\-]{4,}$")

    for ln in lines:
        s = ln.strip()
        if not s:
            buffer.append("")
            continue
        if heading_re.match(s) or (short_upper_re.match(s) and len(s.split()) < 12):
            if buffer:
                sections.append((current_title, " ".join(buffer).strip()))
            title = strip_leading_numbering(s)
            current_title = title[:120]
            buffer = []
        else:
            buffer.append(s)
    if buffer:
        sections.append((current_title, " ".join(buffer).strip()))

    cleaned = [(t, normalize_whitespace(b)) for t, b in sections if b.strip()]
    return cleaned


# ---------------------- GOAL & CATEGORY HELPERS ---------------------- #

GOAL_METRIC_WORDS = [
    "life expectancy", "mortality", "imr", "u5mr", "mmr", "coverage",
    "immunization", "incidence", "prevalence", "%", " per ", "gdp",
    "reduction", "rate",
]

GOAL_VERBS = [
    "reduce", "increase", "improve", "achieve", "eliminate", "raise", 
    "reach", "decrease", "enhance"
]


def is_goal_sentence(s: str) -> bool:
    s_lower = s.lower()
    has_digit = any(ch.isdigit() for ch in s_lower)
    if not has_digit:
        return False
    if not any(w in s_lower for w in GOAL_METRIC_WORDS):
        return False
    if not any(v in s_lower for v in GOAL_VERBS):
        return False
    return True


def categorize_sentence(s: str) -> str:
    s_lower = s.lower()

    if is_goal_sentence(s):
        return "key goals"

    if any(w in s_lower for w in ["principle", "values", "equity", "universal access", "right to health"]):
        return "policy principles"

    if any(w in s_lower for w in ["primary care", "hospital", "service delivery", "referral", "drugs", "diagnostics"]):
        return "service delivery"

    if any(w in s_lower for w in ["prevention", "promotive", "nutrition", "tobacco", "sanitation", "lifestyle"]):
        return "prevention & promotion"

    if any(w in s_lower for w in ["human resources", "hrh", "workforce", "doctors", "nurses", "training"]):
        return "human resources"

    if any(w in s_lower for w in ["financing", "insurance", "spending", "gdp", "expenditure", "private sector", "ppp"]):
        return "financing & private sector"

    if any(w in s_lower for w in ["digital health", "ehr", "telemedicine", "data", "surveillance"]):
        return "digital health"

    if any(w in s_lower for w in ["ayush", "ayurveda", "yoga", "homeopathy"]):
        return "ayush integration"

    if any(w in s_lower for w in ["implementation", "way forward", "roadmap", "action plan", "monitoring", "governance"]):
        return "implementation"

    return "other"


# ---------------------- TF-IDF + TEXTRANK + MMR ---------------------- #

def build_tfidf(sentences: List[str]):
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_df=0.9, min_df=1)
    try:
        mat = vec.fit_transform(sentences)
    except ValueError:
        # handle empty vocab
        return None
    return mat


def textrank_scores(sim_mat: np.ndarray, positional_boost: np.ndarray = None) -> Dict[int, float]:
    np.fill_diagonal(sim_mat, 0.0)
    G = nx.from_numpy_array(sim_mat)
    try:
        pr = nx.pagerank(G, max_iter=200, tol=1e-6)
    except nx.PowerIterationFailedConvergence:
        pr = {i: 0 for i in range(len(sim_mat))}
        
    scores = np.array([pr.get(i, 0.0) for i in range(sim_mat.shape[0])], dtype=float)
    if positional_boost is not None:
        scores = scores * (1.0 + positional_boost)
    return {i: float(scores[i]) for i in range(len(scores))}


def mmr(scores_dict: Dict[int, float], sim_mat: np.ndarray, k: int, lambda_param: float = 0.7) -> List[int]:
    n = sim_mat.shape[0]
    indices = list(range(n))
    scores = np.array([scores_dict.get(i, 0.0) for i in indices], dtype=float)
    if scores.max() > 0:
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
    selected: List[int] = []
    candidates = set(indices)

    while len(selected) < k and candidates:
        best = None
        best_score = -1e9
        for i in list(candidates):
            if not selected:
                div = 0.0
            else:
                div = max(sim_mat[i][j] for j in selected)
            mmr_score = lambda_param * scores[i] - (1 - lambda_param) * div
            if mmr_score > best_score:
                best_score = mmr_score
                best = i
        if best is None:
            break
        selected.append(best)
        candidates.remove(best)
    return selected


def summarize_extractive(raw_text: str, length_choice: str = "medium"):
    cleaned = normalize_whitespace(raw_text)
    sections = extract_sections(cleaned)

    sentences: List[str] = []
    sent_to_section: List[int] = []
    for si, (title, body) in enumerate(sections):
        sents = sentence_split(body)
        for s in sents:
            sentences.append(s)
            sent_to_section.append(si)

    if not sentences:
        sentences = sentence_split(cleaned)
        sent_to_section = [0] * len(sentences)
        sections = [("Document", cleaned)]

    n = len(sentences)
    if n <= 3:
        return sentences, {
            "original_sentences": n,
            "summary_sentences": n,
            "compression_ratio": 100,
        }

    if length_choice == "short":
        ratio, max_s = 0.10, 6
    elif length_choice == "long":
        ratio, max_s = 0.30, 20
    else:
        ratio, max_s = 0.20, 12

    target = min(max(1, int(round(n * ratio))), max_s, n)

    tfidf = build_tfidf(sentences)
    if tfidf is None:
        return sentences[:5], {"original_sentences": n, "summary_sentences": 5, "compression_ratio": 0}
        
    sim = cosine_similarity(tfidf)

    pos_boost = np.zeros(n, dtype=float)
    sec_first_idx: Dict[int, int] = {}
    for idx, sec_idx in enumerate(sent_to_section):
        sec_first_idx.setdefault(sec_idx, None)
        if sec_first_idx[sec_idx] is None:
            sec_first_idx[sec_idx] = idx
    
    tr_scores = textrank_scores(sim, positional_boost=pos_boost)

    ranked_global = sorted(range(n), key=lambda i: -tr_scores.get(i, 0.0))
    selected_idxs = ranked_global[:target] 
    
    try:
        selected_idxs = mmr(tr_scores, sim, target)
    except Exception:
        pass

    selected_idxs = sorted(selected_idxs)
    summary_sentences = [sentences[i].strip() for i in selected_idxs]
    
    stats = {
        "original_sentences": n,
        "summary_sentences": len(summary_sentences),
        "compression_ratio": int(round(100.0 * len(summary_sentences) / max(1, n))),
    }
    return summary_sentences, stats


def build_structured_summary(summary_sentences: List[str], tone: str):
    processed = summary_sentences
    
    abstract_sents = processed[:3] if len(processed) >= 3 else processed
    abstract = " ".join(abstract_sents)

    category_to_sentences: Dict[str, List[str]] = defaultdict(list)
    for s in processed:
        cat = categorize_sentence(s)
        category_to_sentences[cat].append(s)

    section_order = [
        ("key goals", "Key Goals"),
        ("policy principles", "Policy Principles"),
        ("service delivery", "Strengthening Healthcare Delivery"),
        ("prevention & promotion", "Prevention & Health Promotion"),
        ("human resources", "Human Resources for Health"),
        ("financing & private sector", "Financing & Private Sector Engagement"),
        ("digital health", "Digital Health & Information Systems"),
        ("ayush integration", "AYUSH Integration"),
        ("implementation", "Implementation & Way Forward"),
        ("other", "Other Important Points"),
    ]

    sections = []
    for key, title in section_order:
        bullets = category_to_sentences.get(key, [])
        if bullets:
            seen = set()
            unique = []
            for b in bullets:
                if b not in seen:
                    seen.add(b)
                    unique.append(b)
            sections.append({"title": title, "bullets": unique})

    category_counts = {k: len(v) for k, v in category_to_sentences.items()}
    implementation_points = category_to_sentences.get("implementation", [])

    return {
        "abstract": abstract,
        "sections": sections,
        "category_counts": category_counts,
        "implementation_points": implementation_points,
    }


def save_summary_pdf(title: str, abstract: str, sections: List[Dict], out_path: str):
    c = canvas.Canvas(out_path, pagesize=A4)
    width, height = A4
    margin_x, margin_y = 40, 40
    max_width = width - 2 * margin_x

    def draw_paragraph(text, y, font="Helvetica", size=10, leading=13):
        c.setFont(font, size)
        lines = simpleSplit(text, font, size, max_width)
        for line in lines:
            if y < margin_y:
                c.showPage()
                y = height - margin_y
                c.setFont(font, size)
            c.drawString(margin_x, y, line)
            y -= leading
        return y

    y = height - margin_y
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin_x, y, title)
    y -= 30

    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin_x, y, "Abstract")
    y -= 15
    y = draw_paragraph(abstract, y)
    y -= 15

    for sec in sections:
        if y < margin_y + 40:
            c.showPage()
            y = height - margin_y
        c.setFont("Helvetica-Bold", 11)
        c.drawString(margin_x, y, sec["title"])
        y -= 14
        for bullet in sec["bullets"]:
            y = draw_paragraph("• " + bullet, y)
            y -= 2
        y -= 8

    c.save()


# ---------------------- GEMINI HELPERS ---------------------- #

def gemini_extract_text(image_bytes: bytes) -> str:
    """Uses Gemini Vision to extract text (OCR replacement)."""
    if not GEMINI_API_KEY:
        raise Exception("Gemini API Key missing")
    
    try:
        model = genai.GenerativeModel("gemini-2.5-flash") # or gemini-1.5-flash
        
        # Convert bytes to PIL Image
        img = Image.open(io.BytesIO(image_bytes))
        
        prompt = "Transcribe all the text visible in this image exactly as it appears. Do not summarize. Just extract the text."
        response = model.generate_content([prompt, img])
        return response.text
    except Exception as e:
        print(f"Gemini OCR Error: {e}")
        return ""

def gemini_answer(user_message: str, doc_text: str) -> str:
    if not GEMINI_API_KEY:
        return "Gemini API key is not configured."
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = (
            "You are an AI assistant helping a student understand a healthcare policy document.\n"
            "Answer concisely and only using information from the document.\n\n"
            "DOCUMENT:\n"
            f"{doc_text[:60000]}\n\n"
            f"USER QUESTION: {user_message}\n\n"
            "ANSWER:"
        )
        resp = model.generate_content(prompt)
        return (resp.text or "").strip()
    except Exception as e:
        return f"Error contacting AI: {str(e)}"


# ---------------------- FLASK ROUTES ---------------------- #

@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML)


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/summaries/<path:filename>")
def summary_file(filename):
    return send_from_directory(app.config["SUMMARY_FOLDER"], filename, as_attachment=True)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True, silent=True) or {}
    message = (data.get("message") or "").strip()
    doc_text = data.get("doc_text") or ""
    if not message:
        return jsonify({"reply": "Please type a question."})
    reply = gemini_answer(message, doc_text)
    return jsonify({"reply": reply})


@app.route("/summarize", methods=["POST"])
def summarize():
    # Improved File Handling
    file_obj = request.files.get("file")
    camera_obj = request.files.get("file_camera")
    
    f = None
    if file_obj and file_obj.filename != "":
        f = file_obj
    elif camera_obj and camera_obj.filename != "":
        f = camera_obj
    
    if not f:
        return abort(400, "No file uploaded. Please select a file or take a photo.")

    filename = secure_filename(f.filename) or "upload.jpg"
    raw_bytes = f.read()
    
    # Save original
    uid = uuid.uuid4().hex
    stored_name = f"{uid}_{filename}"
    stored_path = os.path.join(app.config["UPLOAD_FOLDER"], stored_name)
    with open(stored_path, "wb") as out:
        out.write(raw_bytes)

    # Detect Type
    lower_name = filename.lower()
    orig_type = "unknown"
    raw_text = ""
    
    try:
        if lower_name.endswith(".pdf"):
            orig_type = "pdf"
            raw_text = extract_text_from_pdf_bytes(raw_bytes)
        elif lower_name.endswith(".txt"):
            orig_type = "text"
            raw_text = raw_bytes.decode("utf-8", errors="ignore")
        else:
            # Assume Image -> Send to Gemini for OCR
            orig_type = "image"
            raw_text = gemini_extract_text(raw_bytes)
            
            if not raw_text:
                return abort(500, "AI could not read text from the image. Please try a clearer photo.")
                
    except Exception as e:
        return abort(500, f"Error processing file: {e}")

    if not raw_text or len(raw_text.strip()) < 50:
        return abort(400, "Could not extract enough text. The document might be empty or blurry.")

    # Continue with ML Model (TF-IDF/TextRank)
    length_choice = request.form.get("length", "medium")
    tone = request.form.get("tone", "academic")

    summary_sentences, stats = summarize_extractive(raw_text, length_choice)
    structured = build_structured_summary(summary_sentences, tone)

    # Generate PDF
    summary_pdf_url = None
    try:
        s_filename = f"{uid}_summary.pdf"
        s_path = os.path.join(app.config["SUMMARY_FOLDER"], s_filename)
        save_summary_pdf("Policy Summary", structured["abstract"], structured["sections"], s_path)
        summary_pdf_url = url_for("summary_file", filename=s_filename)
    except Exception:
        pass

    cat_counts = structured["category_counts"]
    labels = list(cat_counts.keys()) if cat_counts else []
    values = list(cat_counts.values()) if cat_counts else []

    return render_template_string(
        RESULT_HTML,
        title="Policy Brief Summary",
        abstract=structured["abstract"],
        sections=structured["sections"],
        stats=stats,
        category_labels=labels,
        category_values=values,
        orig_type=orig_type,
        orig_url=url_for("uploaded_file", filename=stored_name),
        orig_text=raw_text[:20000],
        doc_context=raw_text[:10000],
        summary_pdf_url=summary_pdf_url
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
