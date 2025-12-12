import io
import os
import re
import uuid
import json
import time
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Any

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
from deep_translator import GoogleTranslator

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

# Configure Gemini
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
  <title>Med.AI | Advanced Policy Summarizer</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700;800&display=swap" rel="stylesheet">
  <script>
    tailwind.config = {
      theme: {
        extend: {
          fontFamily: {
            sans: ['Plus Jakarta Sans', 'sans-serif'],
          },
          colors: {
            teal: { 50: '#f0fdfa', 100: '#ccfbf1', 200: '#99f6e4', 300: '#5eead4', 400: '#2dd4bf', 500: '#14b8a6', 600: '#0d9488', 700: '#0f766e', 800: '#115e59', 900: '#134e4a' },
            slate: { 850: '#1e293b' } 
          },
          animation: {
            'float': 'float 6s ease-in-out infinite',
            'fade-in': 'fadeIn 0.5s ease-out forwards',
          },
          keyframes: {
            float: {
              '0%, 100%': { transform: 'translateY(0)' },
              '50%': { transform: 'translateY(-10px)' },
            },
            fadeIn: {
              '0%': { opacity: '0', transform: 'translateY(10px)' },
              '100%': { opacity: '1', transform: 'translateY(0)' },
            }
          }
        }
      }
    }
  </script>
  <style>
    body { background: #f0f4f8; }
    .glass-morphism {
      background: rgba(255, 255, 255, 0.85);
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      border: 1px solid rgba(255, 255, 255, 0.6);
      box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.05);
    }
    .gradient-text {
      background: linear-gradient(135deg, #0f766e 0%, #06b6d4 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .blob {
      position: absolute;
      border-radius: 50%;
      filter: blur(80px);
      opacity: 0.6;
      z-index: -1;
    }
  </style>
</head>
<body class="text-slate-800 relative min-h-screen flex flex-col overflow-x-hidden">

  <div class="blob bg-teal-200 w-96 h-96 top-0 left-[-100px] animate-pulse"></div>
  <div class="blob bg-blue-200 w-80 h-80 bottom-0 right-[-100px] animate-pulse delay-700"></div>

  <nav class="fixed w-full z-50 glass-morphism border-b border-white/50 transition-all duration-300">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="flex justify-between h-20 items-center">
        <div class="flex items-center gap-3">
          <div class="w-10 h-10 bg-gradient-to-br from-teal-500 to-cyan-600 rounded-xl flex items-center justify-center text-white shadow-lg shadow-teal-500/20 transform hover:rotate-12 transition duration-300">
            <i class="fa-solid fa-notes-medical text-lg"></i>
          </div>
          <span class="font-extrabold text-2xl tracking-tight text-slate-800">
            Med<span class="text-teal-600">.AI</span>
          </span>
        </div>
        <div class="hidden md:flex items-center gap-6">
            <span class="px-3 py-1 rounded-full bg-teal-50 text-teal-700 text-xs font-bold uppercase tracking-wider border border-teal-100">
                v2.0 Enhanced
            </span>
        </div>
      </div>
    </div>
  </nav>

  <main class="flex-grow pt-32 pb-12 px-4">
    <div class="max-w-4xl mx-auto space-y-12">
      
      <div class="text-center space-y-6 animate-fade-in">
        <h1 class="text-5xl md:text-7xl font-extrabold text-slate-900 leading-tight">
          Summarize Medical <br>
          <span class="gradient-text">Policies Instantly</span>
        </h1>
        <p class="text-lg text-slate-500 max-w-2xl mx-auto leading-relaxed">
          Upload multi-page PDFs or use your camera to capture multiple images. 
          Our enhanced ML algorithms generate comprehensive summaries in your preferred language.
        </p>
      </div>

      <div class="glass-morphism rounded-[2.5rem] p-2 shadow-2xl animate-fade-in" style="animation-delay: 0.2s;">
        <div class="bg-white/60 rounded-[2rem] p-8 md:p-12 border border-white">
          
          <form id="uploadForm" action="{{ url_for('summarize') }}" method="post" enctype="multipart/form-data" class="space-y-10">
            
            <div class="group relative w-full h-72 border-4 border-dashed border-slate-200 rounded-3xl bg-slate-50/50 hover:bg-teal-50/30 hover:border-teal-400 transition-all duration-300 flex flex-col items-center justify-center cursor-pointer overflow-hidden" id="drop-zone">
              
              <input id="file-input" type="file" name="file" accept=".pdf,.txt,image/*" multiple class="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-20">
              
              <div id="upload-prompt" class="text-center space-y-5 transition-all duration-300 group-hover:scale-105 pointer-events-none">
                <div class="w-20 h-20 bg-white rounded-2xl shadow-xl flex items-center justify-center mx-auto text-teal-500 text-3xl group-hover:text-teal-600 group-hover:-translate-y-2 transition-transform duration-300">
                  <i class="fa-solid fa-cloud-arrow-up"></i>
                </div>
                <div>
                  <p class="text-xl font-bold text-slate-700">Drop files or Click to Upload</p>
                  <p class="text-sm text-slate-400 mt-2 font-medium">Supports PDF, TXT & Multiple Images</p>
                </div>
                <div class="inline-flex items-center gap-2 px-5 py-2.5 bg-slate-800 text-white rounded-full text-xs font-bold uppercase tracking-wide shadow-lg shadow-slate-800/20">
                   <i class="fa-solid fa-camera"></i> Camera Ready
                </div>
              </div>

              <div id="file-preview" class="hidden absolute inset-0 bg-white/95 backdrop-blur-sm z-10 flex flex-col items-center justify-center p-6 text-center">
                 <div class="w-16 h-16 bg-teal-100 text-teal-600 rounded-full flex items-center justify-center text-2xl mb-4">
                    <i class="fa-solid fa-check"></i>
                 </div>
                 <h3 class="text-lg font-bold text-slate-800 mb-1">Files Selected</h3>
                 <p id="filename-display" class="text-slate-500 text-sm max-w-md truncate mb-6"></p>
                 <button type="button" id="reset-files" class="px-6 py-2 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-600 text-sm font-bold transition z-30 relative">
                    Reset Selection
                 </button>
              </div>

            </div>

            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
              
              <div class="bg-white rounded-2xl p-5 border border-slate-100 shadow-sm hover:shadow-md transition">
                <label class="flex items-center gap-2 text-xs font-extrabold text-slate-400 uppercase tracking-widest mb-4">
                    <i class="fa-solid fa-ruler-horizontal text-teal-500"></i> Summary Length
                </label>
                <div class="grid grid-cols-3 gap-2 bg-slate-50 p-1.5 rounded-xl">
                  <label class="cursor-pointer">
                    <input type="radio" name="length" value="short" class="peer hidden">
                    <div class="py-3 rounded-lg text-center text-xs font-bold text-slate-500 peer-checked:bg-white peer-checked:text-teal-700 peer-checked:shadow-sm transition hover:bg-white/50">
                        Short<br><span class="text-[9px] opacity-70">~1k chars</span>
                    </div>
                  </label>
                  <label class="cursor-pointer">
                    <input type="radio" name="length" value="medium" checked class="peer hidden">
                    <div class="py-3 rounded-lg text-center text-xs font-bold text-slate-500 peer-checked:bg-white peer-checked:text-teal-700 peer-checked:shadow-sm transition hover:bg-white/50">
                        Medium<br><span class="text-[9px] opacity-70">~5k chars</span>
                    </div>
                  </label>
                  <label class="cursor-pointer">
                    <input type="radio" name="length" value="long" class="peer hidden">
                    <div class="py-3 rounded-lg text-center text-xs font-bold text-slate-500 peer-checked:bg-white peer-checked:text-teal-700 peer-checked:shadow-sm transition hover:bg-white/50">
                        Long<br><span class="text-[9px] opacity-70">~10k chars</span>
                    </div>
                  </label>
                </div>
              </div>

              <div class="bg-white rounded-2xl p-5 border border-slate-100 shadow-sm hover:shadow-md transition">
                <label class="flex items-center gap-2 text-xs font-extrabold text-slate-400 uppercase tracking-widest mb-4">
                    <i class="fa-solid fa-sliders text-teal-500"></i> Style / Tone
                </label>
                <div class="grid grid-cols-2 gap-2 bg-slate-50 p-1.5 rounded-xl h-[72px]">
                  <label class="cursor-pointer h-full">
                    <input type="radio" name="tone" value="academic" checked class="peer hidden">
                    <div class="h-full flex flex-col justify-center items-center rounded-lg text-xs font-bold text-slate-500 peer-checked:bg-white peer-checked:text-teal-700 peer-checked:shadow-sm transition hover:bg-white/50">
                        <i class="fa-solid fa-list-ul mb-1"></i> Structured
                    </div>
                  </label>
                  <label class="cursor-pointer h-full">
                    <input type="radio" name="tone" value="simple" class="peer hidden">
                    <div class="h-full flex flex-col justify-center items-center rounded-lg text-xs font-bold text-slate-500 peer-checked:bg-white peer-checked:text-teal-700 peer-checked:shadow-sm transition hover:bg-white/50">
                        <i class="fa-solid fa-align-left mb-1"></i> Simple Para
                    </div>
                  </label>
                </div>
              </div>
            </div>

            <button type="submit" class="w-full py-5 rounded-2xl bg-slate-900 text-white font-bold text-lg shadow-xl shadow-slate-900/20 hover:bg-teal-600 hover:shadow-teal-500/30 hover:scale-[1.01] active:scale-[0.99] transition-all duration-200 flex items-center justify-center gap-3">
              Generate Summary <i class="fa-solid fa-arrow-right"></i>
            </button>

          </form>
        </div>
      </div>
    </div>
  </main>

  <div id="progress-overlay" class="fixed inset-0 bg-white/95 backdrop-blur-xl z-[60] hidden flex-col items-center justify-center transition-opacity duration-300">
    <div class="w-full max-w-sm px-8 text-center space-y-8">
      <div class="relative w-24 h-24 mx-auto">
        <svg class="animate-spin w-full h-full text-slate-200" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        </svg>
        <div class="absolute inset-0 flex items-center justify-center font-bold text-slate-800 text-lg" id="progress-percent">0%</div>
      </div>
      <div>
        <h3 class="text-2xl font-bold text-slate-900 mb-2" id="progress-title">Analyzing...</h3>
        <p class="text-slate-500 text-sm font-medium">Our ML is processing the entire document structure.</p>
      </div>
      <div class="h-2 w-full bg-slate-100 rounded-full overflow-hidden">
        <div id="progress-bar" class="h-full bg-teal-500 rounded-full w-0 transition-all duration-200"></div>
      </div>
    </div>
  </div>

  <script>
    const fileInput = document.getElementById('file-input');
    const filePreview = document.getElementById('file-preview');
    const filenameDisplay = document.getElementById('filename-display');
    const resetBtn = document.getElementById('reset-files');
    const form = document.getElementById('uploadForm');
    const overlay = document.getElementById('progress-overlay');
    const pBar = document.getElementById('progress-bar');
    const pPercent = document.getElementById('progress-percent');
    const pTitle = document.getElementById('progress-title');

    fileInput.addEventListener('change', function() {
        if(this.files.length > 0) {
            filePreview.classList.remove('hidden');
            filePreview.classList.add('flex');
            if(this.files.length === 1) {
                filenameDisplay.textContent = this.files[0].name;
            } else {
                filenameDisplay.textContent = `${this.files.length} files selected`;
            }
        }
    });

    resetBtn.addEventListener('click', function(e) {
        e.preventDefault(); 
        fileInput.value = '';
        filePreview.classList.add('hidden');
        filePreview.classList.remove('flex');
    });

    form.addEventListener('submit', function(e) {
        if(fileInput.files.length === 0) {
            e.preventDefault();
            alert("Please select at least one file.");
            return;
        }

        overlay.classList.remove('hidden');
        overlay.classList.add('flex');
        
        // Simulating progress based on selection
        let progress = 0;
        const targetTime = 8000; // 8 seconds roughly
        const interval = 100;
        const step = 100 / (targetTime / interval);
        
        const timer = setInterval(() => {
            progress += step;
            if(progress > 95) progress = 95;
            
            pBar.style.width = `${progress}%`;
            pPercent.textContent = `${Math.round(progress)}%`;

            if(progress < 30) pTitle.textContent = "Uploading & OCR...";
            else if(progress < 70) pTitle.textContent = "Running Vector Analysis...";
            else pTitle.textContent = "Finalizing Summary...";

        }, interval);
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
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
  <script>
    tailwind.config = {
      theme: {
        extend: {
          fontFamily: { sans: ['Plus Jakarta Sans', 'sans-serif'] },
          colors: { teal: { 500: '#14b8a6', 600: '#0d9488', 700: '#0f766e' } }
        }
      }
    }
  </script>
  <style>
    body { background-color: #f8fafc; }
    .custom-scrollbar::-webkit-scrollbar { width: 6px; }
    .custom-scrollbar::-webkit-scrollbar-track { background: #f1f5f9; }
    .custom-scrollbar::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }
    .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: #94a3b8; }
  </style>
</head>
<body class="bg-slate-50 text-slate-800">

  <nav class="sticky top-0 z-40 bg-white/80 backdrop-blur-md border-b border-slate-200">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="flex justify-between h-16 items-center">
        <div class="flex items-center gap-3">
          <div class="w-8 h-8 bg-teal-600 rounded-lg flex items-center justify-center text-white">
            <i class="fa-solid fa-check-double text-xs"></i>
          </div>
          <span class="font-bold text-xl text-slate-900">Summary Result</span>
        </div>
        <div class="flex items-center gap-3">
            <select id="language-select" class="bg-white border border-slate-200 text-slate-700 text-xs font-bold rounded-lg focus:ring-teal-500 focus:border-teal-500 block p-2">
                <option value="en" selected>English</option>
                <option value="hi">Hindi (हिंदी)</option>
                <option value="te">Telugu (తెలుగు)</option>
                <option value="ta">Tamil (தமிழ்)</option>
                <option value="bn">Bengali (বাংলা)</option>
                <option value="mr">Marathi (मराठी)</option>
                <option value="es">Spanish</option>
                <option value="fr">French</option>
            </select>
            <a href="{{ url_for('index') }}" class="hidden sm:inline-flex items-center px-4 py-2 text-xs font-bold rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 transition">
              <i class="fa-solid fa-rotate-left mr-2"></i> New
            </a>
        </div>
      </div>
    </div>
  </nav>

  <main class="py-10 px-4">
    <div class="max-w-7xl mx-auto grid lg:grid-cols-12 gap-8">
      
      <section class="lg:col-span-7 space-y-6">
        <div class="bg-white rounded-2xl shadow-xl shadow-slate-200/60 border border-slate-100 p-8 relative overflow-hidden">
          
          <div class="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-teal-400 to-blue-500"></div>

          <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-8 border-b border-slate-100 pb-6">
            <div>
               <div class="flex items-center gap-2 mb-2">
                  <span class="px-2 py-0.5 rounded bg-teal-50 text-teal-700 text-[10px] font-bold uppercase tracking-wider border border-teal-100">
                    {{ length_mode }} Summary
                  </span>
                  {% if tone_mode == 'simple' %}
                  <span class="px-2 py-0.5 rounded bg-blue-50 text-blue-700 text-[10px] font-bold uppercase tracking-wider border border-blue-100">
                    Simple Tone
                  </span>
                  {% endif %}
               </div>
               <h1 class="text-2xl font-extrabold text-slate-900">Generated Summary</h1>
            </div>
            {% if summary_pdf_url %}
            <a href="{{ summary_pdf_url }}" class="inline-flex items-center justify-center px-5 py-2.5 rounded-xl bg-slate-900 text-white text-xs font-bold hover:bg-teal-600 transition shadow-lg shadow-slate-900/10">
              <i class="fa-solid fa-download mr-2"></i> Download PDF
            </a>
            {% endif %}
          </div>

          <div id="summary-content" class="transition-opacity duration-300">
              
              <div class="mb-10">
                <h2 class="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4 flex items-center gap-2">
                    <i class="fa-regular fa-bookmark"></i> Overview / Abstract
                </h2>
                <div class="p-6 rounded-xl bg-slate-50 border border-slate-100 text-sm md:text-base leading-relaxed text-slate-700 font-medium">
                    {{ abstract }}
                </div>
              </div>

              {% if tone_mode == 'simple' %}
                 <div>
                    <h2 class="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4 flex items-center gap-2">
                        <i class="fa-solid fa-align-left"></i> Full Summary
                    </h2>
                    <div class="prose prose-slate max-w-none text-sm md:text-base text-slate-700 leading-7">
                        {{ simple_text }}
                    </div>
                 </div>
              {% else %}
                 {% if sections %}
                 <div class="space-y-8">
                    {% for sec in sections %}
                    <div class="p-1">
                       <h3 class="text-lg font-bold text-slate-900 mb-4 flex items-center gap-3">
                         <div class="w-8 h-1 rounded-full bg-teal-500"></div>
                         {{ sec.title }}
                       </h3>
                       <ul class="space-y-3 pl-2">
                         {% for bullet in sec.bullets %}
                         <li class="flex items-start gap-3 text-sm md:text-base text-slate-600 leading-relaxed group hover:bg-slate-50 p-2 rounded-lg transition">
                            <i class="fa-solid fa-caret-right mt-1.5 text-teal-400 group-hover:text-teal-600 transition-colors"></i>
                            <span>{{ bullet }}</span>
                         </li>
                         {% endfor %}
                       </ul>
                    </div>
                    {% endfor %}
                 </div>
                 {% endif %}
              {% endif %}

          </div>
          
          <div id="loading-trans" class="hidden absolute inset-0 bg-white/80 z-20 flex-col items-center justify-center">
             <i class="fa-solid fa-circle-notch fa-spin text-4xl text-teal-600 mb-3"></i>
             <span class="text-sm font-bold text-slate-600">Translating content...</span>
          </div>

        </div>
      </section>

      <section class="lg:col-span-5 space-y-6">
        
        <div class="bg-white rounded-2xl shadow-lg border border-slate-100 p-1">
          <div class="p-4 border-b border-slate-100 flex justify-between items-center bg-slate-50 rounded-t-xl">
             <h2 class="text-xs font-bold text-slate-500 uppercase tracking-widest">Original Text</h2>
             <span class="text-[10px] bg-white border px-2 py-1 rounded text-slate-400">Read-only</span>
          </div>
          <div class="p-4 h-[300px] overflow-y-auto custom-scrollbar bg-white rounded-b-xl text-xs font-mono text-slate-500 leading-relaxed whitespace-pre-wrap">
{{ orig_text }}
          </div>
        </div>

        <div class="bg-slate-900 rounded-2xl shadow-xl shadow-slate-900/20 text-white flex flex-col h-[450px] overflow-hidden">
          <div class="p-5 border-b border-slate-700 bg-slate-800/50 flex items-center gap-3">
            <div class="w-8 h-8 rounded-full bg-teal-500 flex items-center justify-center text-white text-xs shadow-lg shadow-teal-500/50">
                <i class="fa-solid fa-robot"></i>
            </div>
            <div>
                <h2 class="text-sm font-bold">Ask AI Assistant</h2>
                <p class="text-[10px] text-slate-400">Context aware Q&A</p>
            </div>
          </div>
          
          <div id="chat-panel" class="flex-1 overflow-y-auto space-y-4 p-5 custom-scrollbar bg-slate-900">
             <div class="flex gap-3">
                <div class="w-6 h-6 rounded-full bg-teal-500/20 text-teal-400 flex items-center justify-center text-[10px] shrink-0 border border-teal-500/30"><i class="fa-solid fa-robot"></i></div>
                <div class="max-w-[85%] bg-slate-800 rounded-2xl rounded-tl-none p-3 text-xs text-slate-300 leading-relaxed border border-slate-700">
                   Hello! I have read the document. You can ask me to clarify specific points or calculate figures mentioned in the text.
                </div>
             </div>
          </div>

          <div class="p-4 bg-slate-800/50 border-t border-slate-700">
             <div class="relative">
                 <input type="text" id="chat-input" class="w-full pl-4 pr-12 py-3.5 rounded-xl bg-slate-950 border border-slate-700 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-teal-500 focus:ring-1 focus:ring-teal-500 transition" placeholder="Type your question...">
                 <button id="chat-send" class="absolute right-2 top-2 p-1.5 bg-teal-600 hover:bg-teal-500 text-white rounded-lg w-8 h-8 flex items-center justify-center transition">
                    <i class="fa-solid fa-paper-plane text-xs"></i>
                 </button>
             </div>
          </div>
        </div>

      </section>
    </div>
  </main>

  <textarea id="hidden-abstract" class="hidden">{{ abstract }}</textarea>
  <textarea id="hidden-simple" class="hidden">{{ simple_text }}</textarea>
  <textarea id="hidden-sections" class="hidden">{{ sections_json }}</textarea>
  <textarea id="doc-context" class="hidden">{{ doc_context }}</textarea>

  <script>
    // --- Translation Logic ---
    const langSelect = document.getElementById('language-select');
    const summaryContainer = document.getElementById('summary-content');
    const loadingTrans = document.getElementById('loading-trans');
    
    // Store original HTML to revert if needed
    let originalSummaryHTML = summaryContainer.innerHTML;

    langSelect.addEventListener('change', async function() {
        const targetLang = this.value;
        
        if (targetLang === 'en') {
            summaryContainer.innerHTML = originalSummaryHTML;
            return;
        }

        // Show Loader
        loadingTrans.classList.remove('hidden');
        loadingTrans.classList.add('flex');

        try {
            // Collect text to translate
            const abs = document.getElementById('hidden-abstract').value;
            const simp = document.getElementById('hidden-simple').value;
            const secJson = document.getElementById('hidden-sections').value;

            const payload = {
                target_lang: targetLang,
                abstract: abs,
                simple_text: simp,
                sections: secJson ? JSON.parse(secJson) : []
            };

            const res = await fetch('{{ url_for("translate_content") }}', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            });

            const data = await res.json();
            if(data.error) throw new Error(data.error);

            // Rebuild HTML with translated text
            let newHTML = `
              <div class="mb-10">
                <h2 class="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4 flex items-center gap-2">
                    <i class="fa-regular fa-bookmark"></i> Overview / Abstract
                </h2>
                <div class="p-6 rounded-xl bg-slate-50 border border-slate-100 text-sm md:text-base leading-relaxed text-slate-700 font-medium">
                    ${data.abstract}
                </div>
              </div>`;

            if (data.simple_text && data.simple_text.length > 10) {
                 newHTML += `
                 <div>
                    <h2 class="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4 flex items-center gap-2">
                        <i class="fa-solid fa-align-left"></i> Full Summary
                    </h2>
                    <div class="prose prose-slate max-w-none text-sm md:text-base text-slate-700 leading-7">
                        ${data.simple_text}
                    </div>
                 </div>`;
            } else if (data.sections && data.sections.length > 0) {
                 newHTML += '<div class="space-y-8">';
                 data.sections.forEach(sec => {
                    let bulletsHTML = '';
                    sec.bullets.forEach(b => {
                        bulletsHTML += `
                         <li class="flex items-start gap-3 text-sm md:text-base text-slate-600 leading-relaxed group hover:bg-slate-50 p-2 rounded-lg transition">
                            <i class="fa-solid fa-caret-right mt-1.5 text-teal-400 group-hover:text-teal-600 transition-colors"></i>
                            <span>${b}</span>
                         </li>`;
                    });
                    
                    newHTML += `
                    <div class="p-1">
                       <h3 class="text-lg font-bold text-slate-900 mb-4 flex items-center gap-3">
                         <div class="w-8 h-1 rounded-full bg-teal-500"></div>
                         ${sec.title}
                       </h3>
                       <ul class="space-y-3 pl-2">${bulletsHTML}</ul>
                    </div>`;
                 });
                 newHTML += '</div>';
            }

            summaryContainer.innerHTML = newHTML;

        } catch(e) {
            console.error(e);
            alert("Translation failed. Please try again.");
            langSelect.value = 'en'; // Revert
        } finally {
            loadingTrans.classList.add('hidden');
            loadingTrans.classList.remove('flex');
        }
    });


    // --- Chat Logic ---
    const panel = document.getElementById('chat-panel');
    const input = document.getElementById('chat-input');
    const sendBtn = document.getElementById('chat-send');
    const docText = document.getElementById('doc-context').value;

    function addMsg(role, text) {
        const div = document.createElement('div');
        div.className = role === 'user' ? 'flex gap-3 flex-row-reverse' : 'flex gap-3';
        
        const avatar = document.createElement('div');
        avatar.className = `w-6 h-6 rounded-full flex items-center justify-center text-[10px] shrink-0 ${role === 'user' ? 'bg-white text-slate-900' : 'bg-teal-500/20 text-teal-400 border border-teal-500/30'}`;
        avatar.innerHTML = role === 'user' ? '<i class="fa-solid fa-user"></i>' : '<i class="fa-solid fa-robot"></i>';
        
        const bubble = document.createElement('div');
        bubble.className = `max-w-[85%] rounded-2xl p-3 text-xs leading-relaxed ${role === 'user' ? 'bg-teal-600 text-white rounded-tr-none' : 'bg-slate-800 text-slate-300 rounded-tl-none border border-slate-700'}`;
        bubble.textContent = text;

        div.appendChild(avatar);
        div.appendChild(bubble);
        panel.appendChild(div);
        panel.scrollTop = panel.scrollHeight;
    }

    async function sendMessage() {
        const txt = input.value.trim();
        if(!txt) return;
        addMsg('user', txt);
        input.value = '';
        
        // Show typing indicator
        const typingId = 'typing-' + Date.now();
        const typingDiv = document.createElement('div');
        typingDiv.className = 'flex gap-3';
        typingDiv.id = typingId;
        typingDiv.innerHTML = `<div class="w-6 h-6 rounded-full bg-teal-500/20 text-teal-400 flex items-center justify-center text-[10px] border border-teal-500/30"><i class="fa-solid fa-ellipsis fa-bounce"></i></div>`;
        panel.appendChild(typingDiv);
        panel.scrollTop = panel.scrollHeight;
        
        try {
            const res = await fetch('{{ url_for("chat") }}', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ message: txt, doc_text: docText })
            });
            const data = await res.json();
            document.getElementById(typingId).remove();
            addMsg('assistant', data.reply);
        } catch(e) {
            document.getElementById(typingId).remove();
            addMsg('assistant', "Sorry, connection error.");
        }
    }

    sendBtn.onclick = sendMessage;
    input.onkeypress = (e) => { if(e.key === 'Enter') sendMessage(); }
  </script>

</body>
</html>
"""

# ---------------------- TEXT UTILITIES & ML CORE ---------------------- #

def normalize_whitespace(text: str) -> str:
    text = text.replace("\r", " ").replace("\xa0", " ")
    text = re.sub(r'Page \d+ of \d+', '', text)
    text = re.sub(r'\s+', " ", text)
    return text.strip()

def strip_leading_numbering(s: str) -> str:
    return re.sub(r"^\s*\d+(\.\d+)*\s*[:\-\)]?\s*", "", s).strip()

def sentence_split(text: str) -> List[str]:
    text = re.sub(r"\n+", " ", text)
    abbreviations = {
        "Dr.": "Dr<DOT>", "Mr.": "Mr<DOT>", "Ms.": "Ms<DOT>", "Mrs.": "Mrs<DOT>",
        "Fig.": "Fig<DOT>", "No.": "No<DOT>", "Vol.": "Vol<DOT>", "approx.": "approx<DOT>",
        "vs.": "vs<DOT>", "e.g.": "e<DOT>g<DOT>", "i.e.": "i<DOT>e<DOT>"
    }
    for abb, mask in abbreviations.items():
        text = text.replace(abb, mask)

    parts = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+(?=[A-Z"\'“])', text)
    
    sentences = []
    for p in parts:
        for abb, mask in abbreviations.items():
            p = p.replace(mask, abb)
        p = p.strip()
        p = re.sub(r"^[\-\–\•\*]+\s*", "", p)
        p = strip_leading_numbering(p)
        if len(p) < 20: continue 
        if re.match(r'^[0-9\.]+$', p): continue
        sentences.append(p)
        
    return sentences

def extract_text_from_pdf_bytes(raw: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(raw))
        pages = []
        for pg in reader.pages:
            txt = pg.extract_text()
            if txt: pages.append(txt)
        return "\n".join(pages)
    except:
        return ""

# ---------------------- CATEGORIZATION & ML ---------------------- #

POLICY_KEYWORDS = {
    "key goals": [
        "aim", "goal", "objective", "target", "achieve", "reduce", "increase", 
        "coverage", "mortality", "rate", "%", "2025", "2030", "vision", "mission", 
        "outcome", "expectancy", "eliminate"
    ],
    "policy principles": [
        "principle", "equity", "universal", "right", "access", "accountability", 
        "transparency", "inclusive", "patient-centered", "quality", "ethics", 
        "value", "integrity", "holistic"
    ],
    "service delivery": [
        "hospital", "primary care", "secondary care", "tertiary", "referral", 
        "clinic", "health center", "wellness", "ambulance", "emergency", "drug", 
        "diagnostic", "infrastructure", "bed", "supply chain", "logistics"
    ],
    "prevention & promotion": [
        "prevent", "sanitation", "nutrition", "immunization", "vaccine", "tobacco", 
        "alcohol", "hygiene", "awareness", "lifestyle", "pollution", "water", 
        "screening", "diet", "exercise", "community"
    ],
    "human resources": [
        "doctor", "nurse", "staff", "training", "workforce", "recruit", "medical college", 
        "paramedic", "salary", "incentive", "capacity building", "skill", "hrh", 
        "deployment", "specialist"
    ],
    "financing & private sector": [
        "fund", "budget", "finance", "expenditure", "cost", "insurance", "private", 
        "partnership", "ppp", "out-of-pocket", "reimbursement", "allocation", 
        "spending", "gdp", "tax", "claim"
    ],
    "digital health": [
        "digital", "technology", "data", "record", "ehr", "emr", "telemedicine", 
        "mobile", "app", "information system", "cyber", "interoperability", 
        "portal", "online", "software", "ai"
    ],
    "ayush integration": [
        "ayush", "ayurveda", "yoga", "unani", "siddha", "homeopathy", "traditional", 
        "naturopathy", "alternative medicine", "integrative"
    ]
}

def score_sentence_categories(sentence: str) -> str:
    s_lower = sentence.lower()
    scores = {cat: 0 for cat in POLICY_KEYWORDS}
    
    for cat, keywords in POLICY_KEYWORDS.items():
        for kw in keywords:
            if kw in s_lower:
                scores[cat] += 2
    
    if '%' in s_lower or re.search(r'\b20[2-5][0-9]\b', s_lower):
        scores['key goals'] += 2

    best_cat = max(scores, key=scores.get)
    if scores[best_cat] == 0:
        return "other"
    return best_cat

def build_tfidf(sentences: List[str]):
    return TfidfVectorizer(
        stop_words="english", 
        ngram_range=(1, 2), 
        sublinear_tf=True
    ).fit_transform(sentences)

def textrank_scores(sim_mat: np.ndarray, doc_len: int) -> Dict[int, float]:
    np.fill_diagonal(sim_mat, 0.0)
    G = nx.from_numpy_array(sim_mat)
    try:
        pr = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-4)
    except:
        pr = {i: 0.0 for i in range(sim_mat.shape[0])}
    
    scores = {}
    for i in range(sim_mat.shape[0]):
        # Slight position bias to favor intro/outro slightly
        mult = 1.0
        if i < doc_len * 0.05: mult = 1.2
        elif i > doc_len * 0.95: mult = 1.1
        scores[i] = pr.get(i, 0.0) * mult
        
    return scores

def summarize_extractive(raw_text: str, length_choice: str = "medium"):
    """
    Improved Summary Logic:
    Instead of just ratio, we target character counts to ensure 'full document' feel
    appropriate to the user selection.
    Short: ~1000-2000 chars
    Medium: ~5000-7000 chars
    Long: ~10000+ chars (Essentially a dense abridgment)
    """
    cleaned = normalize_whitespace(raw_text)
    sentences = sentence_split(cleaned)
    n = len(sentences)
    
    if n <= 5: return sentences, {}

    # Define target chars
    if length_choice == "short":
        target_chars = 1500
        min_sentences = 5
    elif length_choice == "long":
        target_chars = 12000
        min_sentences = 40
    else: # medium
        target_chars = 5000
        min_sentences = 20

    # 1. Vectorize
    tfidf_mat = build_tfidf(sentences)
    sim_mat = cosine_similarity(tfidf_mat)
    
    # 2. TextRank Scoring
    tr_scores = textrank_scores(sim_mat, n)
    
    # 3. Selection Loop (MMR style, but stopping at char limit)
    selected_idxs = []
    current_chars = 0
    
    # Map indices to scores to sort initially
    candidates = sorted(tr_scores.keys(), key=lambda k: tr_scores[k], reverse=True)
    
    # To reduce redundancy (MMR-lite logic)
    # We greedily pick highest score, but penalize if too similar to existing
    final_candidates = []
    
    for idx in candidates:
        if current_chars >= target_chars and len(selected_idxs) >= min_sentences:
            break
            
        # Check similarity to already selected (simple max similarity check)
        is_redundant = False
        if selected_idxs:
            sims = [sim_mat[idx][existing] for existing in selected_idxs]
            if max(sims) > 0.65: # Threshold for redundancy
                is_redundant = True
        
        if not is_redundant:
            selected_idxs.append(idx)
            current_chars += len(sentences[idx])

    # If we didn't meet min sentences, add more from top remaining
    if len(selected_idxs) < min_sentences and len(selected_idxs) < n:
        remaining = [i for i in candidates if i not in selected_idxs]
        for i in remaining:
            selected_idxs.append(i)
            if len(selected_idxs) >= min_sentences: break

    # Return unsorted (by score) indices first, will sort later based on Tone needs
    return [sentences[i] for i in selected_idxs], selected_idxs

def build_structured_summary(all_sentences: List[str], selected_sentences: List[str], tone: str):
    
    # If "Simple" tone: Sort sentences by their original appearance to create a flow
    # We assume 'selected_sentences' passed here might need re-ordering
    # Note: summarize_extractive returns text list. We need to respect original order.
    
    # Since we lost indices in the return of summarize_extractive, let's just 
    # assume the caller handles sorting if they want flow, or we do a best effort match.
    # ACTUALLY: The best way is to re-match sentences to original list indices, 
    # but `summarize_extractive` output is subset. 
    # Let's fix this in the main route logic.
    
    clean_sents = []
    for s in selected_sentences:
        # Cleanup
        s = re.sub(r'\([^)]*\)', '', s) # remove parens
        s = re.sub(r'\[[\d,\-\s]+\]', '', s) # remove citations
        if len(s) > 15:
            clean_sents.append(s.strip())

    # 1. Abstract generation (top ranked usually)
    abstract = " ".join(clean_sents[:3])
    
    sections = []
    simple_text = ""

    if tone == "simple":
        # Create one giant paragraph
        simple_text = " ".join(clean_sents)
    else:
        # Categorize for Academic
        cat_map = defaultdict(list)
        for s in clean_sents:
            cat = score_sentence_categories(s)
            cat_map[cat].append(s)
        
        section_titles = {
            "key goals": "Key Goals & Targets", 
            "policy principles": "Policy Principles & Vision",
            "service delivery": "Healthcare Delivery Systems", 
            "prevention & promotion": "Prevention & Wellness",
            "human resources": "Workforce (HR)", 
            "financing & private sector": "Financing & Costs",
            "digital health": "Digital Interventions", 
            "ayush integration": "AYUSH / Traditional Medicine",
            "other": "Other Observations"
        }

        for k, title in section_titles.items():
            if cat_map[k]:
                # Unique bullets
                unique = list(dict.fromkeys(cat_map[k]))
                sections.append({"title": title, "bullets": unique})

    return {
        "abstract": abstract,
        "sections": sections,
        "simple_text": simple_text
    }

# ---------------------- GEMINI & UTILS ---------------------- #

def process_images_with_gemini(image_paths: List[str]):
    """
    Handles multiple images (simulating a full document scan).
    Concatenates text extraction.
    """
    if not GEMINI_API_KEY:
        return None, "Gemini API Key missing."

    full_text = ""
    model = genai.GenerativeModel("gemini-2.5-flash")
    
    try:
        # Process sequentially to avoid complexity, append text
        for img_path in image_paths:
            img = Image.open(img_path)
            prompt = "Extract all readable text from this document page strictly."
            response = model.generate_content([prompt, img])
            full_text += response.text + "\n\n"
            
        return full_text, None
    except Exception as e:
        return None, str(e)

def save_summary_pdf(title: str, abstract: str, sections: List[Dict], simple_text: str, out_path: str):
    c = canvas.Canvas(out_path, pagesize=A4)
    width, height = A4
    margin = 50
    y = height - margin
    
    def check_page():
        nonlocal y
        if y < 50:
            c.showPage()
            y = height - margin

    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, title)
    y -= 30
    
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Abstract / Overview")
    y -= 15
    
    c.setFont("Helvetica", 10)
    lines = simpleSplit(abstract, "Helvetica", 10, width - 2*margin)
    for line in lines:
        c.drawString(margin, y, line)
        y -= 12
    y -= 20
    check_page()

    if simple_text:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y, "Full Summary")
        y -= 15
        c.setFont("Helvetica", 10)
        lines = simpleSplit(simple_text, "Helvetica", 10, width - 2*margin)
        for line in lines:
            check_page()
            c.drawString(margin, y, line)
            y -= 12
    else:
        for sec in sections:
            check_page()
            c.setFont("Helvetica-Bold", 11)
            c.drawString(margin, y, sec["title"])
            y -= 15
            
            c.setFont("Helvetica", 10)
            for b in sec["bullets"]:
                blines = simpleSplit(f"• {b}", "Helvetica", 10, width - 2*margin)
                for l in blines:
                    check_page()
                    c.drawString(margin, y, l)
                    y -= 12
                y -= 4
            y -= 10
            
    c.save()

# ---------------------- ROUTES ---------------------- #

@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML)

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/summaries/<path:filename>")
def summary_file(filename):
    return send_from_directory(app.config["SUMMARY_FOLDER"], filename, as_attachment=True)

@app.route("/translate", methods=["POST"])
def translate_content():
    data = request.get_json()
    target_lang = data.get("target_lang", "en")
    abstract = data.get("abstract", "")
    simple_text = data.get("simple_text", "")
    sections = data.get("sections", [])

    try:
        translator = GoogleTranslator(source='auto', target=target_lang)
        
        # Translate Abstract
        t_abstract = translator.translate(abstract) if abstract else ""
        
        # Translate Simple Text (Chunking usually needed for large text, DeepTranslator handles basic chunking but to be safe)
        t_simple = ""
        if simple_text:
            # Basic chunking by 4500 chars to stay safe
            chunks = [simple_text[i:i+4500] for i in range(0, len(simple_text), 4500)]
            t_parts = [translator.translate(c) for c in chunks]
            t_simple = " ".join(t_parts)

        # Translate Sections
        t_sections = []
        for sec in sections:
            t_title = translator.translate(sec['title'])
            t_bullets = []
            for b in sec['bullets']:
                t_bullets.append(translator.translate(b))
            t_sections.append({"title": t_title, "bullets": t_bullets})
            
        return jsonify({
            "abstract": t_abstract,
            "simple_text": t_simple,
            "sections": t_sections
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True, silent=True) or {}
    message = data.get("message", "")
    doc_text = data.get("doc_text", "")
    
    if not GEMINI_API_KEY:
        return jsonify({"reply": "Gemini Key not configured."})
        
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        # Keep context reasonable
        prompt = f"Context: {doc_text[:25000]}\n\nUser: {message}\nAnswer concisely based on context."
        resp = model.generate_content(prompt)
        return jsonify({"reply": resp.text})
    except Exception as e:
        return jsonify({"reply": f"Error: {str(e)}"})

@app.route("/summarize", methods=["POST"])
def summarize():
    files = request.files.getlist("file")
    if not files or files[0].filename == "":
        abort(400, "No file uploaded")
        
    uid = uuid.uuid4().hex
    
    # 1. Aggregate Text
    full_text = ""
    image_paths = []
    
    # Save files and extract text
    for f in files:
        filename = secure_filename(f.filename)
        stored_name = f"{uid}_{filename}"
        stored_path = os.path.join(app.config["UPLOAD_FOLDER"], stored_name)
        f.save(stored_path)
        
        lower = filename.lower()
        
        if lower.endswith(".pdf"):
            with open(stored_path, "rb") as pdf_f:
                full_text += extract_text_from_pdf_bytes(pdf_f.read()) + "\n"
        elif lower.endswith(('.png', '.jpg', '.jpeg', '.webp')):
            image_paths.append(stored_path)
        else:
            # Text file
            with open(stored_path, "r", encoding="utf-8", errors="ignore") as txt_f:
                full_text += txt_f.read() + "\n"

    # If images exist, process them
    if image_paths:
        img_text, err = process_images_with_gemini(image_paths)
        if img_text:
            full_text += img_text
        elif err and not full_text:
             abort(500, f"Image Processing Failed: {err}")

    if len(full_text) < 50:
        abort(400, "Could not extract sufficient text from files.")

    # 2. Parameters
    length = request.form.get("length", "medium")
    tone = request.form.get("tone", "academic") # 'academic' or 'simple'

    # 3. Summarize (ML)
    # Get sentences and their original indices
    # We call summarize_extractive but we need to handle sorting for 'simple' tone here
    
    cleaned_full_text = normalize_whitespace(full_text)
    all_sentences_list = sentence_split(cleaned_full_text)
    
    # Use the ML logic to pick sentences
    # To support "Simple" tone flow, we need to sort the selected sentences by their occurrence in the document
    final_sents, selected_indices = summarize_extractive(cleaned_full_text, length)
    
    if tone == "simple":
        # Sort by index to maintain narrative flow
        selected_indices.sort()
        final_sents = [all_sentences_list[i] for i in selected_indices if i < len(all_sentences_list)]

    # 4. Structure
    structured_data = build_structured_summary(all_sentences_list, final_sents, tone)

    # 5. PDF Generation
    summary_filename = f"{uid}_summary.pdf"
    summary_path = os.path.join(app.config["SUMMARY_FOLDER"], summary_filename)
    save_summary_pdf(
        "Generated Summary",
        structured_data.get("abstract", ""),
        structured_data.get("sections", []),
        structured_data.get("simple_text", ""),
        summary_path
    )
    
    return render_template_string(
        RESULT_HTML,
        title="Med.AI Result",
        orig_text=full_text[:30000], 
        doc_context=full_text[:30000],
        abstract=structured_data.get("abstract", ""),
        sections=structured_data.get("sections", []),
        simple_text=structured_data.get("simple_text", ""),
        sections_json=json.dumps(structured_data.get("sections", [])), # For JS Translation
        summary_pdf_url=url_for("summary_file", filename=summary_filename),
        length_mode=length.capitalize(),
        tone_mode=tone
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
