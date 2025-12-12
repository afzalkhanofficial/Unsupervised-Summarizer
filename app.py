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
# import pytesseract # Not used if using Gemini for Vision, keeping import if needed later
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
  <title>Med | Policy Brief Summarizer</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <script>
    tailwind.config = {
      theme: {
        extend: {
          fontFamily: {
            sans: ['Inter', 'sans-serif'],
            mono: ['JetBrains Mono', 'monospace'],
          },
          colors: {
            teal: { 50: '#f0fdfa', 100: '#ccfbf1', 200: '#99f6e4', 300: '#5eead4', 400: '#2dd4bf', 500: '#14b8a6', 600: '#0d9488', 700: '#0f766e', 800: '#115e59', 900: '#134e4a' },
          },
          animation: {
            'float': 'float 6s ease-in-out infinite',
            'pulse-slow': 'pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
          },
          keyframes: {
            float: {
              '0%, 100%': { transform: 'translateY(0)' },
              '50%': { transform: 'translateY(-10px)' },
            }
          }
        }
      }
    }
  </script>
  <style>
    body { background-color: #f8fafc; }
    .glass-panel {
      background: rgba(255, 255, 255, 0.7);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.5);
    }
    .gradient-text {
      background: linear-gradient(135deg, #0f766e 0%, #0891b2 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    @keyframes progress-stripes {
      from { background-position: 1rem 0; }
      to { background-position: 0 0; }
    }
    .animate-stripes {
      background-image: linear-gradient(45deg,rgba(255,255,255,.15) 25%,transparent 25%,transparent 50%,rgba(255,255,255,.15) 50%,rgba(255,255,255,.15) 75%,transparent 75%,transparent);
      background-size: 1rem 1rem;
      animation: progress-stripes 1s linear infinite;
    }
  </style>
</head>
<body class="text-slate-800 relative overflow-x-hidden min-h-screen flex flex-col">

  <div class="fixed top-[-10%] left-[-10%] w-[40%] h-[40%] bg-teal-200/30 rounded-full blur-3xl -z-10 animate-pulse-slow"></div>
  <div class="fixed bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-blue-200/30 rounded-full blur-3xl -z-10 animate-pulse-slow"></div>

  <nav class="fixed w-full z-40 glass-panel border-b border-slate-200/50">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="flex justify-between h-16 items-center">
        <div class="flex items-center gap-3">
          <div class="w-10 h-10 bg-gradient-to-tr from-teal-600 to-cyan-600 rounded-xl flex items-center justify-center shadow-lg shadow-teal-500/20 text-white">
            <i class="fa-solid fa-staff-snake text-xl"></i>
          </div>
          <span class="font-extrabold text-2xl tracking-tight text-slate-800">
            Med<span class="text-teal-600">.AI</span>
          </span>
        </div>
        <div class="hidden md:flex items-center gap-6 text-xs font-bold uppercase tracking-wider text-slate-500">
          <span>AI Powered Summarizer</span>
          <a href="#workspace" class="px-5 py-2.5 rounded-full bg-slate-900 text-white hover:bg-slate-800 transition shadow-lg shadow-slate-900/20">
            Start Now
          </a>
        </div>
      </div>
    </div>
  </nav>

  <main class="flex-grow pt-32 pb-20 px-4">
    <div class="max-w-5xl mx-auto">
      
      <div class="text-center space-y-6 mb-16">
        <div class="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-teal-50 border border-teal-100 text-teal-700 text-xs font-bold uppercase tracking-wide animate-float">
          <span class="w-2 h-2 rounded-full bg-teal-500 animate-pulse"></span>
          Primary Healthcare Policy Analysis
        </div>
        <h1 class="text-5xl md:text-6xl font-extrabold text-slate-900 leading-tight">
          Simplify Complex <br>
          <span class="gradient-text">Medical Policies</span>
        </h1>
        <p class="text-lg text-slate-600 max-w-2xl mx-auto leading-relaxed">
          Upload PDF, Text, or <span class="font-semibold text-slate-800">Multiple Images</span>. 
          We use Unsupervised ML (TF-IDF + TextRank) to generate accurate summaries spanning the full document.
        </p>
      </div>

      <div id="workspace" class="glass-panel rounded-3xl p-1 shadow-2xl shadow-slate-200/50 max-w-3xl mx-auto">
        <div class="bg-white/50 rounded-[1.3rem] p-6 md:p-10 border border-white/50">
          
          <form id="uploadForm" action="{{ url_for('summarize') }}" method="post" enctype="multipart/form-data" class="space-y-8">
            
            <div class="group relative w-full h-64 border-3 border-dashed border-slate-300 rounded-2xl bg-slate-50/50 hover:bg-teal-50/30 hover:border-teal-400 transition-all duration-300 flex flex-col items-center justify-center cursor-pointer overflow-hidden" id="drop-zone">
              
              <input id="file-input" type="file" name="file" accept=".pdf,.txt,image/*" multiple class="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-20">
              
              <div id="upload-prompt" class="text-center space-y-4 transition-all duration-300 group-hover:scale-105">
                <div class="w-16 h-16 bg-white rounded-full shadow-md flex items-center justify-center mx-auto text-teal-500 text-2xl group-hover:text-teal-600">
                  <i class="fa-solid fa-cloud-arrow-up"></i>
                </div>
                <div>
                  <p class="text-lg font-bold text-slate-700">Click to upload</p>
                  <p class="text-sm text-slate-500 mt-1">PDF, TXT, or Multiple Images</p>
                </div>
                <div class="inline-flex items-center gap-2 px-4 py-2 bg-white rounded-full shadow-sm text-xs font-bold text-slate-600 uppercase tracking-wide border border-slate-200">
                  <i class="fa-solid fa-camera"></i> Camera Ready
                </div>
              </div>

              <div id="file-preview" class="hidden absolute inset-0 bg-white/90 backdrop-blur-sm z-10 flex flex-col items-center justify-center p-6 text-center animate-fade-in">
                 <div id="preview-icon" class="mb-4 text-4xl text-teal-600"></div>
                 <div id="preview-count" class="font-bold text-slate-800 text-lg"></div>
                 <p id="filename-display" class="text-sm text-slate-500 max-w-md break-all mt-2"></p>
                 <p class="text-xs text-teal-600 font-semibold mt-2 uppercase tracking-wider">Ready to Summarize</p>
                 <button type="button" id="change-file-btn" class="mt-4 text-xs text-slate-400 hover:text-slate-600 underline z-30 relative">Change files</button>
              </div>

            </div>

            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div class="bg-white rounded-xl p-4 border border-slate-200 shadow-sm">
                <label class="block text-xs font-bold text-slate-500 uppercase tracking-wider mb-3">Summary Length (Approx)</label>
                <div class="flex bg-slate-100 rounded-lg p-1">
                  <label class="flex-1 text-center cursor-pointer">
                    <input type="radio" name="length" value="short" class="peer hidden">
                    <span class="block py-2 text-xs font-bold text-slate-500 rounded-md peer-checked:bg-white peer-checked:text-teal-700 peer-checked:shadow-sm transition">Short<br><span class="text-[0.6rem] opacity-70">1k+ chars</span></span>
                  </label>
                  <label class="flex-1 text-center cursor-pointer">
                    <input type="radio" name="length" value="medium" checked class="peer hidden">
                    <span class="block py-2 text-xs font-bold text-slate-500 rounded-md peer-checked:bg-white peer-checked:text-teal-700 peer-checked:shadow-sm transition">Medium<br><span class="text-[0.6rem] opacity-70">5k+ chars</span></span>
                  </label>
                  <label class="flex-1 text-center cursor-pointer">
                    <input type="radio" name="length" value="long" class="peer hidden">
                    <span class="block py-2 text-xs font-bold text-slate-500 rounded-md peer-checked:bg-white peer-checked:text-teal-700 peer-checked:shadow-sm transition">Long<br><span class="text-[0.6rem] opacity-70">10k+ chars</span></span>
                  </label>
                </div>
              </div>

              <div class="bg-white rounded-xl p-4 border border-slate-200 shadow-sm">
                <label class="block text-xs font-bold text-slate-500 uppercase tracking-wider mb-3">Tone & Format</label>
                <div class="flex bg-slate-100 rounded-lg p-1">
                  <label class="flex-1 text-center cursor-pointer">
                    <input type="radio" name="tone" value="academic" checked class="peer hidden">
                    <span class="block py-2 text-xs font-bold text-slate-500 rounded-md peer-checked:bg-white peer-checked:text-teal-700 peer-checked:shadow-sm transition">Academic<br><span class="text-[0.6rem] opacity-70">Bullet Points</span></span>
                  </label>
                  <label class="flex-1 text-center cursor-pointer">
                    <input type="radio" name="tone" value="easy" class="peer hidden">
                    <span class="block py-2 text-xs font-bold text-slate-500 rounded-md peer-checked:bg-white peer-checked:text-teal-700 peer-checked:shadow-sm transition">Simple<br><span class="text-[0.6rem] opacity-70">One Paragraph</span></span>
                  </label>
                </div>
              </div>
            </div>

            <button type="submit" class="w-full py-4 rounded-xl bg-gradient-to-r from-teal-600 to-cyan-700 text-white font-bold text-lg shadow-lg shadow-teal-500/30 hover:shadow-xl hover:scale-[1.02] transition-all duration-200 flex items-center justify-center gap-2">
              <i class="fa-solid fa-wand-magic-sparkles"></i> Generate Summary
            </button>

          </form>
        </div>
      </div>

    </div>
  </main>

  <div id="progress-overlay" class="fixed inset-0 bg-white/95 backdrop-blur-md z-50 hidden flex-col items-center justify-center">
    <div class="w-full max-w-md px-6 text-center space-y-6">
      
      <div class="relative w-20 h-20 mx-auto">
        <div class="absolute inset-0 rounded-full border-4 border-slate-100"></div>
        <div class="absolute inset-0 rounded-full border-4 border-teal-500 border-t-transparent animate-spin"></div>
        <div class="absolute inset-0 flex items-center justify-center text-teal-600 font-bold text-xl" id="progress-text">0%</div>
      </div>

      <div class="space-y-2">
        <h3 class="text-xl font-bold text-slate-900" id="progress-stage">Starting...</h3>
        <p class="text-sm text-slate-500">Please wait while we analyze your document.</p>
      </div>

      <div class="w-full h-3 bg-slate-200 rounded-full overflow-hidden relative">
        <div id="progress-bar" class="h-full bg-gradient-to-r from-teal-400 to-cyan-600 animate-stripes w-0 transition-all duration-300 ease-out"></div>
      </div>
    </div>
  </div>

  <script>
    const fileInput = document.getElementById('file-input');
    const uploadPrompt = document.getElementById('upload-prompt');
    const filePreview = document.getElementById('file-preview');
    const filenameDisplay = document.getElementById('filename-display');
    const previewIcon = document.getElementById('preview-icon');
    const previewCount = document.getElementById('preview-count');
    const changeBtn = document.getElementById('change-file-btn');
    const uploadForm = document.getElementById('uploadForm');
    const progressOverlay = document.getElementById('progress-overlay');
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    const progressStage = document.getElementById('progress-stage');

    // 1. File Upload Preview Logic (Multi-file support)
    fileInput.addEventListener('change', function(e) {
      if (this.files && this.files.length > 0) {
        const count = this.files.length;
        
        // Show preview container, hide prompt
        uploadPrompt.classList.add('opacity-0');
        setTimeout(() => {
            uploadPrompt.classList.add('hidden');
            filePreview.classList.remove('hidden');
        }, 300);
        
        previewCount.textContent = count + (count === 1 ? ' File Selected' : ' Files Selected');
        
        // List first few names
        const names = Array.from(this.files).map(f => f.name).slice(0, 3);
        filenameDisplay.textContent = names.join(', ') + (count > 3 ? ` and ${count - 3} more` : '');

        // Icon logic based on first file
        const firstType = this.files[0].type;
        if (firstType.startsWith('image/')) {
           previewIcon.innerHTML = '<i class="fa-solid fa-images text-teal-500"></i>';
        } else if (firstType === 'application/pdf') {
           previewIcon.innerHTML = '<i class="fa-solid fa-file-pdf text-red-500"></i>';
        } else {
           previewIcon.innerHTML = '<i class="fa-solid fa-file-lines text-slate-500"></i>';
        }
      }
    });

    // Change file button
    changeBtn.addEventListener('click', (e) => {
        e.stopPropagation(); 
        fileInput.value = ''; 
        filePreview.classList.add('hidden');
        uploadPrompt.classList.remove('hidden');
        uploadPrompt.classList.remove('opacity-0');
    });

    // 2. Progress Bar Logic
    uploadForm.addEventListener('submit', function(e) {
        if (!fileInput.files.length) {
            e.preventDefault();
            alert("Please select a file first.");
            return;
        }

        progressOverlay.classList.remove('hidden');
        progressOverlay.classList.add('flex');
        
        let width = 0;
        const totalDuration = 8000; 
        const intervalTime = 100;
        const step = 100 / (totalDuration / intervalTime);

        const interval = setInterval(() => {
            if (width >= 95) {
                clearInterval(interval);
                progressStage.textContent = "Finalizing Summary...";
            } else {
                width += step;
                if(Math.random() > 0.5) width += 0.5;
                
                progressBar.style.width = width + '%';
                progressText.textContent = Math.round(width) + '%';

                if (width < 30) {
                    progressStage.textContent = "Processing Files...";
                } else if (width < 70) {
                    progressStage.textContent = "Running ML Algorithms...";
                } else {
                    progressStage.textContent = "Structuring Output...";
                }
            }
        }, intervalTime);
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
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <script>
    tailwind.config = {
      theme: {
        extend: {
          fontFamily: {
            sans: ['Inter', 'sans-serif'],
            mono: ['JetBrains Mono', 'monospace'],
          },
          colors: {
            teal: { 50: '#f0fdfa', 100: '#ccfbf1', 200: '#99f6e4', 300: '#5eead4', 400: '#2dd4bf', 500: '#14b8a6', 600: '#0d9488', 700: '#0f766e', 800: '#115e59', 900: '#134e4a' },
          },
        }
      }
    }
  </script>
</head>
<body class="bg-slate-50 text-slate-800">

  <nav class="fixed w-full z-40 bg-white/80 backdrop-blur-md border-b border-slate-200">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="flex justify-between h-16 items-center">
        <div class="flex items-center gap-3">
          <div class="w-8 h-8 bg-gradient-to-tr from-teal-600 to-cyan-600 rounded-lg flex items-center justify-center text-white">
            <i class="fa-solid fa-staff-snake text-sm"></i>
          </div>
          <span class="font-extrabold text-xl tracking-tight text-slate-900">
            Med<span class="text-teal-600">.AI</span>
          </span>
        </div>
        <a href="{{ url_for('index') }}" class="inline-flex items-center px-4 py-2 text-xs font-bold rounded-full border border-slate-200 hover:border-teal-500 hover:text-teal-600 bg-white transition shadow-sm">
          <i class="fa-solid fa-plus mr-2"></i> New Summary
        </a>
      </div>
    </div>
  </nav>

  <main class="pt-24 pb-12 px-4">
    <div class="max-w-7xl mx-auto grid lg:grid-cols-12 gap-8">
      
      <section class="lg:col-span-7 space-y-6">
        <div class="bg-white rounded-3xl shadow-xl shadow-slate-200/50 border border-slate-100 p-8">
          
          <div class="flex flex-wrap items-start justify-between gap-4 mb-6 border-b border-slate-100 pb-6">
            <div>
              <div class="flex items-center gap-2 mb-2">
                  <span class="px-2 py-1 rounded-md bg-teal-50 text-teal-700 text-[0.65rem] font-bold uppercase tracking-wide border border-teal-100">
                    {{ orig_type }} processed
                  </span>
                  {% if used_model == 'gemini' %}
                  <span class="px-2 py-1 rounded-md bg-violet-50 text-violet-700 text-[0.65rem] font-bold uppercase tracking-wide border border-violet-100">
                    <i class="fa-solid fa-sparkles mr-1"></i> Gemini AI
                  </span>
                  {% else %}
                  <span class="px-2 py-1 rounded-md bg-blue-50 text-blue-700 text-[0.65rem] font-bold uppercase tracking-wide border border-blue-100">
                    ML (TF-IDF + TextRank)
                  </span>
                  {% endif %}
              </div>
              <h1 class="text-2xl font-extrabold text-slate-900 leading-tight">Policy Summary</h1>
            </div>
            
            <div class="flex gap-2">
                <select id="lang-select" onchange="translateSummary()" class="px-3 py-2 rounded-xl bg-slate-50 border border-slate-200 text-xs font-bold text-slate-600 focus:outline-none focus:border-teal-500">
                    <option value="en" selected>English</option>
                    <option value="hi">Hindi (हिंदी)</option>
                    <option value="te">Telugu (తెలుగు)</option>
                    <option value="ta">Tamil (தமிழ்)</option>
                    <option value="bn">Bengali (বাংলা)</option>
                    <option value="mr">Marathi (मराठी)</option>
                    <option value="fr">French</option>
                    <option value="es">Spanish</option>
                </select>

                {% if summary_pdf_url %}
                <a href="{{ summary_pdf_url }}" class="inline-flex items-center px-4 py-2 rounded-xl bg-slate-900 text-white text-xs font-bold hover:bg-teal-600 transition shadow-lg">
                  <i class="fa-solid fa-file-arrow-down mr-2"></i> PDF
                </a>
                {% endif %}
            </div>
          </div>

          <div id="summary-content" class="transition-opacity duration-300">
              
              <div class="mb-8">
                <h2 class="text-sm font-bold text-slate-400 uppercase tracking-widest mb-3 flex items-center gap-2">
                    <i class="fa-solid fa-align-left"></i> Abstract / Intro
                </h2>
                <div class="p-5 rounded-2xl bg-slate-50 border border-slate-100 text-sm leading-relaxed text-slate-700">
                    {{ abstract }}
                </div>
              </div>

              {% if tone_mode == 'easy' %}
                <div class="space-y-6">
                    <div>
                       <h3 class="text-base font-bold text-slate-800 mb-3 flex items-center gap-2">
                         <span class="w-1.5 h-6 rounded-full bg-teal-500 block"></span>
                         Core Summary
                       </h3>
                       <div class="text-sm text-slate-700 leading-7 text-justify">
                           {{ simple_text }}
                       </div>
                    </div>
                </div>
              {% else %}
                {% if sections %}
                <div class="space-y-6">
                    {% for sec in sections %}
                    <div>
                       <h3 class="text-base font-bold text-slate-800 mb-3 flex items-center gap-2">
                         <span class="w-1.5 h-6 rounded-full bg-teal-500 block"></span>
                         {{ sec.title }}
                       </h3>
                       <ul class="space-y-2">
                         {% for bullet in sec.bullets %}
                         <li class="flex items-start gap-3 text-sm text-slate-600">
                            <i class="fa-solid fa-check mt-1 text-teal-500 text-xs"></i>
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
          
          <div id="translation-loader" class="hidden py-10 text-center">
             <i class="fa-solid fa-circle-notch fa-spin text-teal-500 text-2xl"></i>
             <p class="text-xs text-slate-400 mt-2">Translating with AI...</p>
          </div>

        </div>
      </section>

      <section class="lg:col-span-5 space-y-6">
        
        <div class="bg-white rounded-3xl shadow-xl shadow-slate-200/50 border border-slate-100 p-6">
          <h2 class="text-sm font-bold text-slate-400 uppercase tracking-widest mb-4">Original Document</h2>
          <div class="rounded-xl overflow-hidden border border-slate-200 bg-slate-100 h-[300px] relative group">
             {% if orig_type == 'pdf' %}
               <div class="flex items-center justify-center h-full text-slate-400 text-xs text-center p-4">
                 PDF Preview Available in downloaded file.<br>(Browser limitation for local paths)
               </div>
             {% elif orig_type == 'text' %}
               <div class="p-4 overflow-y-auto h-full text-xs font-mono">{{ orig_text }}</div>
             {% elif orig_type == 'image' %}
               <div class="flex items-center justify-center h-full text-slate-400 text-xs">Image processed via Gemini</div>
             {% else %}
               <div class="p-4 overflow-y-auto h-full text-xs font-mono">{{ orig_text }}</div>
             {% endif %}
          </div>
        </div>

        <div class="bg-white rounded-3xl shadow-xl shadow-slate-200/50 border border-slate-100 p-6 flex flex-col h-[400px]">
          <div class="mb-4">
            <h2 class="text-sm font-bold text-slate-800 flex items-center gap-2">
               <i class="fa-solid fa-robot text-teal-600"></i> Ask Gemini
            </h2>
            <p class="text-xs text-slate-400">Ask questions based on the document content.</p>
          </div>
          
          <div id="chat-panel" class="flex-1 overflow-y-auto space-y-3 mb-4 pr-2 custom-scrollbar">
             <div class="flex gap-3">
                <div class="w-8 h-8 rounded-full bg-teal-100 flex items-center justify-center text-teal-600 text-xs shrink-0"><i class="fa-solid fa-robot"></i></div>
                <div class="bg-slate-100 rounded-2xl rounded-tl-none p-3 text-xs text-slate-700 leading-relaxed">
                   Hello! I've analyzed this document. Ask me about specific goals, financing, or strategies.
                </div>
             </div>
          </div>

          <div class="relative">
             <input type="text" id="chat-input" class="w-full pl-4 pr-12 py-3 rounded-full bg-slate-50 border border-slate-200 text-sm focus:outline-none focus:border-teal-500 focus:ring-1 focus:ring-teal-500 transition" placeholder="Type a question...">
             <button id="chat-send" class="absolute right-1 top-1 p-2 bg-teal-600 text-white rounded-full w-8 h-8 flex items-center justify-center hover:bg-teal-700 transition">
                <i class="fa-solid fa-paper-plane text-xs"></i>
             </button>
          </div>
          <textarea id="doc-context" class="hidden">{{ doc_context }}</textarea>
        </div>

      </section>
    </div>
  </main>

  <script>
    // 1. Chat Logic
    const panel = document.getElementById('chat-panel');
    const input = document.getElementById('chat-input');
    const sendBtn = document.getElementById('chat-send');
    const docText = document.getElementById('doc-context').value;

    function addMsg(role, text) {
        const div = document.createElement('div');
        div.className = role === 'user' ? 'flex gap-3 flex-row-reverse' : 'flex gap-3';
        
        const avatar = document.createElement('div');
        avatar.className = `w-8 h-8 rounded-full flex items-center justify-center text-xs shrink-0 ${role === 'user' ? 'bg-slate-800 text-white' : 'bg-teal-100 text-teal-600'}`;
        avatar.innerHTML = role === 'user' ? '<i class="fa-solid fa-user"></i>' : '<i class="fa-solid fa-robot"></i>';
        
        const bubble = document.createElement('div');
        bubble.className = `max-w-[80%] rounded-2xl p-3 text-xs leading-relaxed ${role === 'user' ? 'bg-slate-800 text-white rounded-tr-none' : 'bg-slate-100 text-slate-700 rounded-tl-none'}`;
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
        
        try {
            const res = await fetch('{{ url_for("chat") }}', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ message: txt, doc_text: docText })
            });
            const data = await res.json();
            addMsg('assistant', data.reply);
        } catch(e) {
            addMsg('assistant', "Sorry, I encountered an error.");
        }
    }

    sendBtn.onclick = sendMessage;
    input.onkeypress = (e) => { if(e.key === 'Enter') sendMessage(); }

    // 2. Translation Logic
    const contentDiv = document.getElementById('summary-content');
    const loader = document.getElementById('translation-loader');
    
    // Store original HTML to revert if needed or to use as base for translation
    let originalHTML = contentDiv.innerHTML;

    async function translateSummary() {
        const lang = document.getElementById('lang-select').value;
        
        if (lang === 'en') {
            contentDiv.innerHTML = originalHTML;
            return;
        }

        contentDiv.classList.add('opacity-50');
        loader.classList.remove('hidden');

        try {
            // Get current text content to translate
            // We use the innerText of the abstract and bullets, not raw HTML to be safe
            // But for simplicity in this demo, sending the raw Text of sections
            
            const res = await fetch('{{ url_for("translate") }}', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ 
                    html_content: originalHTML, 
                    target_lang: lang 
                })
            });
            
            const data = await res.json();
            if(data.translated_html) {
                contentDiv.innerHTML = data.translated_html;
            } else {
                alert("Translation failed.");
            }
        } catch(e) {
            console.error(e);
            alert("Translation error.");
        } finally {
            contentDiv.classList.remove('opacity-50');
            loader.classList.add('hidden');
        }
    }
  </script>

</body>
</html>
"""

# ---------------------- TEXT UTILITIES & ML CORE ---------------------- #

def normalize_whitespace(text: str) -> str:
    # Basic cleaning
    text = text.replace("\r", " ").replace("\xa0", " ")
    # Remove excessive PDF headers/footers style artifacts
    text = re.sub(r'Page \d+ of \d+', '', text)
    text = re.sub(r'\s+', " ", text)
    return text.strip()

def strip_leading_numbering(s: str) -> str:
    return re.sub(r"^\s*\d+(\.\d+)*\s*[:\-\)]?\s*", "", s).strip()

def sentence_split(text: str) -> List[str]:
    """
    Improved Sentence Splitter:
    Handles abbreviations (Dr., Mr., Fig., etc.) to avoid false splits.
    """
    text = re.sub(r"\n+", " ", text)
    
    # Pre-mask abbreviations to prevent splitting
    abbreviations = {
        "Dr.": "Dr<DOT>", "Mr.": "Mr<DOT>", "Ms.": "Ms<DOT>", "Mrs.": "Mrs<DOT>",
        "Fig.": "Fig<DOT>", "No.": "No<DOT>", "Vol.": "Vol<DOT>", "approx.": "approx<DOT>",
        "vs.": "vs<DOT>", "e.g.": "e<DOT>g<DOT>", "i.e.": "i<DOT>e<DOT>"
    }
    for abb, mask in abbreviations.items():
        text = text.replace(abb, mask)

    # Split by standard sentence terminators
    # Logic: . ! ? followed by whitespace and a capital letter or quote
    parts = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+(?=[A-Z"\'“])', text)
    
    sentences = []
    for p in parts:
        # Unmask abbreviations
        for abb, mask in abbreviations.items():
            p = p.replace(mask, abb)
            
        p = p.strip()
        p = re.sub(r"^[\-\–\•\*]+\s*", "", p) # Remove bullet start
        p = strip_leading_numbering(p)
        
        # Filter junk
        if len(p) < 15: continue # Too short
        if re.match(r'^[0-9\.]+$', p): continue # Just numbers
        
        sentences.append(p)
        
    return sentences

def extract_text_from_pdf_bytes(raw: bytes) -> str:
    reader = PdfReader(io.BytesIO(raw))
    pages = []
    for pg in reader.pages:
        txt = pg.extract_text()
        if txt:
            pages.append(txt)
    return "\n".join(pages)

# ---------------------- ADVANCED CATEGORIZATION ---------------------- #

# Expanded Dictionary for Higher Accuracy
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
    """
    Scores a sentence against all categories based on keyword density.
    Returns the category with the highest score.
    """
    s_lower = sentence.lower()
    scores = {cat: 0 for cat in POLICY_KEYWORDS}
    
    # Tokenize simply
    words = re.findall(r'\w+', s_lower)
    
    for cat, keywords in POLICY_KEYWORDS.items():
        for kw in keywords:
            if kw in s_lower:
                # Exact match bonus
                scores[cat] += 2
            
    # Boost Goals if it has numbers/percentages
    if '%' in s_lower or re.search(r'\b20[2-5][0-9]\b', s_lower):
        scores['key goals'] += 2

    # Get Max Score
    best_cat = max(scores, key=scores.get)
    
    # Threshold: If the best score is 0, it's "Other"
    if scores[best_cat] == 0:
        return "other"
    
    return best_cat

# ---------------------- ML SUMMARIZER (TextRank + MMR) ---------------------- #

def build_tfidf(sentences: List[str]):
    # Sublinear TF scales counts to logarithmic (helps with varying sentence lengths)
    return TfidfVectorizer(
        stop_words="english", 
        ngram_range=(1, 2), 
        sublinear_tf=True
    ).fit_transform(sentences)

def textrank_scores(sim_mat: np.ndarray, doc_len: int) -> Dict[int, float]:
    """
    Calculates TextRank scores.
    Reduced Position Bias to ensure full document coverage for long summaries.
    """
    np.fill_diagonal(sim_mat, 0.0)
    G = nx.from_numpy_array(sim_mat)
    try:
        pr = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-4)
    except:
        # Fallback for disconnected graphs
        pr = {i: 0.0 for i in range(sim_mat.shape[0])}
    
    scores = {}
    for i in range(sim_mat.shape[0]):
        base_score = pr.get(i, 0.0)
        
        # Slight Position Bias (less aggressive than before)
        mult = 1.0
        if i < doc_len * 0.05: mult = 1.1 # Intro boost
        
        scores[i] = base_score * mult
        
    return scores

def mmr(scores_dict: Dict[int, float], sim_mat: np.ndarray, k: int, lambda_param: float = 0.7) -> List[int]:
    """
    Maximal Marginal Relevance.
    Higher lambda = Relevance focused.
    Lower lambda = Diversity focused.
    """
    indices = list(range(sim_mat.shape[0]))
    scores = np.array([scores_dict.get(i, 0.0) for i in indices])
    
    if scores.max() > 0: 
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
        
    selected = []
    candidates = set(indices)
    
    while len(selected) < k and candidates:
        best_idx = None
        best_mmr = -1e9
        
        for i in list(candidates):
            # Similarity to already selected
            sim_to_selected = 0.0
            if selected:
                sim_to_selected = max([sim_mat[i][j] for j in selected])
            
            # MMR Formula
            curr_mmr = (lambda_param * scores[i]) - ((1 - lambda_param) * sim_to_selected)
            
            if curr_mmr > best_mmr:
                best_mmr = curr_mmr
                best_idx = i
                
        if best_idx is not None:
            selected.append(best_idx)
            candidates.remove(best_idx)
        else:
            break
            
    return selected

def summarize_extractive(raw_text: str, length_choice: str = "medium"):
    # 1. Cleaning
    cleaned = normalize_whitespace(raw_text)
    
    # 2. Splitting
    sentences = sentence_split(cleaned)
    n = len(sentences)
    
    if n <= 3: return sentences, {} # Too short

    # 3. Target Length Logic (Character Count Based)
    # Average sentence length approx 100-150 chars
    avg_char_per_sent = sum(len(s) for s in sentences) / n
    avg_char_per_sent = max(50, avg_char_per_sent) # avoid div by zero issues

    if length_choice == "short":
        target_chars = 1500
        lambda_val = 0.6 # Balance relevance/diversity
    elif length_choice == "long":
        target_chars = 12000
        lambda_val = 0.8 # More relevance to fill content
    else: # medium
        target_chars = 6000
        lambda_val = 0.7

    target_k = int(target_chars / avg_char_per_sent)
    
    # Clamp K
    target_k = max(5, min(target_k, n))

    # 4. Vectorization & Similarity
    tfidf_mat = build_tfidf(sentences)
    sim_mat = cosine_similarity(tfidf_mat)
    
    # 5. Ranking
    tr_scores = textrank_scores(sim_mat, n)
    
    # 6. Selection (MMR)
    # MMR ensures we don't just pick similar sentences, covering the "whole pdf" better
    selected_idxs = mmr(tr_scores, sim_mat, target_k, lambda_param=lambda_val)
    selected_idxs.sort()
    
    final_sents = [sentences[i] for i in selected_idxs]
    return final_sents, {}

def build_structured_summary(summary_sentences: List[str], tone: str):
    
    # 1. Simple Tone: Return a cohesive paragraph
    if tone == "easy":
        # Join sentences.
        simple_text = " ".join(summary_sentences)
        
        # Fallback Abstract (first 2 sentences)
        abstract = " ".join(summary_sentences[:2])
        
        return {
            "abstract": abstract,
            "simple_text": simple_text,
            "tone_mode": "easy"
        }

    # 2. Academic Tone: Categorization
    cat_map = defaultdict(list)
    for s in summary_sentences:
        category = score_sentence_categories(s)
        cat_map[category].append(s)
    
    section_titles = {
        "key goals": "Key Goals & Targets", 
        "policy principles": "Policy Principles & Vision",
        "service delivery": "Healthcare Delivery Systems", 
        "prevention & promotion": "Prevention & Wellness",
        "human resources": "Workforce (HR)", 
        "financing & private sector": "Financing & Costs",
        "digital health": "Digital Interventions", 
        "ayush integration": "AYUSH / Traditional Medicine",
        "other": "Other Key Observations"
    }
    
    sections = []
    
    # Clean bullet points slightly
    def clean_bullet(txt):
        txt = re.sub(r'\([^)]*\)', '', txt)
        txt = re.sub(r'^(However|Therefore|Thus|Hence),?\s*', '', txt, flags=re.IGNORECASE)
        txt = re.sub(r'\[[\d,\-\s]+\]', '', txt)
        return txt.strip()

    for k, title in section_titles.items():
        if cat_map[k]:
            unique = list(dict.fromkeys([clean_bullet(s) for s in cat_map[k]]))
            unique = [u for u in unique if len(u) > 10]
            if unique:
                sections.append({"title": title, "bullets": unique})
            
    # Abstract construction
    abstract_candidates = cat_map['key goals'] + cat_map['policy principles'] + summary_sentences
    abstract_cleaned = [clean_bullet(s) for s in abstract_candidates]
    abstract_cleaned = [s for s in abstract_cleaned if len(s) > 10]
    abstract = " ".join(list(dict.fromkeys(abstract_cleaned))[:3])
    
    return {
        "abstract": abstract,
        "sections": sections,
        "tone_mode": "academic"
    }

# ---------------------- GEMINI IMAGE PROCESSING ---------------------- #

def process_image_with_gemini(image_path: str):
    if not GEMINI_API_KEY:
        return None, "Gemini API Key missing."

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        img = Image.open(image_path)
        
        prompt = "Extract all readable text from this image page. Do not summarize yet, just extract."
        response = model.generate_content([prompt, img])
        return response.text.strip(), None
        
    except Exception as e:
        return None, str(e)

# ---------------------- PDF GENERATION ---------------------- #

def save_summary_pdf(title: str, data: Dict, out_path: str):
    c = canvas.Canvas(out_path, pagesize=A4)
    width, height = A4
    margin = 50
    y = height - margin
    
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, title)
    y -= 30
    
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Abstract")
    y -= 15
    
    c.setFont("Helvetica", 10)
    if data.get("abstract"):
        lines = simpleSplit(data["abstract"], "Helvetica", 10, width - 2*margin)
        for line in lines:
            c.drawString(margin, y, line)
            y -= 12
    y -= 10
    
    # Handle Simple vs Academic PDF layout
    if data.get("tone_mode") == "easy":
        text_block = data.get("simple_text", "")
        lines = simpleSplit(text_block, "Helvetica", 10, width - 2*margin)
        for line in lines:
            if y < 50:
                c.showPage(); y = height - margin
            c.drawString(margin, y, line)
            y -= 12
    else:
        for sec in data.get("sections", []):
            if y < 100:
                c.showPage(); y = height - margin
            
            c.setFont("Helvetica-Bold", 11)
            c.drawString(margin, y, sec["title"])
            y -= 15
            
            c.setFont("Helvetica", 10)
            for b in sec["bullets"]:
                blines = simpleSplit(f"• {b}", "Helvetica", 10, width - 2*margin)
                for l in blines:
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

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True, silent=True) or {}
    message = data.get("message", "")
    doc_text = data.get("doc_text", "")
    
    if not GEMINI_API_KEY:
        return jsonify({"reply": "Gemini Key not configured."})
        
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        chat_session = model.start_chat(history=[])
        prompt = f"Context from document: {doc_text[:30000]}\n\nUser Question: {message}\nAnswer concisely."
        resp = chat_session.send_message(prompt)
        return jsonify({"reply": resp.text})
    except Exception as e:
        return jsonify({"reply": f"Error: {str(e)}"})

@app.route("/translate", methods=["POST"])
def translate():
    """
    Translates the HTML content of the summary to target language using Gemini.
    """
    data = request.get_json(force=True, silent=True) or {}
    html_content = data.get("html_content", "")
    target_lang = data.get("target_lang", "en")
    
    if not GEMINI_API_KEY:
        return jsonify({"translated_html": "Translation unavailable (API Key missing)."})
    
    if target_lang == 'en':
        return jsonify({"translated_html": html_content})

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = f"""
        You are a translator. Translate the text content inside the following HTML snippet to {target_lang}.
        Keep all HTML tags, classes, and structure exactly as they are. Only translate the human-readable text.
        
        HTML:
        {html_content}
        """
        response = model.generate_content(prompt)
        # Cleanup markdown formatting if Gemini adds it
        clean_resp = response.text.replace("```html", "").replace("```", "").strip()
        return jsonify({"translated_html": clean_resp})
        
    except Exception as e:
        print(e)
        return jsonify({"translated_html": "Translation failed."})

@app.route("/summarize", methods=["POST"])
def summarize():
    # Handle multiple files
    files = request.files.getlist("file")
    if not files or files[0].filename == "":
        abort(400, "No file uploaded")
        
    uid = uuid.uuid4().hex
    combined_text = []
    first_filename = files[0].filename
    orig_type = "unknown"
    used_model = "ml"

    # Process all files
    for f in files:
        filename = secure_filename(f.filename)
        stored_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{uid}_{filename}")
        f.save(stored_path)
        
        lower_name = filename.lower()
        
        # IMAGE processing
        if lower_name.endswith(('.png', '.jpg', '.jpeg', '.webp')):
            orig_type = "image" # at least one image
            used_model = "gemini" # Used Gemini for extraction
            extracted, err = process_image_with_gemini(stored_path)
            if extracted:
                combined_text.append(extracted)

        # TEXT/PDF processing
        else:
            with open(stored_path, "rb") as f_in:
                raw_bytes = f_in.read()
            
            if lower_name.endswith(".pdf"):
                orig_type = "pdf"
                extracted = extract_text_from_pdf_bytes(raw_bytes)
                combined_text.append(extracted)
            else:
                orig_type = "text"
                extracted = raw_bytes.decode("utf-8", errors="ignore")
                combined_text.append(extracted)

    full_text = "\n\n".join(combined_text)
    
    if len(full_text) < 50:
        abort(400, "Not enough readable text found in documents.")
        
    length = request.form.get("length", "medium")
    tone = request.form.get("tone", "academic")
    
    # Run ML Summarizer
    sents, _ = summarize_extractive(full_text, length)
    structured_data = build_structured_summary(sents, tone)

    # Generate PDF
    summary_filename = f"{uid}_summary.pdf"
    summary_path = os.path.join(app.config["SUMMARY_FOLDER"], summary_filename)
    save_summary_pdf(
        "Policy Summary",
        structured_data,
        summary_path
    )
    
    return render_template_string(
        RESULT_HTML,
        title="Med.AI Summary",
        orig_type=orig_type,
        orig_text=full_text[:20000], 
        doc_context=full_text[:30000],
        abstract=structured_data.get("abstract", ""),
        sections=structured_data.get("sections", []),
        simple_text=structured_data.get("simple_text", ""),
        tone_mode=structured_data.get("tone_mode"),
        summary_pdf_url=url_for("summary_file", filename=summary_filename),
        used_model=used_model
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
