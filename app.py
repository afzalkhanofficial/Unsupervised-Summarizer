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
import pytesseract

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
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <script>
    tailwind.config = {
      theme: {
        extend: {
          fontFamily: {
            sans: ['"Plus Jakarta Sans"', 'sans-serif'],
            mono: ['JetBrains Mono', 'monospace'],
          },
          colors: {
            teal: { 50: '#f0fdfa', 100: '#ccfbf1', 200: '#99f6e4', 300: '#5eead4', 400: '#2dd4bf', 500: '#14b8a6', 600: '#0d9488', 700: '#0f766e', 800: '#115e59', 900: '#134e4a' },
          },
          animation: {
            'blob': 'blob 7s infinite',
          },
          keyframes: {
            blob: {
              '0%': { transform: 'translate(0px, 0px) scale(1)' },
              '33%': { transform: 'translate(30px, -50px) scale(1.1)' },
              '66%': { transform: 'translate(-20px, 20px) scale(0.9)' },
              '100%': { transform: 'translate(0px, 0px) scale(1)' },
            }
          }
        }
      }
    }
  </script>
  <style>
    body { background-color: #f0f4f8; }
    .glass-card {
      background: rgba(255, 255, 255, 0.65);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid rgba(255, 255, 255, 0.8);
      box-shadow: 0 20px 40px -10px rgba(0,0,0,0.05);
    }
    .gradient-text {
      background: linear-gradient(135deg, #0d9488 0%, #0891b2 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    
    /* Progress Bar */
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
<body class="text-slate-800 relative min-h-screen flex flex-col overflow-x-hidden">

  <div class="fixed top-0 left-0 w-full h-full overflow-hidden -z-10 pointer-events-none">
    <div class="absolute top-0 left-1/4 w-96 h-96 bg-teal-200/40 rounded-full mix-blend-multiply filter blur-3xl opacity-70 animate-blob"></div>
    <div class="absolute top-0 right-1/4 w-96 h-96 bg-cyan-200/40 rounded-full mix-blend-multiply filter blur-3xl opacity-70 animate-blob" style="animation-delay: 2s"></div>
    <div class="absolute -bottom-32 left-1/3 w-96 h-96 bg-blue-200/40 rounded-full mix-blend-multiply filter blur-3xl opacity-70 animate-blob" style="animation-delay: 4s"></div>
  </div>

  <nav class="absolute w-full z-40 px-6 py-6">
    <div class="max-w-7xl mx-auto flex justify-between items-center">
        <div class="flex items-center gap-3">
          <div class="w-10 h-10 bg-gradient-to-tr from-teal-600 to-cyan-600 rounded-xl flex items-center justify-center shadow-lg shadow-teal-500/20 text-white">
            <i class="fa-solid fa-staff-snake text-xl"></i>
          </div>
          <span class="font-extrabold text-2xl tracking-tight text-slate-800">
            Med<span class="text-teal-600">.AI</span>
          </span>
        </div>
    </div>
  </nav>

  <main class="flex-grow flex items-center justify-center p-6 pt-24 lg:pt-0">
    <div class="max-w-7xl w-full grid lg:grid-cols-2 gap-12 lg:gap-20 items-center">
      
      <div class="space-y-8 text-center lg:text-left order-2 lg:order-1">
        <div class="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white border border-slate-200 text-teal-700 text-xs font-bold uppercase tracking-wide shadow-sm">
          <span class="w-2 h-2 rounded-full bg-teal-500 animate-pulse"></span>
          Primary Healthcare Policy Analysis
        </div>
        
        <h1 class="text-5xl lg:text-7xl font-extrabold text-slate-900 leading-[1.1] tracking-tight">
          Simplify Complex <br>
          <span class="gradient-text">Medical Policies</span>
        </h1>
        
        <p class="text-lg text-slate-600 leading-relaxed max-w-xl mx-auto lg:mx-0">
          Upload <strong>PDF, Text, or Images</strong>. We use unsupervised ML to generate structured, actionable summaries instantly.
        </p>

        <div class="flex flex-wrap gap-4 justify-center lg:justify-start text-sm font-semibold text-slate-500">
            <span class="flex items-center gap-2"><i class="fa-solid fa-check text-teal-500"></i> No Hallucinations</span>
            <span class="flex items-center gap-2"><i class="fa-solid fa-check text-teal-500"></i> Extractive ML</span>
            <span class="flex items-center gap-2"><i class="fa-solid fa-check text-teal-500"></i> Multi-language Support</span>
        </div>
      </div>

      <div class="glass-card rounded-[2rem] p-2 order-1 lg:order-2 animate-fade-in-up">
        <div class="bg-white/80 rounded-[1.7rem] p-6 md:p-8 border border-white">
          
          <form id="uploadForm" action="{{ url_for('summarize') }}" method="post" enctype="multipart/form-data" class="space-y-6">
            
            <div class="group relative w-full h-56 border-2 border-dashed border-slate-300 rounded-2xl bg-slate-50 hover:bg-teal-50/50 hover:border-teal-400 transition-all duration-300 flex flex-col items-center justify-center cursor-pointer overflow-hidden" id="drop-zone">
              
              <input id="file-input" type="file" name="file" accept=".pdf,.txt,image/*" class="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-20">
              
              <div id="upload-prompt" class="text-center space-y-3 transition-all duration-300 group-hover:scale-105 p-4">
                <div class="w-14 h-14 bg-white rounded-full shadow-md flex items-center justify-center mx-auto text-teal-500 text-2xl group-hover:text-teal-600">
                  <i class="fa-solid fa-cloud-arrow-up"></i>
                </div>
                <div>
                  <p class="text-base font-bold text-slate-700">Drop file or Browse</p>
                  <p class="text-xs text-slate-400 mt-1">PDF, TXT, Images</p>
                </div>
              </div>

              <div id="file-preview" class="hidden absolute inset-0 bg-white z-10 flex flex-col items-center justify-center p-4 text-center">
                 <div id="preview-icon" class="mb-2 text-3xl text-teal-600"></div>
                 <div id="preview-image-container" class="mb-2 hidden rounded-lg overflow-hidden shadow-sm border border-slate-200 max-h-24">
                    <img id="preview-image" src="" alt="Preview" class="h-full object-contain">
                 </div>
                 <p id="filename-display" class="font-bold text-slate-800 text-sm break-all max-w-[80%]"></p>
                 <button type="button" id="change-file-btn" class="mt-2 text-[10px] text-slate-400 hover:text-teal-600 font-bold uppercase tracking-wider z-30 relative">Change</button>
              </div>
            </div>

            <div class="grid grid-cols-2 gap-4">
              <div class="bg-slate-50 rounded-xl p-3 border border-slate-100">
                <label class="block text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-2">Length</label>
                <div class="flex flex-col gap-1">
                  <label class="flex items-center gap-2 cursor-pointer p-1.5 rounded hover:bg-white transition">
                    <input type="radio" name="length" value="short" class="accent-teal-600">
                    <span class="text-xs font-semibold text-slate-600">Short (~1k chars)</span>
                  </label>
                  <label class="flex items-center gap-2 cursor-pointer p-1.5 rounded hover:bg-white transition">
                    <input type="radio" name="length" value="medium" checked class="accent-teal-600">
                    <span class="text-xs font-semibold text-slate-600">Medium (~5k chars)</span>
                  </label>
                  <label class="flex items-center gap-2 cursor-pointer p-1.5 rounded hover:bg-white transition">
                    <input type="radio" name="length" value="long" class="accent-teal-600">
                    <span class="text-xs font-semibold text-slate-600">Long (~10k chars)</span>
                  </label>
                </div>
              </div>

              <div class="bg-slate-50 rounded-xl p-3 border border-slate-100">
                <label class="block text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-2">Style</label>
                <div class="flex flex-col gap-1">
                  <label class="flex items-center gap-2 cursor-pointer p-1.5 rounded hover:bg-white transition">
                    <input type="radio" name="tone" value="academic" checked class="accent-teal-600">
                    <span class="text-xs font-semibold text-slate-600">Academic (Bullets)</span>
                  </label>
                  <label class="flex items-center gap-2 cursor-pointer p-1.5 rounded hover:bg-white transition">
                    <input type="radio" name="tone" value="easy" class="accent-teal-600">
                    <span class="text-xs font-semibold text-slate-600">Simple (Paragraph)</span>
                  </label>
                </div>
              </div>
            </div>

            <button type="submit" class="w-full py-4 rounded-xl bg-slate-900 hover:bg-teal-600 text-white font-bold text-base shadow-lg shadow-slate-900/20 hover:shadow-teal-500/30 transition-all duration-300 transform hover:-translate-y-1">
              Generate Summary
            </button>

          </form>
        </div>
      </div>

    </div>
  </main>

  <div id="progress-overlay" class="fixed inset-0 bg-white/95 backdrop-blur-md z-50 hidden flex-col items-center justify-center">
    <div class="w-full max-w-sm px-6 text-center space-y-6">
      <div class="relative w-16 h-16 mx-auto">
        <div class="absolute inset-0 rounded-full border-4 border-slate-100"></div>
        <div class="absolute inset-0 rounded-full border-4 border-teal-500 border-t-transparent animate-spin"></div>
      </div>
      <div class="space-y-2">
        <h3 class="text-lg font-bold text-slate-900" id="progress-stage">Initializing...</h3>
        <div class="w-full h-2 bg-slate-200 rounded-full overflow-hidden relative">
          <div id="progress-bar" class="h-full bg-teal-500 animate-stripes w-0 transition-all duration-300 ease-out"></div>
        </div>
        <p class="text-xs text-slate-500 font-mono" id="progress-text">0%</p>
      </div>
    </div>
  </div>

  <script>
    const fileInput = document.getElementById('file-input');
    const uploadPrompt = document.getElementById('upload-prompt');
    const filePreview = document.getElementById('file-preview');
    const filenameDisplay = document.getElementById('filename-display');
    const previewIcon = document.getElementById('preview-icon');
    const previewImgContainer = document.getElementById('preview-image-container');
    const previewImg = document.getElementById('preview-image');
    const changeBtn = document.getElementById('change-file-btn');
    const uploadForm = document.getElementById('uploadForm');
    const progressOverlay = document.getElementById('progress-overlay');
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    const progressStage = document.getElementById('progress-stage');

    fileInput.addEventListener('change', function(e) {
      if (this.files && this.files[0]) {
        const file = this.files[0];
        const reader = new FileReader();
        uploadPrompt.classList.add('hidden');
        filePreview.classList.remove('hidden');
        filenameDisplay.textContent = file.name;
        previewImgContainer.classList.add('hidden');
        previewIcon.innerHTML = '';

        if (file.type.startsWith('image/')) {
           reader.onload = function(e) {
             previewImg.src = e.target.result;
             previewImgContainer.classList.remove('hidden');
           }
           reader.readAsDataURL(file);
        } else if (file.type === 'application/pdf') {
           previewIcon.innerHTML = '<i class="fa-solid fa-file-pdf text-red-500"></i>';
        } else {
           previewIcon.innerHTML = '<i class="fa-solid fa-file-lines text-slate-500"></i>';
        }
      }
    });

    changeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.value = '';
        filePreview.classList.add('hidden');
        uploadPrompt.classList.remove('hidden');
    });

    uploadForm.addEventListener('submit', function(e) {
        if (!fileInput.files.length) {
            e.preventDefault();
            alert("Please select a file first.");
            return;
        }
        progressOverlay.classList.remove('hidden');
        progressOverlay.classList.add('flex');
        
        let width = 0;
        const fileType = fileInput.files[0].type;
        const isImage = fileType.startsWith('image/');
        const totalDuration = isImage ? 12000 : 4000; 
        const intervalTime = 100;
        const step = 100 / (totalDuration / intervalTime);

        const interval = setInterval(() => {
            if (width >= 95) {
                clearInterval(interval);
                progressStage.textContent = "Finalizing...";
            } else {
                width += step;
                if(Math.random() > 0.5) width += 0.5;
                progressBar.style.width = width + '%';
                progressText.textContent = Math.round(width) + '%';
                
                if(width < 30) progressStage.textContent = "Uploading...";
                else if(width < 70) progressStage.textContent = "Processing Text...";
                else progressStage.textContent = "Summarizing...";
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
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    body { background-color: #f8fafc; font-family: 'Plus Jakarta Sans', sans-serif; }
    /* Google Translate Styling */
    .goog-te-banner-frame.skiptranslate { display: none !important; } 
    body { top: 0px !important; }
    .goog-te-gadget-simple {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        padding: 8px 12px !important;
        border-radius: 999px !important;
        font-size: 12px !important;
        line-height: 20px !important;
        display: inline-block;
        cursor: pointer;
        zoom: 1;
    }
    .goog-te-gadget-icon { display: none !important; }
    .goog-te-menu-value span:nth-child(1) { color: #0f766e !important; font-weight: bold; }
    .goog-te-menu-value span:nth-child(3) { display: none !important; }
    .goog-te-menu-value span:nth-child(5) { color: #0f766e !important; }
  </style>
</head>
<body class="text-slate-800">

  <nav class="fixed w-full z-40 bg-white/80 backdrop-blur-md border-b border-slate-200">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="flex justify-between h-16 items-center">
        <div class="flex items-center gap-3">
          <div class="w-8 h-8 bg-teal-600 rounded-lg flex items-center justify-center text-white">
            <i class="fa-solid fa-staff-snake text-sm"></i>
          </div>
          <span class="font-extrabold text-xl tracking-tight text-slate-900">
            Med<span class="text-teal-600">.AI</span>
          </span>
        </div>
        <div class="flex items-center gap-4">
            <div id="google_translate_element"></div>
            
            <a href="{{ url_for('index') }}" class="hidden md:inline-flex items-center px-4 py-2 text-xs font-bold rounded-full border border-slate-200 hover:bg-slate-50 transition">
              <i class="fa-solid fa-plus mr-2"></i> New
            </a>
        </div>
      </div>
    </div>
  </nav>

  <main class="pt-24 pb-12 px-4">
    <div class="max-w-7xl mx-auto grid lg:grid-cols-12 gap-8">
      
      <section class="lg:col-span-7 space-y-6">
        <div class="bg-white rounded-3xl shadow-sm border border-slate-200 p-8">
          
          <div class="flex flex-wrap items-center justify-between gap-4 mb-6 border-b border-slate-100 pb-4">
            <h1 class="text-2xl font-extrabold text-slate-900">Policy Summary</h1>
            {% if summary_pdf_url %}
            <a href="{{ summary_pdf_url }}" class="inline-flex items-center px-4 py-2 rounded-lg bg-slate-900 text-white text-xs font-bold hover:bg-teal-600 transition">
              <i class="fa-solid fa-file-arrow-down mr-2"></i> PDF
            </a>
            {% endif %}
          </div>

          {% if mode == 'paragraph' %}
             <div class="prose prose-sm max-w-none text-slate-700 leading-8">
                <p>{{ abstract }}</p>
             </div>
          {% else %}
             <div class="mb-6 p-4 bg-slate-50 rounded-2xl border border-slate-100 text-sm text-slate-700 italic">
                <strong>Abstract:</strong> {{ abstract }}
             </div>

             <div class="space-y-8">
                {% for sec in sections %}
                <div>
                   <h3 class="text-base font-bold text-slate-900 mb-3 flex items-center gap-2 uppercase tracking-wide">
                     <span class="w-2 h-2 rounded-full bg-teal-500"></span>
                     {{ sec.title }}
                   </h3>
                   <ul class="space-y-3">
                     {% for bullet in sec.bullets %}
                     <li class="flex items-start gap-3 text-sm text-slate-600 leading-relaxed">
                        <i class="fa-solid fa-chevron-right mt-1.5 text-[10px] text-teal-400 shrink-0"></i>
                        <span>{{ bullet }}</span>
                     </li>
                     {% endfor %}
                   </ul>
                </div>
                {% endfor %}
             </div>
          {% endif %}

        </div>
      </section>

      <section class="lg:col-span-5 space-y-6">
        
        <div class="bg-white rounded-3xl shadow-sm border border-slate-200 p-6">
          <div class="flex justify-between items-center mb-4">
             <h2 class="text-sm font-bold text-slate-400 uppercase tracking-widest">Source</h2>
             <span class="px-2 py-1 rounded bg-slate-100 text-[10px] font-bold uppercase text-slate-500">{{ orig_type }}</span>
          </div>
          <div class="rounded-xl overflow-hidden border border-slate-200 bg-slate-50 h-[300px]">
             {% if orig_type == 'pdf' %}
               <iframe src="{{ orig_url }}" class="w-full h-full"></iframe>
             {% elif orig_type == 'text' %}
               <div class="p-4 overflow-y-auto h-full text-xs font-mono text-slate-600">{{ orig_text }}</div>
             {% elif orig_type == 'image' %}
               <img src="{{ orig_url }}" class="w-full h-full object-contain">
             {% endif %}
          </div>
        </div>

        <div class="bg-white rounded-3xl shadow-sm border border-slate-200 p-6 h-[450px] flex flex-col">
          <div class="mb-4 border-b border-slate-100 pb-2">
            <h2 class="text-sm font-bold text-slate-800 flex items-center gap-2">
               <i class="fa-solid fa-robot text-teal-600"></i> Ask Med.AI
            </h2>
          </div>
          
          <div id="chat-panel" class="flex-1 overflow-y-auto space-y-4 mb-4 pr-2">
             <div class="flex gap-3">
                <div class="w-8 h-8 rounded-full bg-teal-100 flex items-center justify-center text-teal-600 text-xs shrink-0"><i class="fa-solid fa-robot"></i></div>
                <div class="bg-slate-50 rounded-2xl rounded-tl-none p-3 text-xs text-slate-700 leading-relaxed border border-slate-100">
                   Hello! I've read the document. You can ask me clarifying questions here.
                </div>
             </div>
          </div>

          <div class="relative">
             <input type="text" id="chat-input" class="w-full pl-4 pr-12 py-3 rounded-full bg-slate-50 border border-slate-200 text-sm focus:outline-none focus:border-teal-500 focus:ring-1 focus:ring-teal-500 transition" placeholder="Ask a question...">
             <button id="chat-send" class="absolute right-1 top-1 p-2 bg-teal-600 text-white rounded-full w-8 h-8 flex items-center justify-center hover:bg-teal-700 transition">
                <i class="fa-solid fa-paper-plane text-xs"></i>
             </button>
          </div>
          <textarea id="doc-context" class="hidden">{{ doc_context }}</textarea>
        </div>

      </section>
    </div>
  </main>

  <script type="text/javascript">
    function googleTranslateElementInit() {
      new google.translate.TranslateElement({
        pageLanguage: 'en', 
        includedLanguages: 'hi,te,ta,bn,kn,ml,mr,gu,pa,en', 
        layout: google.translate.TranslateElement.InlineLayout.SIMPLE
      }, 'google_translate_element');
    }
  </script>
  <script type="text/javascript" src="//translate.google.com/translate_a/element.js?cb=googleTranslateElementInit"></script>

  <script>
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
        bubble.className = `max-w-[85%] rounded-2xl p-3 text-xs leading-relaxed ${role === 'user' ? 'bg-slate-800 text-white rounded-tr-none' : 'bg-slate-50 border border-slate-100 text-slate-700 rounded-tl-none'}`;
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
  </script>
</body>
</html>
"""

# ---------------------- TEXT UTILITIES & ML CORE ---------------------- #

def normalize_whitespace(text: str) -> str:
    text = text.replace("\r", " ").replace("\xa0", " ")
    text = re.sub(r'Page \d+ of \d+', '', text) # PDF artifact removal
    text = re.sub(r'\s+', " ", text)
    return text.strip()

def strip_leading_numbering(s: str) -> str:
    # Remove "1.2", "(a)", etc
    return re.sub(r"^\s*(\d+(\.\d+)*|[a-z])\s*[:\-\)\.]?\s*", "", s).strip()

def sentence_split(text: str) -> List[str]:
    """Smart sentence splitting that respects abbreviations."""
    text = re.sub(r"\n+", " ", text)
    abbreviations = { "Dr.": "Dr<DOT>", "Mr.": "Mr<DOT>", "Fig.": "Fig<DOT>", "e.g.": "e<DOT>g<DOT>", "i.e.": "i<DOT>e<DOT>" }
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
        if len(p) > 20: # Filter short junk
            sentences.append(p)
    return sentences

def extract_text_from_pdf_bytes(raw: bytes) -> str:
    reader = PdfReader(io.BytesIO(raw))
    return "\n".join([pg.extract_text() or "" for pg in reader.pages])

# ---------------------- CATEGORIZATION LOGIC ---------------------- #

POLICY_KEYWORDS = {
    "key goals": ["aim", "goal", "target", "achieve", "reduce", "increase", "mortality", "rate", "%", "2025", "2030"],
    "financing": ["fund", "budget", "finance", "expenditure", "cost", "insurance", "spending", "allocation"],
    "service delivery": ["hospital", "primary care", "referral", "ambulance", "drug", "diagnostic", "infrastructure"],
    "human resources": ["doctor", "nurse", "staff", "training", "workforce", "recruit", "paramedic"],
    "digital health": ["digital", "data", "ehr", "telemedicine", "app", "portal", "online"],
    "prevention": ["prevent", "sanitation", "nutrition", "immunization", "tobacco", "hygiene", "awareness"]
}

def score_category(sentence: str) -> str:
    s_lower = sentence.lower()
    scores = {cat: 0 for cat in POLICY_KEYWORDS}
    for cat, kws in POLICY_KEYWORDS.items():
        for kw in kws:
            if kw in s_lower: scores[cat] += 1
    
    # Heuristics
    if re.search(r'\b(20[2-5][0-9]|%)\b', s_lower): scores['key goals'] += 2
    if "rupee" in s_lower or "rs." in s_lower: scores['financing'] += 2
    
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "other"

# ---------------------- SUMMARIZER LOGIC (Updated for Length/Tone) ---------------------- #

def summarize_extractive(raw_text: str, length_choice: str):
    cleaned = normalize_whitespace(raw_text)
    sentences = sentence_split(cleaned)
    n = len(sentences)
    if n < 5: return sentences

    # 1. Calculate Target Characters
    if length_choice == 'short': target_chars = 1200
    elif length_choice == 'medium': target_chars = 5000
    else: target_chars = 12000

    # 2. TextRank
    tfidf = TfidfVectorizer(stop_words='english', sublinear_tf=True).fit_transform(sentences)
    sim_mat = cosine_similarity(tfidf)
    np.fill_diagonal(sim_mat, 0.0)
    scores = nx.pagerank(nx.from_numpy_array(sim_mat))

    # 3. Select Sentences until Char Limit reached
    ranked_indices = sorted(scores, key=scores.get, reverse=True)
    selected_indices = []
    current_chars = 0
    
    # Always take first few sentences (Intro bias)
    for i in range(min(3, n)):
        if i not in selected_indices:
            selected_indices.append(i)
            current_chars += len(sentences[i])

    for idx in ranked_indices:
        if current_chars >= target_chars: break
        if idx not in selected_indices:
            selected_indices.append(idx)
            current_chars += len(sentences[idx])
    
    selected_indices.sort()
    return [sentences[i] for i in selected_indices]

def build_output(sentences: List[str], tone: str):
    # Remove implementation specific code as requested
    
    if tone == 'easy':
        # Single Paragraph Mode
        # Clean text aggressively
        clean_sents = []
        for s in sentences:
            s = re.sub(r'\[.*?\]', '', s) # Remove citations
            s = re.sub(r'\s+', ' ', s)
            clean_sents.append(s)
        
        text_block = " ".join(clean_sents)
        return {"mode": "paragraph", "abstract": text_block, "sections": []}
    
    else:
        # Academic Mode (Categorized Bullets)
        cat_map = defaultdict(list)
        for s in sentences:
            cat_map[score_category(s)].append(s)
        
        sections = []
        titles = {
            "key goals": "Key Goals & Targets",
            "financing": "Financing & Costs",
            "service delivery": "Service Delivery",
            "human resources": "Workforce",
            "digital health": "Digital Health",
            "prevention": "Prevention",
            "other": "Other Points"
        }
        
        for k, title in titles.items():
            if cat_map[k]:
                # Deduplicate and clean
                unique = list(dict.fromkeys(cat_map[k]))
                sections.append({"title": title, "bullets": unique})
        
        abstract = " ".join(sentences[:3]) # First 3 sentences as abstract
        return {"mode": "bullets", "abstract": abstract, "sections": sections}

# ---------------------- GEMINI ---------------------- #

def process_image_gemini(path, tone):
    if not GEMINI_API_KEY: return None, "No API Key"
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        img = Image.open(path)
        
        # Adjust prompt based on tone
        structure_prompt = ""
        if tone == 'easy':
            structure_prompt = "Return a JSON with 'mode': 'paragraph' and 'abstract': 'A single simple paragraph summary'."
        else:
            structure_prompt = "Return a JSON with 'mode': 'bullets', 'abstract': '...', and 'sections': [{'title': '...', 'bullets': ['...']}]"

        prompt = f"Analyze image. Extract text. {structure_prompt}. Return strict JSON."
        resp = model.generate_content([prompt, img])
        text = resp.text.replace("```json", "").replace("```", "")
        return json.loads(text), None
    except Exception as e:
        return None, str(e)

# ---------------------- PDF EXPORT ---------------------- #

def save_pdf(title, data, path):
    c = canvas.Canvas(path, pagesize=A4)
    w, h = A4
    y = h - 50
    margin = 50
    
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, title)
    y -= 30
    
    c.setFont("Helvetica", 10)
    
    text_to_print = []
    if data['mode'] == 'paragraph':
        text_to_print.append(("Summary", data['abstract']))
    else:
        text_to_print.append(("Abstract", data.get('abstract', '')))
        for sec in data.get('sections', []):
            bullets = "\n".join([f"• {b}" for b in sec['bullets']])
            text_to_print.append((sec['title'], bullets))
            
    for header, content in text_to_print:
        if y < 100: c.showPage(); y = h - 50
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y, header)
        y -= 15
        c.setFont("Helvetica", 10)
        lines = simpleSplit(content, "Helvetica", 10, w - 2*margin)
        for line in lines:
            c.drawString(margin, y, line)
            y -= 12
        y -= 15
        
    c.save()

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
    data = request.get_json(force=True, silent=True)
    msg = data.get("message", "")
    context = data.get("doc_text", "")
    if not GEMINI_API_KEY: return jsonify({"reply": "API Key missing"})
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        resp = model.generate_content(f"Context: {context[:30000]}\nQ: {msg}\nShort Answer:")
        return jsonify({"reply": resp.text})
    except Exception as e:
        return jsonify({"reply": str(e)})

@app.route("/summarize", methods=["POST"])
def summarize():
    f = request.files.get("file")
    if not f: abort(400)
    
    # Save File
    fname = secure_filename(f.filename)
    uid = uuid.uuid4().hex
    store_name = f"{uid}_{fname}"
    path = os.path.join(app.config["UPLOAD_FOLDER"], store_name)
    f.save(path)
    
    # Inputs
    length = request.form.get("length", "medium")
    tone = request.form.get("tone", "academic")
    
    # Logic
    orig_text = ""
    orig_type = "unknown"
    data = {}
    used_model = "ml"
    
    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        orig_type = "image"
        used_model = "gemini"
        g_data, err = process_image_gemini(path, tone)
        if err: abort(500, err)
        data = g_data
        orig_text = data.get("extracted_text", "")
    else:
        # PDF/Text -> Local ML
        used_model = "ml"
        with open(path, "rb") as file_in:
            raw = file_in.read()
        
        if fname.lower().endswith(".pdf"):
            orig_type = "pdf"
            orig_text = extract_text_from_pdf_bytes(raw)
        else:
            orig_type = "text"
            orig_text = raw.decode('utf-8', errors='ignore')
            
        sents = summarize_extractive(orig_text, length)
        data = build_output(sents, tone)

    # PDF Gen
    pdf_name = f"{uid}_sum.pdf"
    save_pdf("Policy Summary", data, os.path.join(app.config["SUMMARY_FOLDER"], pdf_name))
    
    return render_template_string(
        RESULT_HTML,
        title="Med.AI Output",
        orig_type=orig_type,
        orig_url=url_for('uploaded_file', filename=store_name),
        orig_text=orig_text[:20000],
        doc_context=orig_text[:20000],
        abstract=data.get('abstract', ''),
        sections=data.get('sections', []),
        mode=data.get('mode', 'bullets'),
        summary_pdf_url=url_for('summary_file', filename=pdf_name),
        used_model=used_model
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
