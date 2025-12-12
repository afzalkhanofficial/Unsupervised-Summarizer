import io
import os
import re
import uuid
import json
import time
from collections import defaultdict
from typing import List, Dict, Any

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
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import simpleSplit
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

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
  <title>Med.AI | Intelligent Policy Summarizer</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
  <script>
    tailwind.config = {
      theme: {
        extend: {
          fontFamily: {
            sans: ['"Plus Jakarta Sans"', 'sans-serif'],
          },
          colors: {
            brand: { 50: '#f0fdfa', 100: '#ccfbf1', 200: '#99f6e4', 300: '#5eead4', 400: '#2dd4bf', 500: '#14b8a6', 600: '#0d9488', 700: '#0f766e', 800: '#115e59', 900: '#134e4a' },
            accent: { 500: '#6366f1', 600: '#4f46e5' }
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
    body { background-color: #f0f4f8; background-image: radial-gradient(#e2e8f0 1px, transparent 1px); background-size: 24px 24px; }
    .glass-card {
      background: rgba(255, 255, 255, 0.85);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid rgba(255, 255, 255, 0.6);
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 10px 15px -3px rgba(0, 0, 0, 0.05);
    }
    .gradient-text {
      background: linear-gradient(135deg, #0f766e 0%, #4f46e5 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .custom-radio:checked + div {
      background-color: #f0fdfa;
      border-color: #14b8a6;
      color: #0f766e;
    }
    .custom-radio:checked + div .check-icon { opacity: 1; transform: scale(1); }
  </style>
</head>
<body class="text-slate-800 min-h-screen flex flex-col">

  <div class="fixed top-0 left-0 w-full h-96 bg-gradient-to-b from-brand-100/50 to-transparent -z-10"></div>
  <div class="fixed top-20 right-[-5%] w-72 h-72 bg-purple-200/30 rounded-full blur-3xl -z-10"></div>
  <div class="fixed bottom-20 left-[-5%] w-72 h-72 bg-brand-200/30 rounded-full blur-3xl -z-10"></div>

  <nav class="fixed w-full z-40 glass-card border-b-0 top-4 left-0 right-0 max-w-7xl mx-auto rounded-2xl h-16 px-6 flex items-center justify-between">
    <div class="flex items-center gap-2">
      <div class="w-8 h-8 bg-brand-600 rounded-lg flex items-center justify-center text-white shadow-lg shadow-brand-500/30">
        <i class="fa-solid fa-notes-medical"></i>
      </div>
      <span class="font-bold text-xl tracking-tight text-slate-800">Med.<span class="text-brand-600">AI</span></span>
    </div>
    <div class="hidden md:flex gap-6 text-sm font-semibold text-slate-500">
      <span class="hover:text-brand-600 cursor-pointer transition">How it Works</span>
      <span class="hover:text-brand-600 cursor-pointer transition">About</span>
    </div>
  </nav>

  <main class="flex-grow pt-32 pb-20 px-4 md:px-8">
    <div class="max-w-4xl mx-auto space-y-12">
      
      <div class="text-center space-y-4 animate-fade-in">
        <div class="inline-block px-4 py-1.5 rounded-full bg-white border border-slate-200 text-xs font-bold uppercase tracking-wider text-brand-700 shadow-sm mb-2">
          <i class="fa-solid fa-sparkles mr-1 text-yellow-500"></i> AI-Powered Analysis
        </div>
        <h1 class="text-4xl md:text-6xl font-extrabold text-slate-900 tracking-tight leading-[1.1]">
          Medical Policy <br>
          <span class="gradient-text">Simplified & Translated</span>
        </h1>
        <p class="text-lg text-slate-600 max-w-2xl mx-auto">
          Transform complex PDF documents into clear, actionable summaries. 
          Choose your language, depth, and tone instantly.
        </p>
      </div>

      <div class="glass-card rounded-3xl p-1 shadow-2xl shadow-slate-300/40 animate-fade-in" style="animation-delay: 0.1s;">
        <div class="bg-white/60 rounded-[1.4rem] p-6 md:p-10">
          
          <form id="uploadForm" action="{{ url_for('summarize') }}" method="post" enctype="multipart/form-data" class="space-y-8">
            
            <div class="relative w-full h-56 border-2 border-dashed border-slate-300 rounded-2xl bg-slate-50/50 hover:bg-brand-50/50 hover:border-brand-400 transition-all duration-300 group cursor-pointer overflow-hidden flex flex-col items-center justify-center text-center">
              <input id="file-input" type="file" name="file" accept=".pdf,.txt,image/*" class="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-20">
              
              <div id="upload-prompt" class="group-hover:scale-105 transition-transform duration-300">
                <div class="w-14 h-14 bg-white rounded-full shadow-md flex items-center justify-center mx-auto text-brand-500 text-2xl mb-4">
                  <i class="fa-solid fa-cloud-arrow-up"></i>
                </div>
                <p class="text-base font-bold text-slate-700">Drop your file here</p>
                <p class="text-sm text-slate-400 mt-1">PDF, TXT, or Image</p>
              </div>

              <div id="file-preview" class="hidden absolute inset-0 bg-white/95 z-30 flex-col items-center justify-center p-4">
                 <div class="text-4xl text-brand-600 mb-2"><i class="fa-regular fa-file-pdf"></i></div>
                 <p id="filename-display" class="font-bold text-slate-800 text-sm truncate max-w-[200px]"></p>
                 <button type="button" id="change-file-btn" class="mt-3 text-xs text-slate-400 hover:text-red-500 font-semibold uppercase tracking-wide">Remove & Change</button>
              </div>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
              
              <div class="space-y-3">
                <label class="text-xs font-bold text-slate-500 uppercase tracking-wider ml-1">Summary Length</label>
                <div class="space-y-2">
                  <label class="cursor-pointer block">
                    <input type="radio" name="length" value="short" class="custom-radio hidden">
                    <div class="border border-slate-200 rounded-xl p-3 flex items-center justify-between hover:border-brand-300 transition bg-white">
                      <div class="flex items-center gap-3">
                        <div class="w-8 h-8 rounded-full bg-slate-100 flex items-center justify-center text-slate-500 text-xs"><i class="fa-solid fa-bolt"></i></div>
                        <div>
                           <p class="text-sm font-bold text-slate-700">Short</p>
                           <p class="text-[10px] text-slate-400">~1,000 chars</p>
                        </div>
                      </div>
                      <i class="fa-solid fa-circle-check text-brand-600 opacity-0 check-icon transition-all transform scale-50"></i>
                    </div>
                  </label>
                  
                  <label class="cursor-pointer block">
                    <input type="radio" name="length" value="medium" checked class="custom-radio hidden">
                    <div class="border border-slate-200 rounded-xl p-3 flex items-center justify-between hover:border-brand-300 transition bg-white">
                      <div class="flex items-center gap-3">
                        <div class="w-8 h-8 rounded-full bg-slate-100 flex items-center justify-center text-slate-500 text-xs"><i class="fa-solid fa-file-lines"></i></div>
                        <div>
                           <p class="text-sm font-bold text-slate-700">Medium</p>
                           <p class="text-[10px] text-slate-400">~5,000 chars</p>
                        </div>
                      </div>
                      <i class="fa-solid fa-circle-check text-brand-600 opacity-0 check-icon transition-all transform scale-50"></i>
                    </div>
                  </label>

                  <label class="cursor-pointer block">
                    <input type="radio" name="length" value="long" class="custom-radio hidden">
                    <div class="border border-slate-200 rounded-xl p-3 flex items-center justify-between hover:border-brand-300 transition bg-white">
                      <div class="flex items-center gap-3">
                        <div class="w-8 h-8 rounded-full bg-slate-100 flex items-center justify-center text-slate-500 text-xs"><i class="fa-solid fa-book-open"></i></div>
                        <div>
                           <p class="text-sm font-bold text-slate-700">Long</p>
                           <p class="text-[10px] text-slate-400">~10,000 chars</p>
                        </div>
                      </div>
                      <i class="fa-solid fa-circle-check text-brand-600 opacity-0 check-icon transition-all transform scale-50"></i>
                    </div>
                  </label>
                </div>
              </div>

              <div class="space-y-3">
                <label class="text-xs font-bold text-slate-500 uppercase tracking-wider ml-1">Tone & Format</label>
                <div class="space-y-2">
                   <label class="cursor-pointer block h-full">
                    <input type="radio" name="tone" value="academic" checked class="custom-radio hidden">
                    <div class="h-full border border-slate-200 rounded-xl p-3 hover:border-brand-300 transition bg-white flex flex-col justify-center">
                      <div class="flex items-center justify-between mb-1">
                         <span class="text-sm font-bold text-slate-700">Academic</span>
                         <i class="fa-solid fa-circle-check text-brand-600 opacity-0 check-icon transition-all transform scale-50"></i>
                      </div>
                      <p class="text-[10px] text-slate-400 leading-tight">Structured bullets, headers, formal language.</p>
                    </div>
                  </label>
                  <label class="cursor-pointer block h-full">
                    <input type="radio" name="tone" value="easy" class="custom-radio hidden">
                    <div class="h-full border border-slate-200 rounded-xl p-3 hover:border-brand-300 transition bg-white flex flex-col justify-center">
                      <div class="flex items-center justify-between mb-1">
                         <span class="text-sm font-bold text-slate-700">Simple</span>
                         <i class="fa-solid fa-circle-check text-brand-600 opacity-0 check-icon transition-all transform scale-50"></i>
                      </div>
                      <p class="text-[10px] text-slate-400 leading-tight">One clean, easy-to-read narrative paragraph.</p>
                    </div>
                  </label>
                </div>
              </div>

              <div class="space-y-3">
                <label class="text-xs font-bold text-slate-500 uppercase tracking-wider ml-1">Output Language</label>
                <div class="relative">
                  <select name="language" class="w-full appearance-none bg-white border border-slate-200 text-slate-700 py-3 px-4 pr-8 rounded-xl font-semibold text-sm focus:outline-none focus:border-brand-500 focus:ring-1 focus:ring-brand-500">
                    <option value="English">🇬🇧 English</option>
                    <option value="Hindi">🇮🇳 Hindi (हिंदी)</option>
                    <option value="Telugu">🇮🇳 Telugu (తెలుగు)</option>
                    <option value="Tamil">🇮🇳 Tamil (தமிழ்)</option>
                    <option value="Spanish">🇪🇸 Spanish</option>
                    <option value="French">🇫🇷 French</option>
                  </select>
                  <div class="pointer-events-none absolute inset-y-0 right-0 flex items-center px-4 text-slate-500">
                    <i class="fa-solid fa-chevron-down text-xs"></i>
                  </div>
                </div>
                <p class="text-[10px] text-slate-400 px-1 leading-relaxed">
                  <i class="fa-solid fa-circle-info text-brand-400"></i> 
                  AI translation is applied after extraction to ensure accuracy.
                </p>
              </div>

            </div>

            <button type="submit" class="w-full py-4 rounded-xl bg-slate-900 text-white font-bold text-lg shadow-xl shadow-slate-900/20 hover:bg-brand-600 hover:shadow-brand-500/30 hover:scale-[1.01] transition-all duration-200 flex items-center justify-center gap-3">
              Generate Summary <i class="fa-solid fa-arrow-right"></i>
            </button>

          </form>
        </div>
      </div>

    </div>
  </main>

  <div id="progress-overlay" class="fixed inset-0 bg-white/90 backdrop-blur-md z-50 hidden flex-col items-center justify-center">
    <div class="w-full max-w-sm text-center">
      <div class="w-16 h-16 bg-brand-50 rounded-2xl flex items-center justify-center mx-auto mb-6 animate-bounce">
         <i class="fa-solid fa-robot text-brand-600 text-2xl"></i>
      </div>
      <h3 class="text-2xl font-bold text-slate-900 mb-2" id="progress-stage">Analyzing Document</h3>
      <p class="text-slate-500 text-sm mb-8" id="progress-desc">Reading full PDF content...</p>
      
      <div class="w-full bg-slate-200 rounded-full h-2 overflow-hidden relative">
        <div id="progress-bar" class="bg-brand-600 h-2 rounded-full w-0 transition-all duration-300"></div>
      </div>
      <p class="text-xs font-bold text-brand-600 mt-2" id="progress-text">0%</p>
    </div>
  </div>

  <script>
    const fileInput = document.getElementById('file-input');
    const filePreview = document.getElementById('file-preview');
    const filenameDisplay = document.getElementById('filename-display');
    const changeBtn = document.getElementById('change-file-btn');
    const uploadForm = document.getElementById('uploadForm');
    const progressOverlay = document.getElementById('progress-overlay');
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    const progressStage = document.getElementById('progress-stage');
    const progressDesc = document.getElementById('progress-desc');

    fileInput.addEventListener('change', function(e) {
      if (this.files && this.files[0]) {
        filenameDisplay.textContent = this.files[0].name;
        filePreview.classList.remove('hidden');
        filePreview.classList.add('flex');
      }
    });

    changeBtn.addEventListener('click', (e) => {
        e.preventDefault();
        fileInput.value = ''; 
        filePreview.classList.add('hidden');
        filePreview.classList.remove('flex');
    });

    uploadForm.addEventListener('submit', function(e) {
        if (!fileInput.files.length) {
            e.preventDefault();
            alert("Please upload a document.");
            return;
        }

        progressOverlay.classList.remove('hidden');
        progressOverlay.classList.add('flex');
        
        let width = 0;
        const interval = setInterval(() => {
            if (width >= 90) {
                clearInterval(interval);
            } else {
                width += Math.random() * 2;
                progressBar.style.width = width + '%';
                progressText.textContent = Math.round(width) + '%';
                
                if (width > 30 && width < 60) {
                    progressStage.textContent = "Extracting Key Points";
                    progressDesc.textContent = "Identifying critical policy data...";
                } else if (width >= 60) {
                    progressStage.textContent = "Finalizing Output";
                    progressDesc.textContent = "Translating and formatting...";
                }
            }
        }, 200);
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
  <title>Summary | {{ title }}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <script>
    tailwind.config = {
      theme: {
        extend: {
          fontFamily: { sans: ['"Plus Jakarta Sans"', 'sans-serif'] },
          colors: {
            brand: { 50: '#f0fdfa', 100: '#ccfbf1', 200: '#99f6e4', 500: '#14b8a6', 600: '#0d9488', 900: '#134e4a' },
          }
        }
      }
    }
  </script>
  <style>
    body { background-color: #f8fafc; }
    .scrollbar-hide::-webkit-scrollbar { display: none; }
    .glass-header { background: rgba(255,255,255,0.9); backdrop-filter: blur(10px); border-bottom: 1px solid #e2e8f0; }
  </style>
</head>
<body class="text-slate-800 h-screen flex flex-col overflow-hidden">

  <header class="glass-header h-16 shrink-0 z-30">
    <div class="h-full max-w-7xl mx-auto px-4 flex items-center justify-between">
      <div class="flex items-center gap-3">
         <a href="{{ url_for('index') }}" class="w-8 h-8 bg-slate-100 rounded-lg flex items-center justify-center text-slate-500 hover:bg-slate-200 transition">
           <i class="fa-solid fa-arrow-left"></i>
         </a>
         <h1 class="font-bold text-lg text-slate-800 hidden md:block">Policy Summary</h1>
         <span class="px-2 py-0.5 rounded-full bg-brand-50 text-brand-600 text-xs font-bold border border-brand-100 uppercase">{{ language }}</span>
      </div>
      <div class="flex items-center gap-3">
        {% if summary_pdf_url %}
        <a href="{{ summary_pdf_url }}" class="flex items-center gap-2 px-4 py-2 bg-brand-600 text-white rounded-lg text-xs font-bold hover:bg-brand-700 transition shadow-md shadow-brand-500/20">
          <i class="fa-solid fa-download"></i> <span class="hidden sm:inline">Download PDF</span>
        </a>
        {% endif %}
      </div>
    </div>
  </header>

  <main class="flex-grow flex flex-col md:flex-row h-full overflow-hidden">
    
    <section class="flex-1 h-full overflow-y-auto p-4 md:p-8 custom-scrollbar bg-white">
      <div class="max-w-3xl mx-auto space-y-8 pb-20">
        
        <div class="flex items-center justify-between border-b border-slate-100 pb-4">
           <div>
             <h2 class="text-3xl font-bold text-slate-900 mb-1">Executive Summary</h2>
             <p class="text-sm text-slate-400">Generated using Extractive AI + Translation</p>
           </div>
           <div class="text-right hidden sm:block">
              <span class="block text-xs font-bold text-slate-400 uppercase">Tone</span>
              <span class="text-sm font-semibold text-brand-600 capitalize">{{ tone }}</span>
           </div>
        </div>

        {% if tone == 'easy' %}
            <div class="prose prose-slate max-w-none">
                <div class="p-6 bg-slate-50 rounded-2xl border border-slate-100 text-base leading-relaxed text-slate-700 shadow-sm text-justify">
                   {{ simple_text | safe }}
                </div>
            </div>
        {% else %}
            <div class="bg-brand-50/50 rounded-2xl p-6 border border-brand-100">
                <h3 class="text-xs font-bold text-brand-600 uppercase tracking-widest mb-3 flex items-center gap-2">
                    <i class="fa-solid fa-bullseye"></i> Abstract
                </h3>
                <p class="text-sm leading-relaxed text-slate-700">{{ abstract }}</p>
            </div>

            <div class="space-y-8">
                {% for sec in sections %}
                <div class="animate-fade-in">
                    <h3 class="text-lg font-bold text-slate-800 mb-4 flex items-center gap-2">
                        <span class="w-1 h-5 bg-brand-500 rounded-full"></span>
                        {{ sec.title }}
                    </h3>
                    <ul class="space-y-3">
                        {% for bullet in sec.bullets %}
                        <li class="flex items-start gap-3 group">
                            <i class="fa-solid fa-angle-right text-xs text-slate-300 mt-1.5 group-hover:text-brand-500 transition-colors"></i>
                            <span class="text-sm text-slate-600 leading-relaxed group-hover:text-slate-900 transition-colors">{{ bullet }}</span>
                        </li>
                        {% endfor %}
                    </ul>
                </div>
                {% endfor %}
            </div>
        {% endif %}

      </div>
    </section>

    <section class="hidden md:flex flex-col w-[400px] border-l border-slate-200 bg-slate-50 h-full">
      
      <div class="flex border-b border-slate-200 bg-white">
        <button class="flex-1 py-3 text-xs font-bold text-brand-600 border-b-2 border-brand-600">
           <i class="fa-solid fa-robot mr-1"></i> AI Assistant
        </button>
        <button class="flex-1 py-3 text-xs font-bold text-slate-400 hover:text-slate-600">
           <i class="fa-solid fa-file-lines mr-1"></i> Source Text
        </button>
      </div>

      <div class="flex-1 flex flex-col overflow-hidden relative">
        <div id="chat-panel" class="flex-1 overflow-y-auto p-4 space-y-4">
            <div class="flex gap-3">
                <div class="w-8 h-8 rounded-full bg-brand-100 flex items-center justify-center text-brand-600 text-xs shrink-0"><i class="fa-solid fa-robot"></i></div>
                <div class="bg-white border border-slate-200 rounded-2xl rounded-tl-none p-3 text-xs text-slate-600 shadow-sm">
                   Hello! I have analyzed the document in <b>{{ language }}</b>. Ask me specific questions about it.
                </div>
            </div>
        </div>

        <div class="p-4 bg-white border-t border-slate-200">
            <div class="relative">
                <input type="text" id="chat-input" class="w-full pl-4 pr-10 py-3 rounded-xl bg-slate-100 border-transparent focus:bg-white focus:border-brand-500 focus:ring-1 focus:ring-brand-500 text-sm transition outline-none" placeholder="Ask a question...">
                <button id="chat-send" class="absolute right-2 top-2 p-1.5 bg-brand-600 text-white rounded-lg hover:bg-brand-700 transition">
                    <i class="fa-solid fa-paper-plane text-xs"></i>
                </button>
            </div>
        </div>
      </div>
      
      <textarea id="doc-context" class="hidden">{{ doc_context }}</textarea>
    </section>

  </main>

  <script>
    const panel = document.getElementById('chat-panel');
    const input = document.getElementById('chat-input');
    const sendBtn = document.getElementById('chat-send');
    const docText = document.getElementById('doc-context').value;

    function addMsg(role, text) {
        const div = document.createElement('div');
        div.className = role === 'user' ? 'flex gap-3 flex-row-reverse' : 'flex gap-3';
        
        const avatar = document.createElement('div');
        avatar.className = `w-8 h-8 rounded-full flex items-center justify-center text-xs shrink-0 ${role === 'user' ? 'bg-slate-800 text-white' : 'bg-brand-100 text-brand-600'}`;
        avatar.innerHTML = role === 'user' ? '<i class="fa-solid fa-user"></i>' : '<i class="fa-solid fa-robot"></i>';
        
        const bubble = document.createElement('div');
        bubble.className = `max-w-[85%] rounded-2xl p-3 text-xs leading-relaxed shadow-sm ${role === 'user' ? 'bg-slate-800 text-white rounded-tr-none' : 'bg-white border border-slate-200 text-slate-700 rounded-tl-none'}`;
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
            addMsg('assistant', "Connection error.");
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
        # Filter junk
        if len(p) < 15 or re.match(r'^[0-9\.]+$', p): continue
        sentences.append(p)
    return sentences

def extract_text_from_pdf_bytes(raw: bytes) -> str:
    reader = PdfReader(io.BytesIO(raw))
    full_text = []
    # Loop through ALL pages to ensure full coverage
    for pg in reader.pages:
        txt = pg.extract_text()
        if txt:
            full_text.append(txt)
    return "\n".join(full_text)

# ---------------------- CATEGORIZATION & SCORING ---------------------- #

POLICY_KEYWORDS = {
    "key goals": ["aim", "goal", "objective", "target", "achieve", "reduce", "increase", "%", "2025", "2030", "outcome"],
    "financing": ["fund", "budget", "finance", "cost", "insurance", "expenditure", "allocation"],
    "infrastructure": ["hospital", "clinic", "bed", "equipment", "digital", "technology", "supply chain"],
    "human_resources": ["staff", "doctor", "nurse", "training", "recruitment", "workforce"],
    "governance": ["committee", "monitor", "regulation", "law", "act", "audit", "compliance"]
}

def score_sentence_category(sentence: str) -> str:
    s_lower = sentence.lower()
    scores = {cat: 0 for cat in POLICY_KEYWORDS}
    for cat, kws in POLICY_KEYWORDS.items():
        for kw in kws:
            if kw in s_lower: scores[cat] += 1
    best_cat = max(scores, key=scores.get)
    return best_cat if scores[best_cat] > 0 else "general"

# ---------------------- SUMMARIZATION LOGIC ---------------------- #

def summarize_extractive(raw_text: str, length_choice: str = "medium"):
    cleaned = normalize_whitespace(raw_text)
    sentences = sentence_split(cleaned)
    n = len(sentences)
    if n == 0: return [], {}

    # 1. Determine Target Sentence Count based on Characters
    # Avg chars per sentence (approx)
    avg_chars = sum(len(s) for s in sentences) / n
    
    if length_choice == "short": target_chars = 1000
    elif length_choice == "long": target_chars = 10000
    else: target_chars = 5000

    target_k = int(target_chars / avg_chars)
    # Clamp target_k between 3 and N
    target_k = max(3, min(target_k, n))

    # 2. Vectorize
    tfidf = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(sentences)
    sim_mat = cosine_similarity(tfidf)

    # 3. TextRank (Importance)
    nx_graph = nx.from_numpy_array(sim_mat)
    try:
        scores = nx.pagerank(nx_graph)
    except:
        scores = {i: 0 for i in range(n)}

    # 4. MMR (Diversity - ensures coverage of whole PDF)
    # We select sentences that are high score but low similarity to already selected
    selected_indices = []
    candidate_indices = list(range(n))
    
    while len(selected_indices) < target_k and candidate_indices:
        best_idx = -1
        best_mmr_val = -1e9
        
        for i in candidate_indices:
            # Relevance
            relevance = scores.get(i, 0)
            
            # Redundancy (max sim to already selected)
            redundancy = 0
            if selected_indices:
                redundancy = max([sim_mat[i][j] for j in selected_indices])
            
            # MMR Score (Lambda 0.7 favors relevance, 0.3 favors diversity)
            mmr_val = 0.7 * relevance - 0.3 * redundancy
            if mmr_val > best_mmr_val:
                best_mmr_val = mmr_val
                best_idx = i
        
        selected_indices.append(best_idx)
        candidate_indices.remove(best_idx)

    selected_indices.sort() # Restore original order
    final_sents = [sentences[i] for i in selected_indices]
    
    return final_sents, {}

def build_structured_data(sentences: List[str]):
    # Categorize sentences for "Academic" view
    cat_map = defaultdict(list)
    for s in sentences:
        cat = score_sentence_category(s)
        cat_map[cat].append(s)

    # Map internal keys to display titles
    titles = {
        "key goals": "Key Goals & Objectives",
        "financing": "Financing & Budget",
        "infrastructure": "Infrastructure & Technology",
        "human_resources": "Workforce & Training",
        "governance": "Governance & Regulation",
        "general": "Key Findings"
    }

    sections = []
    # Prioritize specific order
    order = ["key goals", "financing", "infrastructure", "human_resources", "governance", "general"]
    
    for k in order:
        if cat_map[k]:
            # Dedup and limit
            unique_pts = list(dict.fromkeys(cat_map[k]))
            sections.append({"title": titles[k], "bullets": unique_pts})

    # Create Abstract (First 3 sentences usually)
    abstract = " ".join(sentences[:3])
    
    return abstract, sections

# ---------------------- AI REFINEMENT (TRANSLATION + TONE) ---------------------- #

def refine_with_gemini(text_draft: str, tone: str, language: str):
    """
    Uses Gemini to translate and reformat the extractive summary.
    This handles the "Simple Paragraph" vs "Academic" and Language requirements.
    """
    if not GEMINI_API_KEY:
        return None # Fallback to raw extractive

    model = genai.GenerativeModel("gemini-2.5-flash")
    
    if tone == "easy":
        prompt = f"""
        Act as a professional translator and editor.
        Task: Rewrite the following policy summary into {language}.
        Format: A SINGLE, clean, easy-to-read paragraph. No bullet points.
        Content:
        {text_draft[:15000]}
        """
    else:
        prompt = f"""
        Act as a professional translator and editor.
        Task: Rewrite the following policy summary into {language}.
        Format: Structured format with an Abstract and Bulleted Sections.
        Output JSON: {{ "abstract": "...", "sections": [ {{ "title": "...", "bullets": ["..."] }} ] }}
        Content:
        {text_draft[:15000]}
        """
    
    try:
        resp = model.generate_content(prompt)
        return resp.text
    except:
        return None

# ---------------------- ROUTES ---------------------- #

@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML)

@app.route("/summarize", methods=["POST"])
def summarize():
    f = request.files.get("file")
    if not f or f.filename == "": abort(400)
    
    filename = secure_filename(f.filename)
    raw_bytes = f.read()
    
    # 1. Extract Text
    text = ""
    if filename.lower().endswith(".pdf"):
        text = extract_text_from_pdf_bytes(raw_bytes)
    else:
        text = raw_bytes.decode("utf-8", errors="ignore")

    # 2. Get User Preferences
    length_opt = request.form.get("length", "medium")
    tone_opt = request.form.get("tone", "academic")
    language = request.form.get("language", "English")

    # 3. Extractive Summary (The Foundation)
    # This ensures we get the most important points from the WHOLE PDF first
    extractive_sents, _ = summarize_extractive(text, length_opt)
    extractive_text = " ".join(extractive_sents)

    # 4. AI Refinement (Translation & Formatting)
    # If Gemini is available, use it to Translate + Format
    simple_text_out = ""
    sections_out = []
    abstract_out = ""

    ai_result = refine_with_gemini(extractive_text, tone_opt, language)

    if ai_result:
        # Parse AI Result
        if tone_opt == "easy":
            simple_text_out = ai_result.replace("```", "") # Cleanup
        else:
            # Try to parse JSON from AI
            try:
                clean_json = ai_result.replace("```json", "").replace("```", "").strip()
                data = json.loads(clean_json)
                abstract_out = data.get("abstract", "")
                sections_out = data.get("sections", [])
            except:
                # Fallback if JSON fails
                abstract_out = "Summary generated (Formatting issue)."
                sections_out = [{"title": "Key Points", "bullets": [ai_result]}]
    else:
        # Fallback to Extractive (English Only)
        if tone_opt == "easy":
            simple_text_out = extractive_text
        else:
            abstract_out, sections_out = build_structured_data(extractive_sents)

    # 5. Render Result
    return render_template_string(
        RESULT_HTML,
        title=filename,
        language=language,
        tone=tone_opt,
        simple_text=simple_text_out,
        abstract=abstract_out,
        sections=sections_out,
        doc_context=text[:30000], # For Chat context
        summary_pdf_url="#" # Placeholder
    )

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True, silent=True) or {}
    message = data.get("message", "")
    doc_text = data.get("doc_text", "")
    
    if not GEMINI_API_KEY:
        return jsonify({"reply": "AI Chat unavailable (Missing API Key)."})
        
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        chat = model.start_chat(history=[])
        prompt = f"Context: {doc_text[:25000]}\nUser: {message}\nAnswer briefly."
        resp = chat.send_message(prompt)
        return jsonify({"reply": resp.text})
    except Exception as e:
        return jsonify({"reply": f"Error: {str(e)}"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
