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
# import pytesseract # Uncomment if using OCR locally
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

# Replicating the design from index.html/about.html
COMMON_HEAD = """
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet"/>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <script>
        tailwind.config = {
            darkMode: "class",
            theme: {
                extend: {
                    colors: {
                        "background-light": "#FFFFFF",
                        "background-dark": "#0D0D0F",
                        "surface-dark": "#161b22",
                        "afzal-purple": "#8C4FFF",
                        "afzal-blue": "#4D9CFF",
                        "afzal-red": "#FF5757",
                        "text-light": "#1F2937",
                        "text-dark": "#F3F4F6",
                    },
                    fontFamily: {
                        sans: ['Inter', 'sans-serif'],
                        mono: ['JetBrains Mono', 'monospace'],
                    },
                    backgroundImage: {
                        'grid-pattern-dark': "linear-gradient(to right, rgba(255,255,255,0.05) 1px, transparent 1px), linear-gradient(to bottom, rgba(255,255,255,0.05) 1px, transparent 1px)",
                        'radial-glow': "radial-gradient(circle at center, rgba(140, 79, 255, 0.15) 0%, transparent 70%)",
                    },
                    animation: {
                        'pulse-slow': 'pulse-opacity 4s ease-in-out infinite',
                        'float': 'float 6s ease-in-out infinite',
                    },
                    keyframes: {
                        'pulse-opacity': {
                            '0%, 100%': { opacity: 0.2, transform: 'scale(1)' },
                            '50%': { opacity: 0.5, transform: 'scale(1.1)' },
                        },
                        'float': {
                            '0%, 100%': { transform: 'translateY(0)' },
                            '50%': { transform: 'translateY(-10px)' },
                        }
                    }
                },
            },
        };
    </script>
    <style>
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #0D0D0F; }
        ::-webkit-scrollbar-thumb { background: #374151; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #4b5563; }
        body { background-color: #0D0D0F; color: #F3F4F6; }
        .glass-panel {
            background: rgba(22, 27, 34, 0.6);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .gradient-text {
            background: linear-gradient(to right, #8C4FFF, #4D9CFF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
    </style>
"""

INDEX_HTML = f"""
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8">
  <title>Med.AI | Workspace</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  {COMMON_HEAD}
</head>
<body class="font-sans antialiased text-gray-300 bg-background-dark overflow-x-hidden selection:bg-afzal-purple selection:text-white">

  <div class="fixed inset-0 bg-grid-pattern-dark bg-[size:50px_50px] opacity-20 pointer-events-none"></div>
  <div class="fixed top-[-10%] left-[-10%] w-[40%] h-[40%] bg-afzal-purple/20 rounded-full blur-3xl -z-10 animate-pulse-slow"></div>
  <div class="fixed bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-afzal-blue/20 rounded-full blur-3xl -z-10 animate-pulse-slow"></div>

  <nav class="border-b border-gray-800 sticky top-0 z-50 bg-background-dark/80 backdrop-blur-md h-16">
    <div class="w-full h-full max-w-7xl mx-auto px-6">
        <div class="flex items-center justify-between h-full">
            <a href="/" class="flex items-center space-x-2.5">
                <div class="w-8 h-8 bg-white rounded-full flex items-center justify-center">
                    <i class="fa-solid fa-staff-snake text-black text-lg"></i>
                </div>
                <span class="font-bold text-2xl tracking-tight text-white">Med.AI</span>
            </a>
            <div class="hidden md:flex items-center gap-1">
                 <span class="text-xs font-mono text-afzal-purple bg-afzal-purple/10 border border-afzal-purple/20 px-2 py-1 rounded">v2.5.0-beta</span>
            </div>
        </div>
    </div>
  </nav>

  <main class="relative z-10 pt-16 pb-20 px-4">
    <div class="max-w-4xl mx-auto">
       
      <div class="text-center space-y-6 mb-12">
        <div class="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-surface-dark border border-gray-700 text-gray-400 text-xs font-mono uppercase tracking-wide animate-float">
          <span class="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
          System Operational
        </div>
        <h1 class="text-5xl md:text-6xl font-light text-white leading-tight">
          Summarize Policy<br>
          <span class="gradient-text font-semibold">in Seconds.</span>
        </h1>
        <p class="text-lg text-gray-400 max-w-xl mx-auto font-light">
          Upload PDF, Text, or Images. Our hybrid ML engine extracts key entities and generates structured, actionable insights.
        </p>
      </div>

      <div id="workspace" class="bg-surface-dark border border-gray-800 rounded-xl p-1 shadow-2xl relative overflow-hidden">
        <div class="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-afzal-purple via-afzal-blue to-afzal-purple opacity-50"></div>
        
        <div class="bg-background-dark/50 rounded-[0.7rem] p-6 md:p-10">
           
          <form id="uploadForm" action="{{{{ url_for('summarize') }}}}" method="post" enctype="multipart/form-data" class="space-y-8">
            
            <div class="group relative w-full h-64 border-2 border-dashed border-gray-700 rounded-lg bg-surface-dark/50 hover:bg-surface-dark hover:border-afzal-purple transition-all duration-300 flex flex-col items-center justify-center cursor-pointer overflow-hidden" id="drop-zone">
               
              <input id="file-input" type="file" name="file" accept=".pdf,.txt,image/*" class="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-20">
               
              <div id="upload-prompt" class="text-center space-y-4 transition-all duration-300 group-hover:scale-105 pointer-events-none">
                <div class="w-16 h-16 bg-gray-800 rounded-full flex items-center justify-center mx-auto text-gray-400 text-2xl group-hover:text-afzal-purple group-hover:bg-gray-700 transition-colors">
                  <i class="fa-solid fa-cloud-arrow-up"></i>
                </div>
                <div>
                  <p class="text-lg font-medium text-gray-300">Click to upload or Drag & Drop</p>
                  <p class="text-sm text-gray-500 mt-1 font-mono">PDF, TXT, JPG, PNG</p>
                </div>
              </div>

              <div id="file-preview" class="hidden absolute inset-0 bg-background-dark z-10 flex flex-col items-center justify-center p-6 text-center">
                  <div id="preview-icon" class="mb-4 text-4xl text-afzal-purple"></div>
                  <div id="preview-image-container" class="mb-4 hidden rounded overflow-hidden border border-gray-700 max-h-32">
                     <img id="preview-image" src="" alt="Preview" class="h-full object-contain">
                  </div>
                  <p id="filename-display" class="font-mono text-white text-sm break-all max-w-md bg-surface-dark px-3 py-1 rounded border border-gray-700"></p>
                  <button type="button" id="change-file-btn" class="mt-4 text-xs text-gray-500 hover:text-white underline z-30 relative">Change file</button>
              </div>

            </div>

            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
              
              <div class="bg-surface-dark rounded-lg p-4 border border-gray-800">
                <label class="block text-xs font-bold text-gray-500 uppercase tracking-wider mb-3">Output Length</label>
                <div class="flex bg-background-dark rounded p-1 border border-gray-700">
                  <label class="flex-1 text-center cursor-pointer">
                    <input type="radio" name="length" value="short" class="peer hidden">
                    <span class="block py-2 text-xs font-medium text-gray-400 rounded hover:text-white peer-checked:bg-gray-700 peer-checked:text-white transition">Short</span>
                  </label>
                  <label class="flex-1 text-center cursor-pointer">
                    <input type="radio" name="length" value="medium" checked class="peer hidden">
                    <span class="block py-2 text-xs font-medium text-gray-400 rounded hover:text-white peer-checked:bg-gray-700 peer-checked:text-white transition">Medium</span>
                  </label>
                  <label class="flex-1 text-center cursor-pointer">
                    <input type="radio" name="length" value="long" class="peer hidden">
                    <span class="block py-2 text-xs font-medium text-gray-400 rounded hover:text-white peer-checked:bg-gray-700 peer-checked:text-white transition">Long</span>
                  </label>
                </div>
              </div>

              <div class="bg-surface-dark rounded-lg p-4 border border-gray-800">
                <label class="block text-xs font-bold text-gray-500 uppercase tracking-wider mb-3">Analysis Model</label>
                <div class="flex bg-background-dark rounded p-1 border border-gray-700">
                  <label class="flex-1 text-center cursor-pointer">
                    <input type="radio" name="tone" value="academic" checked class="peer hidden">
                    <span class="block py-2 text-xs font-medium text-gray-400 rounded hover:text-white peer-checked:bg-gray-700 peer-checked:text-white transition">Technical</span>
                  </label>
                  <label class="flex-1 text-center cursor-pointer">
                    <input type="radio" name="tone" value="easy" class="peer hidden">
                    <span class="block py-2 text-xs font-medium text-gray-400 rounded hover:text-white peer-checked:bg-gray-700 peer-checked:text-white transition">Simplified</span>
                  </label>
                </div>
              </div>
            </div>

            <button type="submit" class="w-full py-4 rounded bg-afzal-purple hover:bg-[#7a3ee3] text-white font-semibold text-sm transition-colors shadow-lg shadow-purple-900/20 flex items-center justify-center gap-2">
              <i class="fa-solid fa-bolt"></i> Generate Analysis
            </button>

          </form>
        </div>
      </div>

    </div>
  </main>

  <div id="progress-overlay" class="fixed inset-0 bg-background-dark/95 backdrop-blur-md z-50 hidden flex-col items-center justify-center">
    <div class="w-full max-w-md px-6 text-center space-y-6">
       
      <div class="relative w-20 h-20 mx-auto">
        <div class="absolute inset-0 rounded-full border-2 border-gray-800"></div>
        <div class="absolute inset-0 rounded-full border-2 border-afzal-purple border-t-transparent animate-spin"></div>
        <div class="absolute inset-0 flex items-center justify-center text-white font-mono text-xl" id="progress-text">0%</div>
      </div>

      <div class="space-y-2">
        <h3 class="text-xl font-light text-white" id="progress-stage">Initializing...</h3>
        <p class="text-sm text-gray-500 font-mono">Do not close this window.</p>
      </div>

      <div class="w-full h-1 bg-gray-800 rounded-full overflow-hidden relative">
        <div id="progress-bar" class="h-full bg-gradient-to-r from-afzal-purple to-afzal-blue w-0 transition-all duration-300 ease-out"></div>
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
           previewIcon.innerHTML = '<i class="fa-regular fa-file-pdf"></i>';
        } else {
           previewIcon.innerHTML = '<i class="fa-regular fa-file-lines"></i>';
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
        
        const totalDuration = isImage ? 12000 : 5000; 
        const intervalTime = 100;
        const step = 100 / (totalDuration / intervalTime);

        const interval = setInterval(() => {
            if (width >= 95) {
                clearInterval(interval);
                progressStage.textContent = "Rendering Output...";
            } else {
                width += step;
                if(Math.random() > 0.5) width += 0.5;
                
                progressBar.style.width = width + '%';
                progressText.textContent = Math.round(width) + '%';

                if (width < 30) {
                    progressStage.textContent = "Uploading Data...";
                } else if (width < 70) {
                    progressStage.textContent = "Running Extraction Models...";
                } else {
                    progressStage.textContent = "Structuring JSON...";
                }
            }
        }, intervalTime);
    });
  </script>
</body>
</html>
"""

RESULT_HTML = f"""
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8">
  <title>{{{{ title }}}}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  {COMMON_HEAD}
</head>
<body class="bg-background-dark text-gray-300">

  <nav class="border-b border-gray-800 sticky top-0 z-50 bg-background-dark/80 backdrop-blur-md h-16">
    <div class="w-full h-full max-w-7xl mx-auto px-6">
      <div class="flex justify-between h-full items-center">
        <div class="flex items-center gap-3">
            <a href="/" class="flex items-center space-x-2.5">
                <div class="w-8 h-8 bg-white rounded-full flex items-center justify-center">
                    <i class="fa-solid fa-staff-snake text-black text-lg"></i>
                </div>
                <span class="font-bold text-xl text-white">Med.AI</span>
            </a>
            <span class="text-gray-600">/</span>
            <span class="text-sm font-mono text-afzal-purple">Report</span>
        </div>
        <a href="{{{{ url_for('index') }}}}" class="inline-flex items-center px-4 py-2 text-xs font-semibold rounded bg-surface-dark border border-gray-700 hover:border-white text-white transition-colors">
          <i class="fa-solid fa-plus mr-2"></i> New Analysis
        </a>
      </div>
    </div>
  </nav>

  <main class="pt-10 pb-12 px-4 relative">
    <div class="fixed top-20 left-10 w-64 h-64 bg-afzal-purple/10 rounded-full blur-[100px] pointer-events-none"></div>

    <div class="max-w-7xl mx-auto grid lg:grid-cols-12 gap-8 relative z-10">
       
      <section class="lg:col-span-7 space-y-6">
        <div class="glass-panel rounded-xl p-8 shadow-2xl">
           
          <div class="flex flex-wrap items-start justify-between gap-4 mb-6 border-b border-gray-800 pb-6">
            <div>
              <div class="flex items-center gap-2 mb-2">
                  <span class="px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wide bg-gray-800 text-gray-400 border border-gray-700">
                    {{{{ orig_type }}}} source
                  </span>
                  {{% if used_model == 'gemini' %}}
                  <span class="px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wide bg-afzal-purple/10 text-afzal-purple border border-afzal-purple/20">
                    <i class="fa-solid fa-wand-magic-sparkles mr-1"></i> Gemini 2.5
                  </span>
                  {{% else %}}
                  <span class="px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wide bg-blue-500/10 text-blue-400 border border-blue-500/20">
                    BERTSum + TextRank
                  </span>
                  {{% endif %}}
              </div>
              <h1 class="text-2xl font-light text-white leading-tight">Extracted Summary</h1>
            </div>
             
            {{% if summary_pdf_url %}}
            <a href="{{{{ summary_pdf_url }}}}" class="inline-flex items-center px-4 py-2 rounded bg-white text-black text-xs font-bold hover:bg-gray-200 transition shadow-lg">
              <i class="fa-solid fa-file-arrow-down mr-2"></i> Download PDF
            </a>
            {{% endif %}}
          </div>

          <div class="mb-10">
            <h2 class="text-xs font-bold text-afzal-blue uppercase tracking-widest mb-4 flex items-center gap-2">
                <i class="fa-solid fa-quote-left"></i> Abstract
            </h2>
            <div class="p-5 rounded-lg bg-[#0a0a0c] border border-gray-800 text-sm leading-relaxed text-gray-300 font-light">
                {{{{ abstract }}}}
            </div>
          </div>

          {{% if simple_text %}}
          <div class="mb-8">
            <h2 class="text-xs font-bold text-gray-500 uppercase tracking-widest mb-4">
                Full Text Analysis
            </h2>
            <div class="p-6 rounded-lg bg-surface-dark border border-gray-800 text-sm leading-8 text-gray-300 text-justify">
                {{{{ simple_text }}}}
            </div>
          </div>
          {{% endif %}}

          {{% if sections %}}
          <div class="space-y-8">
            {{% for sec in sections %}}
            <div class="relative pl-4 border-l-2 border-gray-800 hover:border-afzal-purple transition-colors duration-300 group">
               <h3 class="text-lg font-medium text-white mb-3 group-hover:text-afzal-purple transition-colors">
                 {{{{ sec.title }}}}
               </h3>
               <ul class="space-y-3">
                 {{% for bullet in sec.bullets %}}
                 <li class="flex items-start gap-3 text-sm text-gray-400">
                    <span class="mt-1.5 w-1.5 h-1.5 rounded-full bg-gray-600 shrink-0 group-hover:bg-afzal-purple"></span>
                    <span>{{{{ bullet }}}}</span>
                 </li>
                 {{% endfor %}}
               </ul>
            </div>
            {{% endfor %}}
          </div>
          {{% endif %}}

        </div>
      </section>

      <section class="lg:col-span-5 space-y-6">
         
        <div class="glass-panel rounded-xl p-1 shadow-lg">
            <div class="bg-[#0d1117] rounded-lg p-4 border border-gray-800 h-[300px] flex flex-col">
                <div class="flex justify-between items-center mb-2">
                     <h2 class="text-xs font-bold text-gray-500 uppercase tracking-widest">Source Preview</h2>
                </div>
                <div class="flex-1 overflow-hidden border border-gray-800 bg-[#050505] rounded relative">
                     {{% if orig_type == 'pdf' %}}
                       <iframe src="{{{{ orig_url }}}}" class="w-full h-full opacity-80 hover:opacity-100 transition-opacity" title="Original PDF"></iframe>
                     {{% elif orig_type == 'text' %}}
                       <div class="p-4 overflow-y-auto h-full text-xs font-mono text-gray-500">{{{{ orig_text }}}}</div>
                     {{% elif orig_type == 'image' %}}
                       <img src="{{{{ orig_url }}}}" class="w-full h-full object-contain">
                     {{% endif %}}
                </div>
            </div>
        </div>

        <div class="glass-panel rounded-xl p-6 flex flex-col h-[500px] border border-gray-800">
          <div class="mb-4 flex items-center gap-2 pb-4 border-b border-gray-800">
            <div class="w-8 h-8 rounded bg-gradient-to-br from-afzal-purple to-blue-600 flex items-center justify-center text-white shadow-lg">
                <i class="fa-solid fa-robot text-sm"></i>
            </div>
            <div>
                <h2 class="text-sm font-bold text-white">AI Assistant</h2>
                <p class="text-[10px] text-gray-500">Powered by Gemini 1.5 Flash</p>
            </div>
          </div>
           
          <div id="chat-panel" class="flex-1 overflow-y-auto space-y-4 mb-4 pr-2 custom-scrollbar">
             <div class="flex gap-3">
                <div class="w-6 h-6 rounded bg-surface-dark border border-gray-700 flex items-center justify-center text-afzal-purple text-[10px] shrink-0">AI</div>
                <div class="bg-surface-dark border border-gray-800 rounded-lg rounded-tl-none p-3 text-xs text-gray-300 leading-relaxed shadow-sm">
                   Document processed. I can answer specific questions regarding the extracted entities, figures, or policy mandates.
                </div>
             </div>
          </div>

          <div class="relative mt-auto">
             <input type="text" id="chat-input" class="w-full pl-4 pr-12 py-3.5 rounded bg-[#0a0a0c] border border-gray-800 text-sm text-white focus:outline-none focus:border-afzal-purple focus:ring-1 focus:ring-afzal-purple transition placeholder-gray-600" placeholder="Ask a question about this policy...">
             <button id="chat-send" class="absolute right-2 top-2 p-1.5 bg-white text-black rounded hover:bg-gray-200 transition">
                <i class="fa-solid fa-arrow-up text-xs font-bold"></i>
             </button>
          </div>
          <textarea id="doc-context" class="hidden">{{{{ doc_context }}}}</textarea>
        </div>

      </section>
    </div>
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
        avatar.className = `w-6 h-6 rounded flex items-center justify-center text-[10px] shrink-0 border ${role === 'user' ? 'bg-white text-black border-white' : 'bg-surface-dark text-afzal-purple border-gray-700'}`;
        avatar.textContent = role === 'user' ? 'ME' : 'AI';
        
        const bubble = document.createElement('div');
        bubble.className = `max-w-[85%] rounded-lg p-3 text-xs leading-relaxed border ${role === 'user' ? 'bg-afzal-purple text-white border-afzal-purple rounded-tr-none' : 'bg-surface-dark text-gray-300 border-gray-800 rounded-tl-none'}`;
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
        
        // Simulating typing indicator could go here
        
        try {
            const res = await fetch('{{{{ url_for("chat") }}}}', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ message: txt, doc_text: docText })
            });
            const data = await res.json();
            addMsg('assistant', data.reply);
        } catch(e) {
            addMsg('assistant', "Connection interrupted. Please try again.");
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

# ---------------------- ML SUMMARIZER (TextRank + Bucketing) ---------------------- #

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
    """
    np.fill_diagonal(sim_mat, 0.0)
    G = nx.from_numpy_array(sim_mat)
    try:
        pr = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-4)
    except:
        # Fallback for disconnected graphs
        pr = {i: 0.0 for i in range(sim_mat.shape[0])}
    
    return pr

def summarize_extractive(raw_text: str, length_choice: str = "medium"):
    # 1. Cleaning
    cleaned = normalize_whitespace(raw_text)
    
    # 2. Splitting
    sentences = sentence_split(cleaned)
    n = len(sentences)
    
    if n <= 3: return sentences, {} # Too short to summarize

    # 3. Target Length (Characters approximation -> Sentence Count)
    # Average sentence is approx 120-150 chars.
    if length_choice == "short":
        # Target ~1000-1500 chars -> approx 12 sentences
        target_sentences = 12
    elif length_choice == "long":
        # Target ~10000 chars -> approx 90 sentences
        target_sentences = 90
    else: # medium
        # Target ~5000 chars -> approx 45 sentences
        target_sentences = 45

    if target_sentences > n:
        target_sentences = n
    
    # 4. Vectorization & Similarity
    tfidf_mat = build_tfidf(sentences)
    sim_mat = cosine_similarity(tfidf_mat)
    
    # 5. Ranking (TextRank)
    tr_scores = textrank_scores(sim_mat, n)
    
    # 6. Selection Strategy: "BUCKETING" for 100% Coverage
    # To cover the "Whole PDF", we divide the document into 'target_sentences' number of buckets.
    # From each bucket, we pick the sentence with the highest score.
    
    selected_idxs = []
    
    if target_sentences > 0:
        bucket_size = n / target_sentences
        
        for i in range(target_sentences):
            start_idx = int(i * bucket_size)
            end_idx = int((i + 1) * bucket_size)
            
            # Handle edge case for last bucket
            if i == target_sentences - 1:
                end_idx = n
            
            # Find best sentence in this range (bucket)
            best_in_bucket_idx = -1
            best_in_bucket_score = -1.0
            
            # Ensure start < end
            if start_idx >= end_idx:
                # If bucket is empty (rare, small docs), just pick start
                if start_idx < n:
                      selected_idxs.append(start_idx)
                continue

            for j in range(start_idx, end_idx):
                score = tr_scores.get(j, 0.0)
                if score > best_in_bucket_score:
                    best_in_bucket_score = score
                    best_in_bucket_idx = j
            
            if best_in_bucket_idx != -1:
                selected_idxs.append(best_in_bucket_idx)

    selected_idxs.sort()
    
    final_sents = [sentences[i] for i in selected_idxs]
    return final_sents, {}

def build_structured_summary(summary_sentences: List[str], tone: str):
    
    # 1. Simple Tone: Return as single clean paragraph
    if tone == "easy":
        # Join selected sentences.
        text_block = " ".join(summary_sentences)
        # Clean common connector words for flow
        text_block = re.sub(r'\([^)]*\)', '', text_block)
        text_block = re.sub(r'\s+', ' ', text_block)
        return {
            "abstract": summary_sentences[0] if summary_sentences else "No abstract generated.",
            "sections": [],
            "simple_text": text_block,
            "category_counts": {}
        }

    # 2. Academic Tone: Use Categorization
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
    
    # Helper to clean text
    def clean_bullet(txt):
        # Remove citation brackets [1], [12-14]
        txt = re.sub(r'\[[\d,\-\s]+\]', '', txt)
        return txt.strip()

    for k, title in section_titles.items():
        if cat_map[k]:
            unique = list(dict.fromkeys([clean_bullet(s) for s in cat_map[k]]))
            # Filter out empty strings after cleaning
            unique = [u for u in unique if len(u) > 10]
            if unique:
                sections.append({"title": title, "bullets": unique})
            
    # Abstract construction
    abstract_candidates = cat_map['key goals'] + cat_map['policy principles'] + summary_sentences
    # Apply cleaning to abstract too
    abstract_cleaned = [clean_bullet(s) for s in abstract_candidates]
    abstract_cleaned = [s for s in abstract_cleaned if len(s) > 10]
    abstract = " ".join(list(dict.fromkeys(abstract_cleaned))[:3])
    
    return {
        "abstract": abstract,
        "sections": sections,
        "simple_text": None,
        "category_counts": {k: len(v) for k, v in cat_map.items()}
    }

# ---------------------- GEMINI IMAGE PROCESSING ---------------------- #

def process_image_with_gemini(image_path: str):
    if not GEMINI_API_KEY:
        return None, "Gemini API Key missing."

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        img = Image.open(image_path)
        
        prompt = """
        Analyze this image of a policy document. 
        Perform two tasks:
        1. Extract the main text content.
        2. Create a structured summary.
        
        Output strictly valid JSON:
        {
            "extracted_text": "...",
            "summary_structure": {
                "abstract": "...",
                "sections": [
                    { "title": "Key Goals", "bullets": ["..."] },
                    { "title": "Financing", "bullets": ["..."] }
                ]
            }
        }
        """
        response = model.generate_content([prompt, img])
        text_resp = response.text.strip()
        
        if text_resp.startswith("```json"):
            text_resp = text_resp.replace("```json", "").replace("```", "")
        
        data = json.loads(text_resp)
        return data, None
        
    except Exception as e:
        return None, str(e)

# ---------------------- PDF GENERATION ---------------------- #

def save_summary_pdf(title: str, abstract: str, sections: List[Dict], simple_text: str, out_path: str):
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
    if abstract:
        lines = simpleSplit(abstract, "Helvetica", 10, width - 2*margin)
        for line in lines:
            c.drawString(margin, y, line)
            y -= 12
    y -= 10
    
    # Check if simple text mode
    if simple_text:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y, "Full Summary")
        y -= 15
        c.setFont("Helvetica", 10)
        lines = simpleSplit(simple_text, "Helvetica", 10, width - 2*margin)
        for line in lines:
            if y < 50:
                 c.showPage(); y = height - margin
            c.drawString(margin, y, line)
            y -= 12
    else:
        # Sections mode
        for sec in sections:
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
        chat = model.start_chat(history=[])
        prompt = f"Context from document: {doc_text[:30000]}\n\nUser Question: {message}\nAnswer concisely."
        resp = chat.send_message(prompt)
        return jsonify({"reply": resp.text})
    except Exception as e:
        return jsonify({"reply": f"Error: {str(e)}"})

@app.route("/summarize", methods=["POST"])
def summarize():
    f = request.files.get("file")
    if not f or f.filename == "":
        abort(400, "No file uploaded")
        
    filename = secure_filename(f.filename)
    uid = uuid.uuid4().hex
    stored_name = f"{uid}_{filename}"
    stored_path = os.path.join(app.config["UPLOAD_FOLDER"], stored_name)
    f.save(stored_path)
    
    lower_name = filename.lower()
    
    structured_data = {}
    orig_text = ""
    orig_type = "unknown"
    used_model = "ml" 
    
    # CASE 1: IMAGE -> GEMINI
    if lower_name.endswith(('.png', '.jpg', '.jpeg', '.webp')):
        orig_type = "image"
        used_model = "gemini"
        gemini_data, err = process_image_with_gemini(stored_path)
        if err or not gemini_data:
            abort(500, f"Gemini Image Processing Failed: {err}")
        orig_text = gemini_data.get("extracted_text", "")
        structured_data = gemini_data.get("summary_structure", {})
        
        # Defaults
        if "abstract" not in structured_data: structured_data["abstract"] = "Summary not generated."
        if "sections" not in structured_data: structured_data["sections"] = []

    # CASE 2: PDF/TXT -> IMPROVED ML
    else:
        used_model = "ml"
        with open(stored_path, "rb") as f_in:
            raw_bytes = f_in.read()
            
        if lower_name.endswith(".pdf"):
            orig_type = "pdf"
            orig_text = extract_text_from_pdf_bytes(raw_bytes)
        else:
            orig_type = "text"
            orig_text = raw_bytes.decode("utf-8", errors="ignore")
            
        if len(orig_text) < 50:
            abort(400, "Not enough text found.")
            
        length = request.form.get("length", "medium")
        tone = request.form.get("tone", "academic")
        
        sents, _ = summarize_extractive(orig_text, length)
        structured_data = build_structured_summary(sents, tone)

    # Generate PDF
    summary_filename = f"{uid}_summary.pdf"
    summary_path = os.path.join(app.config["SUMMARY_FOLDER"], summary_filename)
    save_summary_pdf(
        "Policy Summary",
        structured_data.get("abstract", ""),
        structured_data.get("sections", []),
        structured_data.get("simple_text", None),
        summary_path
    )
    
    return render_template_string(
        RESULT_HTML,
        title="Med.AI Summary",
        orig_type=orig_type,
        orig_url=url_for("uploaded_file", filename=stored_name),
        orig_text=orig_text[:20000], 
        doc_context=orig_text[:20000],
        abstract=structured_data.get("abstract", ""),
        sections=structured_data.get("sections", []),
        simple_text=structured_data.get("simple_text", None),
        summary_pdf_url=url_for("summary_file", filename=summary_filename),
        used_model=used_model
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
