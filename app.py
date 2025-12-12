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

# Language mapping for translation
LANGUAGE_MAP = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
    "ta": "Tamil",
    "bn": "Bengali",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi"
}

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
  <script src="https://cdn.jsdelivr.net/npm/three@0.132.2/build/three.min.js"></script>
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
            gradient: {
              start: '#0f766e',
              mid: '#0891b2',
              end: '#7c3aed'
            }
          },
          animation: {
            'float': 'float 6s ease-in-out infinite',
            'pulse-slow': 'pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
            'bounce-slow': 'bounce 3s infinite',
            'wave': 'wave 12s linear infinite',
          },
          keyframes: {
            float: {
              '0%, 100%': { transform: 'translateY(0)' },
              '50%': { transform: 'translateY(-15px)' },
            },
            wave: {
              '0%': { transform: 'translateX(0) translateZ(0) scaleY(1)' },
              '50%': { transform: 'translateX(-25%) translateZ(0) scaleY(0.85)' },
              '100%': { transform: 'translateX(-50%) translateZ(0) scaleY(1)' },
            }
          }
        }
      }
    }
  </script>
  <style>
    body { 
      background: linear-gradient(135deg, #f8fafc 0%, #f0fdfa 50%, #e0f2fe 100%);
      overflow-x: hidden;
    }
    .glass-panel {
      background: rgba(255, 255, 255, 0.85);
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      border: 1px solid rgba(255, 255, 255, 0.3);
      box-shadow: 0 20px 60px rgba(15, 118, 110, 0.08);
    }
    .gradient-text {
      background: linear-gradient(135deg, #0f766e 0%, #0891b2 30%, #7c3aed 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-size: 200% auto;
      animation: textShine 3s ease-in-out infinite alternate;
    }
    @keyframes textShine {
      to {
        background-position: 200% center;
      }
    }
    .floating-shape {
      position: absolute;
      border-radius: 50%;
      background: linear-gradient(135deg, rgba(14, 165, 233, 0.15), rgba(20, 184, 166, 0.1));
      filter: blur(40px);
      z-index: 0;
    }
    .custom-scrollbar::-webkit-scrollbar {
      width: 6px;
    }
    .custom-scrollbar::-webkit-scrollbar-track {
      background: #f1f1f1;
      border-radius: 10px;
    }
    .custom-scrollbar::-webkit-scrollbar-thumb {
      background: linear-gradient(to bottom, #0d9488, #0891b2);
      border-radius: 10px;
    }
    .section-glow {
      box-shadow: 0 0 40px rgba(20, 184, 166, 0.2);
    }
    .hover-glow:hover {
      box-shadow: 0 10px 40px rgba(20, 184, 166, 0.3);
    }
    .animate-3d-rotate {
      animation: rotate3d 20s infinite linear;
    }
    @keyframes rotate3d {
      0% { transform: perspective(1000px) rotateY(0deg) rotateX(5deg); }
      100% { transform: perspective(1000px) rotateY(360deg) rotateX(5deg); }
    }
  </style>
</head>
<body class="text-slate-800 relative overflow-x-hidden min-h-screen flex flex-col">

  <!-- Animated Background Shapes -->
  <div class="floating-shape w-96 h-96 -top-48 -left-48 animate-float"></div>
  <div class="floating-shape w-80 h-80 bottom-0 right-0 animate-float" style="animation-delay: 1s;"></div>
  <div class="floating-shape w-64 h-64 top-1/4 right-1/4 animate-float" style="animation-delay: 2s;"></div>

  <!-- Wave Animation Background -->
  <div class="absolute inset-0 overflow-hidden opacity-10">
    <div class="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgdmlld0JveD0iMCAwIDEwMCAxMDAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PGNpcmNsZSBjeD0iNTAiIGN5PSI1MCIgcj0iNDAiIHN0cm9rZT0iIzBkNjk0OCIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtZGFzaGFycmF5PSIxMCAxMCILz48L3N2Zz4=')] animate-wave"></div>
  </div>

  <nav class="fixed w-full z-40 glass-panel border-b border-white/30">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="flex justify-between h-20 items-center">
        <div class="flex items-center gap-3">
          <div class="relative">
            <div class="w-12 h-12 bg-gradient-to-tr from-teal-600 to-violet-600 rounded-2xl flex items-center justify-center shadow-xl shadow-teal-500/30 text-white animate-pulse-slow">
              <i class="fa-solid fa-staff-snake text-2xl"></i>
            </div>
            <div class="absolute -inset-1 bg-gradient-to-r from-teal-400 to-violet-400 rounded-2xl blur opacity-30 animate-pulse"></div>
          </div>
          <span class="font-black text-3xl tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-teal-700 to-violet-700">
            Med<span class="text-teal-600">.AI</span>
          </span>
        </div>
        <div class="hidden md:flex items-center gap-6">
          <a href="#workspace" class="group relative px-6 py-3 rounded-full bg-gradient-to-r from-teal-600 to-violet-600 text-white font-bold text-sm hover:shadow-2xl hover:shadow-teal-500/30 transition-all duration-300 transform hover:scale-105">
            <span class="relative z-10 flex items-center gap-2">
              <i class="fa-solid fa-bolt"></i> Start Analysis
            </span>
            <div class="absolute inset-0 rounded-full bg-gradient-to-r from-teal-400 to-violet-400 opacity-0 group-hover:opacity-100 blur transition-opacity duration-300"></div>
          </a>
        </div>
      </div>
    </div>
  </nav>

  <main class="flex-grow pt-40 pb-20 px-4 relative z-10">
    <div class="max-w-6xl mx-auto">
      
      <!-- Hero Section with 3D Elements -->
      <div class="text-center space-y-8 mb-20 relative">
        <div class="inline-flex items-center gap-2 px-5 py-2.5 rounded-full bg-gradient-to-r from-teal-100 to-violet-100 border border-teal-200 text-teal-800 text-sm font-bold uppercase tracking-wider mb-6 animate-bounce-slow">
          <span class="w-2 h-2 rounded-full bg-gradient-to-r from-teal-500 to-violet-500 animate-pulse"></span>
          AI-Powered Healthcare Intelligence
        </div>
        
        <div class="relative">
          <h1 class="text-6xl md:text-7xl font-black mb-6 leading-tight">
            <span class="gradient-text">Simplify</span> Complex<br>
            Medical Policies
          </h1>
          
          <!-- Animated 3D Element -->
          <div class="absolute -right-20 -top-10 w-40 h-40 opacity-20 animate-3d-rotate">
            <div class="w-full h-full relative">
              <div class="absolute inset-0 bg-gradient-to-r from-teal-400/20 to-violet-400/20 rounded-full blur-xl"></div>
              <div class="absolute inset-4 bg-gradient-to-r from-teal-300/10 to-violet-300/10 rounded-full animate-ping"></div>
            </div>
          </div>
        </div>
        
        <p class="text-xl text-slate-600 max-w-3xl mx-auto leading-relaxed font-medium">
          Upload PDF, Text, or <span class="font-bold text-slate-800 bg-gradient-to-r from-teal-500 to-violet-500 bg-clip-text text-transparent">Use Your Camera</span>. 
          Advanced ML algorithms extract actionable insights with precision.
        </p>
        
        <div class="flex flex-wrap gap-4 justify-center mt-10">
          <div class="flex items-center gap-2 px-4 py-2 bg-white/50 rounded-full border border-slate-200">
            <div class="w-2 h-2 rounded-full bg-teal-500 animate-pulse"></div>
            <span class="text-sm font-medium">TF-IDF + TextRank</span>
          </div>
          <div class="flex items-center gap-2 px-4 py-2 bg-white/50 rounded-full border border-slate-200">
            <div class="w-2 h-2 rounded-full bg-violet-500 animate-pulse"></div>
            <span class="text-sm font-medium">Gemini AI Integration</span>
          </div>
          <div class="flex items-center gap-2 px-4 py-2 bg-white/50 rounded-full border border-slate-200">
            <div class="w-2 h-2 rounded-full bg-cyan-500 animate-pulse"></div>
            <span class="text-sm font-medium">Multi-Language Support</span>
          </div>
        </div>
      </div>

      <!-- Workspace Card -->
      <div id="workspace" class="glass-panel rounded-3xl p-1.5 shadow-2xl shadow-teal-500/10 max-w-4xl mx-auto section-glow">
        <div class="bg-gradient-to-br from-white/60 to-white/40 rounded-2xl p-8 md:p-10 border border-white/50">
          
          <form id="uploadForm" action="{{ url_for('summarize') }}" method="post" enctype="multipart/form-data" class="space-y-10">
            
            <!-- Upload Area -->
            <div class="group relative w-full h-72 border-3 border-dashed border-slate-300 rounded-2xl bg-gradient-to-br from-white to-teal-50/30 hover:from-white hover:to-teal-50/50 hover:border-teal-400 transition-all duration-500 flex flex-col items-center justify-center cursor-pointer overflow-hidden" id="drop-zone">
              
              <input id="file-input" type="file" name="file" accept=".pdf,.txt,image/*" class="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-20">
              
              <div id="upload-prompt" class="text-center space-y-6 transition-all duration-500 group-hover:scale-105">
                <div class="relative">
                  <div class="w-20 h-20 bg-gradient-to-br from-white to-teal-50 rounded-2xl shadow-lg flex items-center justify-center mx-auto text-teal-500 text-3xl group-hover:text-teal-600">
                    <i class="fa-solid fa-cloud-arrow-up"></i>
                  </div>
                  <div class="absolute -inset-1 bg-gradient-to-r from-teal-400 to-cyan-400 rounded-2xl blur opacity-30 group-hover:opacity-50 transition-opacity"></div>
                </div>
                <div class="space-y-2">
                  <p class="text-2xl font-bold text-slate-800">Drop Your Document Here</p>
                  <p class="text-slate-500">PDF, TXT, or Images • Up to 50MB</p>
                </div>
                <div class="flex flex-wrap gap-3 justify-center">
                  <div class="inline-flex items-center gap-2 px-4 py-2 bg-white rounded-full shadow-sm text-sm font-semibold text-slate-700 border border-slate-200">
                    <i class="fa-solid fa-camera text-teal-600"></i> Camera Upload
                  </div>
                  <div class="inline-flex items-center gap-2 px-4 py-2 bg-white rounded-full shadow-sm text-sm font-semibold text-slate-700 border border-slate-200">
                    <i class="fa-solid fa-file-pdf text-red-500"></i> PDF Supported
                  </div>
                </div>
              </div>

              <div id="file-preview" class="hidden absolute inset-0 bg-white/95 backdrop-blur-sm z-10 flex flex-col items-center justify-center p-8 text-center animate-fade-in">
                 <div id="preview-icon" class="mb-6 text-5xl"></div>
                 <div id="preview-image-container" class="mb-6 hidden rounded-xl overflow-hidden shadow-2xl border-2 border-white max-h-40">
                    <img id="preview-image" src="" alt="Preview" class="h-full object-contain">
                 </div>
                 <p id="filename-display" class="font-bold text-slate-800 text-xl break-all max-w-md"></p>
                 <p class="text-sm text-teal-600 font-semibold mt-3 uppercase tracking-wider">Ready for AI Analysis</p>
                 <button type="button" id="change-file-btn" class="mt-6 px-4 py-2 text-sm text-slate-500 hover:text-slate-700 hover:bg-slate-100 rounded-full transition">Change Document</button>
              </div>

            </div>

            <!-- Settings Grid -->
            <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
              <!-- Summary Length -->
              <div class="bg-gradient-to-br from-white to-teal-50/30 rounded-2xl p-5 border border-slate-200 hover-glow transition-all duration-300">
                <label class="block text-xs font-bold text-slate-600 uppercase tracking-widest mb-4 flex items-center gap-2">
                  <i class="fa-solid fa-ruler text-teal-600"></i> Summary Length
                </label>
                <div class="flex flex-col space-y-3">
                  <label class="flex items-center gap-3 cursor-pointer group">
                    <input type="radio" name="length" value="short" class="peer hidden">
                    <div class="w-5 h-5 rounded-full border-2 border-slate-300 group-hover:border-teal-400 flex items-center justify-center peer-checked:border-teal-600">
                      <div class="w-2.5 h-2.5 rounded-full bg-teal-600 hidden peer-checked:block"></div>
                    </div>
                    <span class="flex-1 text-sm font-medium text-slate-700 group-hover:text-slate-900">Short (3-5 points)</span>
                    <div class="w-8 h-1.5 bg-gradient-to-r from-teal-200 to-teal-300 rounded-full"></div>
                  </label>
                  <label class="flex items-center gap-3 cursor-pointer group">
                    <input type="radio" name="length" value="medium" checked class="peer hidden">
                    <div class="w-5 h-5 rounded-full border-2 border-slate-300 group-hover:border-teal-400 flex items-center justify-center peer-checked:border-teal-600">
                      <div class="w-2.5 h-2.5 rounded-full bg-teal-600 hidden peer-checked:block"></div>
                    </div>
                    <span class="flex-1 text-sm font-medium text-slate-700 group-hover:text-slate-900">Medium (7-10 points)</span>
                    <div class="w-16 h-1.5 bg-gradient-to-r from-teal-300 to-teal-400 rounded-full"></div>
                  </label>
                  <label class="flex items-center gap-3 cursor-pointer group">
                    <input type="radio" name="length" value="long" class="peer hidden">
                    <div class="w-5 h-5 rounded-full border-2 border-slate-300 group-hover:border-teal-400 flex items-center justify-center peer-checked:border-teal-600">
                      <div class="w-2.5 h-2.5 rounded-full bg-teal-600 hidden peer-checked:block"></div>
                    </div>
                    <span class="flex-1 text-sm font-medium text-slate-700 group-hover:text-slate-900">Long (12-15 points)</span>
                    <div class="w-24 h-1.5 bg-gradient-to-r from-teal-400 to-teal-500 rounded-full"></div>
                  </label>
                </div>
              </div>

              <!-- Tone Selection -->
              <div class="bg-gradient-to-br from-white to-violet-50/30 rounded-2xl p-5 border border-slate-200 hover-glow transition-all duration-300">
                <label class="block text-xs font-bold text-slate-600 uppercase tracking-widest mb-4 flex items-center gap-2">
                  <i class="fa-solid fa-comment-dots text-violet-600"></i> Output Tone
                </label>
                <div class="grid grid-cols-2 gap-3">
                  <label class="cursor-pointer">
                    <input type="radio" name="tone" value="academic" checked class="peer hidden">
                    <div class="h-full p-4 rounded-xl border-2 border-slate-200 peer-checked:border-teal-500 peer-checked:bg-teal-50/50 transition-all duration-300 hover:border-teal-300 hover:bg-teal-50/30">
                      <div class="text-center">
                        <i class="fa-solid fa-graduation-cap text-xl text-teal-600 mb-2"></i>
                        <p class="text-sm font-semibold text-slate-800">Academic</p>
                        <p class="text-xs text-slate-500 mt-1">Structured Analysis</p>
                      </div>
                    </div>
                  </label>
                  <label class="cursor-pointer">
                    <input type="radio" name="tone" value="simple" class="peer hidden">
                    <div class="h-full p-4 rounded-xl border-2 border-slate-200 peer-checked:border-violet-500 peer-checked:bg-violet-50/50 transition-all duration-300 hover:border-violet-300 hover:bg-violet-50/30">
                      <div class="text-center">
                        <i class="fa-solid fa-sparkles text-xl text-violet-600 mb-2"></i>
                        <p class="text-sm font-semibold text-slate-800">Simple</p>
                        <p class="text-xs text-slate-500 mt-1">Clean Paragraph</p>
                      </div>
                    </div>
                  </label>
                </div>
              </div>

              <!-- Language Selection -->
              <div class="bg-gradient-to-br from-white to-cyan-50/30 rounded-2xl p-5 border border-slate-200 hover-glow transition-all duration-300">
                <label class="block text-xs font-bold text-slate-600 uppercase tracking-widest mb-4 flex items-center gap-2">
                  <i class="fa-solid fa-language text-cyan-600"></i> Output Language
                </label>
                <select name="language" class="w-full px-4 py-3 rounded-xl bg-white/50 border border-slate-200 text-sm text-slate-700 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-transparent transition">
                  <option value="en">English</option>
                  <option value="hi">Hindi (हिन्दी)</option>
                  <option value="te">Telugu (తెలుగు)</option>
                  <option value="ta">Tamil (தமிழ்)</option>
                  <option value="bn">Bengali (বাংলা)</option>
                  <option value="mr">Marathi (मराठी)</option>
                  <option value="gu">Gujarati (ગુજરાતી)</option>
                </select>
              </div>
            </div>

            <!-- Submit Button -->
            <button type="submit" class="group relative w-full py-5 rounded-2xl bg-gradient-to-r from-teal-600 via-violet-600 to-cyan-700 text-white font-bold text-lg shadow-xl shadow-teal-500/30 hover:shadow-2xl hover:shadow-violet-500/30 hover:scale-[1.02] transition-all duration-300 flex items-center justify-center gap-3 overflow-hidden">
              <div class="absolute inset-0 bg-gradient-to-r from-teal-400 via-violet-400 to-cyan-500 opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
              <span class="relative z-10 flex items-center gap-3">
                <i class="fa-solid fa-brain text-xl"></i>
                Generate Intelligent Summary
              </span>
              <i class="fa-solid fa-arrow-right relative z-10 animate-pulse"></i>
            </button>

          </form>
        </div>
      </div>

    </div>
  </main>

  <div id="progress-overlay" class="fixed inset-0 bg-gradient-to-br from-white via-teal-50/50 to-white z-50 hidden flex-col items-center justify-center backdrop-blur-xl">
    <div class="w-full max-w-md px-6 text-center space-y-8">
      
      <div class="relative">
        <div class="w-32 h-32 mx-auto">
          <div class="absolute inset-0 rounded-full border-[6px] border-white shadow-xl"></div>
          <div class="absolute inset-6 rounded-full border-[6px] border-teal-500 border-t-transparent animate-spin"></div>
          <div class="absolute inset-12 rounded-full border-[6px] border-cyan-500 border-t-transparent animate-spin" style="animation-direction: reverse;"></div>
          <div class="absolute inset-0 flex items-center justify-center">
            <div class="text-center">
              <div id="progress-text" class="text-3xl font-black bg-gradient-to-r from-teal-600 to-cyan-600 bg-clip-text text-transparent">0%</div>
              <div class="text-xs text-slate-500 mt-1">Processing</div>
            </div>
          </div>
        </div>
        
        <!-- Floating icons around progress -->
        <div class="absolute -top-2 -left-2 w-8 h-8 bg-white rounded-full shadow-lg flex items-center justify-center text-teal-600 animate-bounce">
          <i class="fa-solid fa-file"></i>
        </div>
        <div class="absolute -top-2 -right-2 w-8 h-8 bg-white rounded-full shadow-lg flex items-center justify-center text-violet-600 animate-bounce" style="animation-delay: 0.2s;">
          <i class="fa-solid fa-robot"></i>
        </div>
        <div class="absolute -bottom-2 left-4 w-8 h-8 bg-white rounded-full shadow-lg flex items-center justify-center text-cyan-600 animate-bounce" style="animation-delay: 0.4s;">
          <i class="fa-solid fa-brain"></i>
        </div>
      </div>

      <div class="space-y-4">
        <h3 id="progress-stage" class="text-2xl font-bold text-slate-900">Starting Analysis...</h3>
        <p class="text-slate-500">Extracting key insights from your document</p>
      </div>

      <div class="w-full h-4 bg-gradient-to-r from-white to-slate-100 rounded-full overflow-hidden relative shadow-inner">
        <div id="progress-bar" class="h-full bg-gradient-to-r from-teal-400 via-violet-400 to-cyan-500 w-0 transition-all duration-500 ease-out relative">
          <div class="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent animate-shimmer"></div>
        </div>
      </div>

      <div class="pt-4 border-t border-slate-200">
        <div class="flex justify-between text-xs text-slate-500">
          <span>Document Processing</span>
          <span>AI Analysis</span>
          <span>Structuring Output</span>
        </div>
      </div>
    </div>
  </div>

  <script>
    // Enhanced animations
    document.addEventListener('DOMContentLoaded', function() {
      // Create floating particles
      for(let i = 0; i < 15; i++) {
        createParticle();
      }
    });

    function createParticle() {
      const particle = document.createElement('div');
      particle.className = 'floating-shape';
      particle.style.width = Math.random() * 40 + 20 + 'px';
      particle.style.height = particle.style.width;
      particle.style.left = Math.random() * 100 + 'vw';
      particle.style.top = Math.random() * 100 + 'vh';
      particle.style.animationDelay = Math.random() * 5 + 's';
      particle.style.background = `linear-gradient(135deg, 
        rgba(${Math.random() * 100 + 155}, ${Math.random() * 100 + 200}, ${Math.random() * 100 + 200}, 0.1),
        rgba(${Math.random() * 100 + 100}, ${Math.random() * 100 + 100}, ${Math.random() * 100 + 255}, 0.05)
      )`;
      document.body.appendChild(particle);
    }

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

    // File Upload Preview Logic
    fileInput.addEventListener('change', function(e) {
      if (this.files && this.files[0]) {
        const file = this.files[0];
        const reader = new FileReader();

        uploadPrompt.classList.add('opacity-0', 'scale-95');
        setTimeout(() => {
            uploadPrompt.classList.add('hidden');
            filePreview.classList.remove('hidden');
            filePreview.classList.add('flex');
        }, 300);
        
        filenameDisplay.textContent = file.name;

        // Reset styling
        previewImgContainer.classList.add('hidden');
        previewIcon.innerHTML = '';

        if (file.type.startsWith('image/')) {
           reader.onload = function(e) {
             previewImg.src = e.target.result;
             previewImgContainer.classList.remove('hidden');
             previewIcon.innerHTML = '<i class="fa-solid fa-image text-emerald-500"></i>';
           }
           reader.readAsDataURL(file);
        } else if (file.type === 'application/pdf') {
           previewIcon.innerHTML = '<i class="fa-solid fa-file-pdf text-red-500"></i>';
        } else {
           previewIcon.innerHTML = '<i class="fa-solid fa-file-lines text-teal-500"></i>';
        }
      }
    });

    // Change file button
    changeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.value = '';
        filePreview.classList.add('hidden');
        filePreview.classList.remove('flex');
        uploadPrompt.classList.remove('hidden', 'opacity-0', 'scale-95');
    });

    // Progress bar simulation
    uploadForm.addEventListener('submit', function(e) {
        if (!fileInput.files.length) {
            e.preventDefault();
            alert("Please select a document first.");
            return;
        }

        progressOverlay.classList.remove('hidden');
        progressOverlay.classList.add('flex');
        
        let width = 0;
        const fileType = fileInput.files[0].type;
        const isImage = fileType.startsWith('image/');
        
        const totalDuration = isImage ? 8000 : 12000; 
        const intervalTime = 100;
        const step = 100 / (totalDuration / intervalTime);

        const interval = setInterval(() => {
            if (width >= 100) {
                clearInterval(interval);
                progressStage.textContent = "Generating Final Summary...";
                progressText.textContent = "100%";
            } else {
                width += step;
                width = Math.min(width, 99);
                
                progressBar.style.width = width + '%';
                progressText.textContent = Math.round(width) + '%';

                if (width < 25) {
                    progressStage.textContent = "Uploading & Parsing Document...";
                } else if (width < 50) {
                    progressStage.textContent = "Running ML Algorithms (TF-IDF)...";
                } else if (width < 75) {
                    progressStage.textContent = "Applying TextRank Analysis...";
                } else if (width < 90) {
                    progressStage.textContent = "Structuring Policy Insights...";
                } else {
                    progressStage.textContent = "Finalizing Summary...";
                }
            }
        }, intervalTime);
    });

    // Drag and drop support
    const dropZone = document.getElementById('drop-zone');
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    function highlight() {
        dropZone.classList.add('border-teal-500', 'bg-teal-50/50');
    }

    function unhighlight() {
        dropZone.classList.remove('border-teal-500', 'bg-teal-50/50');
    }

    dropZone.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        fileInput.files = files;
        fileInput.dispatchEvent(new Event('change'));
    }
  </script>

  <style>
    @keyframes shimmer {
      0% { transform: translateX(-100%); }
      100% { transform: translateX(100%); }
    }
    .animate-shimmer {
      animation: shimmer 2s infinite;
    }
  </style>
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
            violet: { 50: '#f5f3ff', 100: '#ede9fe', 200: '#ddd6fe', 300: '#c4b5fd', 400: '#a78bfa', 500: '#8b5cf6', 600: '#7c3aed', 700: '#6d28d9', 800: '#5b21b6', 900: '#4c1d95' }
          },
        }
      }
    }
  </script>
  <style>
    body {
      background: linear-gradient(135deg, #f8fafc 0%, #f0fdfa 30%, #f5f3ff 100%);
    }
    .glass-card {
      background: rgba(255, 255, 255, 0.85);
      backdrop-filter: blur(20px);
      border: 1px solid rgba(255, 255, 255, 0.3);
      box-shadow: 0 20px 60px rgba(15, 118, 110, 0.08);
    }
    .gradient-border {
      position: relative;
      background: linear-gradient(white, white) padding-box,
                  linear-gradient(135deg, #0d9488, #7c3aed, #0891b2) border-box;
      border: 2px solid transparent;
    }
    .language-option.active {
      background: linear-gradient(135deg, #0d9488, #7c3aed);
      color: white;
      box-shadow: 0 4px 20px rgba(124, 58, 237, 0.3);
    }
    .fade-in {
      animation: fadeIn 0.5s ease-in;
    }
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(10px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .scrollbar-hide::-webkit-scrollbar {
      display: none;
    }
  </style>
</head>
<body class="text-slate-800">

  <nav class="fixed w-full z-40 glass-card border-b border-white/30">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="flex justify-between h-20 items-center">
        <div class="flex items-center gap-3">
          <div class="w-10 h-10 bg-gradient-to-tr from-teal-600 to-violet-600 rounded-xl flex items-center justify-center shadow-lg shadow-teal-500/20 text-white">
            <i class="fa-solid fa-staff-snake"></i>
          </div>
          <span class="font-black text-2xl tracking-tight bg-gradient-to-r from-teal-700 to-violet-700 bg-clip-text text-transparent">
            Med<span class="text-teal-600">.AI</span>
          </span>
        </div>
        <div class="flex items-center gap-4">
          <div id="language-selector" class="flex items-center gap-2 px-3 py-2 bg-white/50 rounded-lg border border-slate-200">
            <i class="fa-solid fa-language text-teal-600"></i>
            <select id="language-select" class="bg-transparent text-sm font-medium text-slate-700 focus:outline-none">
              <option value="en">English</option>
              <option value="hi">Hindi</option>
              <option value="te">Telugu</option>
              <option value="ta">Tamil</option>
              <option value="bn">Bengali</option>
              <option value="mr">Marathi</option>
              <option value="gu">Gujarati</option>
            </select>
          </div>
          <a href="{{ url_for('index') }}" class="px-5 py-2.5 rounded-xl bg-gradient-to-r from-teal-600 to-violet-600 text-white text-sm font-bold hover:shadow-lg hover:shadow-teal-500/30 transition-all transform hover:scale-105">
            <i class="fa-solid fa-plus mr-2"></i> New Analysis
          </a>
        </div>
      </div>
    </div>
  </nav>

  <main class="pt-24 pb-12 px-4">
    <div class="max-w-7xl mx-auto grid lg:grid-cols-12 gap-8">
      
      <!-- Summary Section -->
      <section class="lg:col-span-8 space-y-6">
        <div class="glass-card rounded-3xl p-8 fade-in">
          
          <!-- Header -->
          <div class="flex flex-wrap items-start justify-between gap-4 mb-8 pb-6 border-b border-slate-100">
            <div class="space-y-2">
              <div class="flex flex-wrap items-center gap-2">
                <span class="px-3 py-1.5 rounded-lg bg-teal-50 text-teal-700 text-xs font-bold uppercase tracking-wide border border-teal-100">
                  {{ orig_type | upper }}
                </span>
                {% if used_model == 'gemini' %}
                <span class="px-3 py-1.5 rounded-lg bg-violet-50 text-violet-700 text-xs font-bold uppercase tracking-wide border border-violet-100">
                  <i class="fa-solid fa-sparkles mr-1"></i> Gemini AI
                </span>
                {% else %}
                <span class="px-3 py-1.5 rounded-lg bg-gradient-to-r from-teal-50 to-cyan-50 text-teal-700 text-xs font-bold uppercase tracking-wide border border-teal-100">
                  <i class="fa-solid fa-brain mr-1"></i> ML Enhanced
                </span>
                {% endif %}
                <span class="px-3 py-1.5 rounded-lg bg-amber-50 text-amber-700 text-xs font-bold uppercase tracking-wide border border-amber-100">
                  {{ tone | capitalize }}
                </span>
              </div>
              <h1 class="text-3xl font-black text-slate-900 leading-tight">Policy Analysis Summary</h1>
              <p class="text-sm text-slate-500">Generated with {{ "Gemini AI" if used_model == 'gemini' else "TF-IDF + TextRank" }} • Language: <span id="current-language" class="font-semibold text-teal-600">English</span></p>
            </div>
            
            {% if summary_pdf_url %}
            <a href="{{ summary_pdf_url }}" class="group relative px-5 py-3 rounded-xl bg-gradient-to-r from-slate-900 to-slate-800 text-white text-sm font-bold hover:shadow-xl hover:shadow-slate-900/30 transition-all">
              <span class="relative z-10 flex items-center gap-2">
                <i class="fa-solid fa-file-arrow-down"></i> Export PDF
              </span>
              <div class="absolute inset-0 rounded-xl bg-gradient-to-r from-teal-600 to-violet-600 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
            </a>
            {% endif %}
          </div>

          <!-- Abstract -->
          <div class="mb-8">
            <h2 class="text-sm font-bold text-slate-400 uppercase tracking-widest mb-4 flex items-center gap-2">
              <i class="fa-solid fa-sparkles text-teal-500"></i> Executive Summary
            </h2>
            <div class="gradient-border rounded-2xl p-6 bg-gradient-to-br from-white to-teal-50/30">
              <div id="abstract-content" class="text-slate-700 leading-relaxed">
                {{ abstract }}
              </div>
            </div>
          </div>

          <!-- Sections (Only for Academic tone) -->
          {% if tone == 'academic' and sections %}
          <div id="sections-container" class="space-y-6">
            {% for sec in sections %}
            <div class="fade-in" style="animation-delay: {{ loop.index * 0.1 }}s;">
              <div class="flex items-center gap-3 mb-4">
                <div class="w-2 h-8 rounded-full bg-gradient-to-b from-teal-500 to-cyan-500"></div>
                <h3 class="text-lg font-bold text-slate-800">{{ sec.title }}</h3>
              </div>
              <div class="pl-5">
                <ul class="space-y-3">
                  {% for bullet in sec.bullets %}
                  <li class="flex items-start gap-3 text-slate-600 group">
                    <i class="fa-solid fa-check mt-1 text-teal-500 group-hover:text-teal-600 transition-colors"></i>
                    <span class="flex-1">{{ bullet }}</span>
                  </li>
                  {% endfor %}
                </ul>
              </div>
            </div>
            {% endfor %}
          </div>
          {% elif tone == 'simple' %}
          <!-- Simple tone shows single paragraph -->
          <div class="mt-8 pt-6 border-t border-slate-100">
            <div class="gradient-border rounded-2xl p-6 bg-gradient-to-br from-white to-violet-50/30">
              <div id="simple-summary" class="text-slate-700 leading-relaxed text-justify">
                {{ abstract }}
              </div>
            </div>
          </div>
          {% endif %}

        </div>
      </section>

      <!-- Sidebar -->
      <section class="lg:col-span-4 space-y-6">
        
        <!-- Original Document -->
        <div class="glass-card rounded-3xl p-6">
          <div class="flex items-center justify-between mb-4">
            <h2 class="text-sm font-bold text-slate-400 uppercase tracking-widest">Original Document</h2>
            <span class="px-2 py-1 text-xs font-bold bg-slate-100 text-slate-600 rounded">{{ orig_type | upper }}</span>
          </div>
          <div class="rounded-xl overflow-hidden border border-slate-200 bg-gradient-to-br from-slate-50 to-white h-64 relative group">
            {% if orig_type == 'pdf' %}
              <iframe src="{{ orig_url }}" class="w-full h-full" title="Original PDF"></iframe>
              <div class="absolute inset-0 bg-gradient-to-t from-black/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>
            {% elif orig_type == 'text' %}
              <div class="p-4 overflow-y-auto h-full text-xs font-mono bg-slate-900 text-slate-200 rounded-xl">{{ orig_text | truncate(2000) }}</div>
            {% elif orig_type == 'image' %}
              <img src="{{ orig_url }}" class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500">
            {% endif %}
          </div>
        </div>

        <!-- Chat with AI -->
        <div class="glass-card rounded-3xl p-6 flex flex-col h-[500px]">
          <div class="mb-4">
            <div class="flex items-center gap-3 mb-2">
              <div class="w-10 h-10 rounded-xl bg-gradient-to-br from-teal-500 to-violet-500 flex items-center justify-center text-white">
                <i class="fa-solid fa-robot"></i>
              </div>
              <div>
                <h2 class="font-bold text-slate-800">Ask AI Assistant</h2>
                <p class="text-xs text-slate-400">Ask questions based on document content</p>
              </div>
            </div>
          </div>
          
          <!-- Chat Messages -->
          <div id="chat-panel" class="flex-1 overflow-y-auto space-y-4 mb-4 pr-2 custom-scrollbar scrollbar-hide">
            <div class="flex gap-3">
              <div class="w-8 h-8 rounded-full bg-gradient-to-br from-teal-100 to-teal-50 flex items-center justify-center text-teal-600 text-sm shrink-0">
                <i class="fa-solid fa-robot"></i>
              </div>
              <div class="max-w-[85%] bg-gradient-to-br from-slate-50 to-white rounded-2xl rounded-tl-none p-4 text-sm text-slate-700 leading-relaxed shadow-sm">
                I've analyzed this policy document. Ask me about specific goals, strategies, or implementation details.
              </div>
            </div>
          </div>

          <!-- Chat Input -->
          <div class="relative mt-auto">
            <div class="absolute left-4 top-3.5 text-slate-400">
              <i class="fa-solid fa-message"></i>
            </div>
            <input type="text" id="chat-input" class="w-full pl-12 pr-12 py-4 rounded-xl bg-gradient-to-br from-slate-50 to-white border border-slate-200 text-sm focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-transparent transition placeholder-slate-400" placeholder="Ask about the policy...">
            <button id="chat-send" class="absolute right-2 top-2 p-2.5 bg-gradient-to-r from-teal-500 to-teal-600 text-white rounded-lg hover:from-teal-600 hover:to-teal-700 transition-all transform hover:scale-105 active:scale-95">
              <i class="fa-solid fa-paper-plane text-sm"></i>
            </button>
          </div>
          <textarea id="doc-context" class="hidden">{{ doc_context }}</textarea>
        </div>

      </section>
    </div>
  </main>

  <script>
    let currentLanguage = 'en';
    let originalSummary = {
      abstract: `{{ abstract | safe }}`,
      sections: {{ sections | tojson | safe }},
      tone: '{{ tone }}'
    };

    // Language selection
    const languageSelect = document.getElementById('language-select');
    const currentLanguageSpan = document.getElementById('current-language');
    
    // Update current language display
    const languageNames = {
      en: 'English',
      hi: 'Hindi',
      te: 'Telugu',
      ta: 'Tamil',
      bn: 'Bengali',
      mr: 'Marathi',
      gu: 'Gujarati'
    };

    languageSelect.value = '{{ language }}';
    currentLanguage = '{{ language }}';
    currentLanguageSpan.textContent = languageNames[currentLanguage];

    languageSelect.addEventListener('change', async function() {
      const newLang = this.value;
      if (newLang === currentLanguage) return;
      
      // Show loading
      currentLanguageSpan.innerHTML = `<i class="fa-solid fa-spinner animate-spin"></i> Translating...`;
      
      try {
        const response = await fetch('/translate_summary', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            text: originalSummary.abstract,
            sections: originalSummary.sections,
            tone: originalSummary.tone,
            target_lang: newLang
          })
        });
        
        const data = await response.json();
        
        if (data.success) {
          // Update abstract
          document.getElementById('abstract-content').textContent = data.translated.abstract;
          
          // For simple tone, update the simple summary
          if (originalSummary.tone === 'simple') {
            document.getElementById('simple-summary').textContent = data.translated.abstract;
          }
          
          // For academic tone, update sections
          if (originalSummary.tone === 'academic' && data.translated.sections) {
            const sectionsContainer = document.getElementById('sections-container');
            sectionsContainer.innerHTML = '';
            
            data.translated.sections.forEach((section, index) => {
              const sectionHTML = `
                <div class="fade-in" style="animation-delay: ${index * 0.1}s">
                  <div class="flex items-center gap-3 mb-4">
                    <div class="w-2 h-8 rounded-full bg-gradient-to-b from-teal-500 to-cyan-500"></div>
                    <h3 class="text-lg font-bold text-slate-800">${section.title}</h3>
                  </div>
                  <div class="pl-5">
                    <ul class="space-y-3">
                      ${section.bullets.map(bullet => `
                        <li class="flex items-start gap-3 text-slate-600 group">
                          <i class="fa-solid fa-check mt-1 text-teal-500 group-hover:text-teal-600 transition-colors"></i>
                          <span class="flex-1">${bullet}</span>
                        </li>
                      `).join('')}
                    </ul>
                  </div>
                </div>
              `;
              sectionsContainer.innerHTML += sectionHTML;
            });
          }
          
          currentLanguage = newLang;
          currentLanguageSpan.textContent = languageNames[newLang];
        }
      } catch (error) {
        console.error('Translation error:', error);
        currentLanguageSpan.textContent = languageNames[currentLanguage];
        alert('Translation failed. Please try again.');
      }
    });

    // Chat functionality
    const panel = document.getElementById('chat-panel');
    const input = document.getElementById('chat-input');
    const sendBtn = document.getElementById('chat-send');
    const docText = document.getElementById('doc-context').value;

    function addMsg(role, text) {
      const div = document.createElement('div');
      div.className = role === 'user' ? 'flex gap-3 flex-row-reverse' : 'flex gap-3';
      
      const avatar = document.createElement('div');
      avatar.className = role === 'user' 
        ? 'w-8 h-8 rounded-full bg-gradient-to-br from-slate-800 to-slate-900 flex items-center justify-center text-white text-sm shrink-0'
        : 'w-8 h-8 rounded-full bg-gradient-to-br from-teal-100 to-teal-50 flex items-center justify-center text-teal-600 text-sm shrink-0';
      avatar.innerHTML = role === 'user' ? '<i class="fa-solid fa-user"></i>' : '<i class="fa-solid fa-robot"></i>';
      
      const bubble = document.createElement('div');
      bubble.className = role === 'user'
        ? 'max-w-[85%] bg-gradient-to-br from-teal-500 to-teal-600 text-white rounded-2xl rounded-tr-none p-4 text-sm leading-relaxed shadow-sm'
        : 'max-w-[85%] bg-gradient-to-br from-slate-50 to-white rounded-2xl rounded-tl-none p-4 text-sm text-slate-700 leading-relaxed shadow-sm';
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
          body: JSON.stringify({ 
            message: txt, 
            doc_text: docText,
            current_lang: currentLanguage
          })
        });
        const data = await res.json();
        addMsg('assistant', data.reply);
      } catch(e) {
        addMsg('assistant', "Sorry, I encountered an error. Please try again.");
      }
    }

    sendBtn.onclick = sendMessage;
    input.onkeypress = (e) => { if(e.key === 'Enter') sendMessage(); }

    // Auto-focus chat input
    setTimeout(() => {
      input.focus();
    }, 1000);
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
        "vs.": "vs<DOT>", "e.g.": "e<DOT>g<DOT>", "i.e.": "i<DOT>e<DOT>",
        "et al.": "et_al<DOT>", "i.e.": "i<DOT>e<DOT>", "e.g.": "e<DOT>g<DOT>"
    }
    
    for abb, mask in abbreviations.items():
        text = text.replace(abb, mask)

    parts = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+(?=[A-Z"\'"("])', text)
    
    sentences = []
    for p in parts:
        for abb, mask in abbreviations.items():
            p = p.replace(mask, abb)
            
        p = p.strip()
        if len(p) < 20: 
            continue
        if re.match(r'^[0-9\.\s]+$', p): 
            continue
        
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

# ---------------------- IMPROVED ML SUMMARIZER ---------------------- #

POLICY_KEYWORDS = {
    "key goals": [
        "aim", "goal", "objective", "target", "achieve", "reduce", "increase", 
        "coverage", "mortality", "rate", "%", "2025", "2030", "vision", "mission", 
        "outcome", "expectancy", "eliminate", "improve", "enhance"
    ],
    "policy principles": [
        "principle", "equity", "universal", "right", "access", "accountability", 
        "transparency", "inclusive", "patient-centered", "quality", "ethics", 
        "value", "integrity", "holistic", "sustainable", "equitable"
    ],
    "service delivery": [
        "hospital", "primary care", "secondary care", "tertiary", "referral", 
        "clinic", "health center", "wellness", "ambulance", "emergency", "drug", 
        "diagnostic", "infrastructure", "bed", "supply chain", "logistics", "facility"
    ],
    "prevention & promotion": [
        "prevent", "sanitation", "nutrition", "immunization", "vaccine", "tobacco", 
        "alcohol", "hygiene", "awareness", "lifestyle", "pollution", "water", 
        "screening", "diet", "exercise", "community", "campaign", "education"
    ],
    "human resources": [
        "doctor", "nurse", "staff", "training", "workforce", "recruit", "medical college", 
        "paramedic", "salary", "incentive", "capacity building", "skill", "hrh", 
        "deployment", "specialist", "professionals", "expertise"
    ],
    "financing & costs": [
        "fund", "budget", "finance", "expenditure", "cost", "insurance", "private", 
        "partnership", "ppp", "out-of-pocket", "reimbursement", "allocation", 
        "spending", "gdp", "tax", "claim", "revenue", "investment"
    ],
    "digital health": [
        "digital", "technology", "data", "record", "ehr", "emr", "telemedicine", 
        "mobile", "app", "information system", "cyber", "interoperability", 
        "portal", "online", "software", "ai", "electronic", "monitoring"
    ]
}

def score_sentence_categories(sentence: str) -> str:
    s_lower = sentence.lower()
    scores = {cat: 0 for cat in POLICY_KEYWORDS}
    
    words = re.findall(r'\w+', s_lower)
    
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
        ngram_range=(1, 3),
        sublinear_tf=True,
        max_features=5000
    ).fit_transform(sentences)

def textrank_scores(sim_mat: np.ndarray, doc_len: int) -> Dict[int, float]:
    np.fill_diagonal(sim_mat, 0.0)
    G = nx.from_numpy_array(sim_mat)
    try:
        pr = nx.pagerank(G, alpha=0.85, max_iter=200, tol=1e-6)
    except:
        pr = {i: 1.0/doc_len for i in range(sim_mat.shape[0])}
    
    scores = {}
    for i in range(sim_mat.shape[0]):
        base_score = pr.get(i, 0.0)
        
        # Enhanced position bias
        position = i / doc_len
        if position < 0.05:  # Introduction
            mult = 1.5
        elif position > 0.95:  # Conclusion
            mult = 1.3
        elif position < 0.15:  # Early important
            mult = 1.2
        else:
            mult = 1.0
            
        scores[i] = base_score * mult
        
    return scores

def mmr(scores_dict: Dict[int, float], sim_mat: np.ndarray, k: int, lambda_param: float = 0.7) -> List[int]:
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
            sim_to_selected = 0.0
            if selected:
                sim_to_selected = max([sim_mat[i][j] for j in selected])
            
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
    cleaned = normalize_whitespace(raw_text)
    sentences = sentence_split(cleaned)
    n = len(sentences)
    
    if n <= 3: 
        return sentences, {}
    
    # SIGNIFICANTLY DIFFERENT RATIOS FOR CLEAR DISTINCTION
    if length_choice == "short":
        ratio = 0.08  # 8% for short
        min_k = 2
        max_k = 5
    elif length_choice == "long":
        ratio = 0.35  # 35% for long
        min_k = 12
        max_k = 20
    else:  # medium
        ratio = 0.18  # 18% for medium
        min_k = 6
        max_k = 10
    
    target_k = min(max(min_k, int(n * ratio)), max_k, n)
    
    if n < 10:  # For very short documents
        target_k = min(3, n)
    
    tfidf_mat = build_tfidf(sentences)
    sim_mat = cosine_similarity(tfidf_mat)
    
    tr_scores = textrank_scores(sim_mat, n)
    selected_idxs = mmr(tr_scores, sim_mat, target_k, lambda_param=0.6 if length_choice == "short" else 0.7)
    selected_idxs.sort()
    
    final_sents = [sentences[i] for i in selected_idxs]
    return final_sents, {}

def clean_for_simple_tone(text: str) -> str:
    """Clean text for simple tone output"""
    # Remove citations
    text = re.sub(r'\[[\d,\-\s]+\]', '', text)
    # Remove parenthetical content
    text = re.sub(r'\([^)]*\)', '', text)
    # Remove academic connectors
    connectors = ['however', 'therefore', 'furthermore', 'moreover', 'thus', 
                  'hence', 'consequently', 'accordingly', 'nevertheless', 
                  'nonetheless', 'in conclusion', 'in summary', 'to conclude',
                  'as a result', 'on the other hand', 'in contrast']
    for conn in connectors:
        text = re.sub(fr'\b{conn}\b[,\s]*', '', text, flags=re.IGNORECASE)
    # Clean up extra spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def build_structured_summary(summary_sentences: List[str], tone: str, language: str = "en"):
    if tone == "simple":
        # For simple tone: create single paragraph
        simple_text = " ".join(summary_sentences)
        simple_text = clean_for_simple_tone(simple_text)
        
        # Ensure it reads like a coherent paragraph
        sentences = re.split(r'(?<=[.!?])\s+', simple_text)
        if len(sentences) > 1:
            # Join with appropriate connectors
            cleaned_sentences = []
            for i, sent in enumerate(sentences):
                sent = sent.strip()
                if i > 0 and not sent[0].islower():
                    sent = sent[0].lower() + sent[1:]
                cleaned_sentences.append(sent)
            
            simple_text = " ".join(cleaned_sentences)
            simple_text = simple_text[0].upper() + simple_text[1:]
        
        return {
            "abstract": simple_text,
            "sections": [],
            "tone": "simple"
        }
    
    # Academic tone (original structured approach)
    cat_map = defaultdict(list)
    for s in summary_sentences:
        category = score_sentence_categories(s)
        cat_map[category].append(s)
    
    section_titles = {
        "key goals": "Key Goals & Targets", 
        "policy principles": "Policy Principles & Vision",
        "service delivery": "Healthcare Delivery Systems", 
        "prevention & promotion": "Prevention & Wellness Programs",
        "human resources": "Healthcare Workforce Development", 
        "financing & costs": "Financing & Resource Allocation",
        "digital health": "Digital Health Initiatives", 
        "other": "Additional Observations"
    }
    
    sections = []
    for k, title in section_titles.items():
        if cat_map[k]:
            unique_sents = []
            seen = set()
            for s in cat_map[k]:
                clean_s = clean_for_simple_tone(s)
                if clean_s and clean_s not in seen:
                    seen.add(clean_s)
                    unique_sents.append(clean_s)
            
            if unique_sents:
                sections.append({"title": title, "bullets": unique_sents})
    
    # Create abstract from key goals and principles
    abstract_candidates = []
    for cat in ["key goals", "policy principles"]:
        if cat in cat_map:
            abstract_candidates.extend(cat_map[cat][:2])
    
    if not abstract_candidates and summary_sentences:
        abstract_candidates = summary_sentences[:3]
    
    abstract = " ".join(abstract_candidates[:3])
    abstract = clean_for_simple_tone(abstract)
    
    return {
        "abstract": abstract,
        "sections": sections,
        "tone": "academic"
    }

# ---------------------- TRANSLATION FUNCTION ---------------------- #

def translate_summary_with_gemini(text: str, sections: List[Dict], target_lang: str, tone: str):
    if not GEMINI_API_KEY or target_lang == "en":
        return text, sections
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        if tone == "simple":
            # Translate simple paragraph
            prompt = f"""
            Translate this medical policy summary to {LANGUAGE_MAP.get(target_lang, target_lang)}.
            Keep the tone natural and professional. Preserve all technical terms.
            
            Text to translate: {text}
            
            Return only the translated text.
            """
            response = model.generate_content(prompt)
            return response.text.strip(), []
        
        else:
            # Translate structured summary
            sections_text = "\n".join([
                f"{sec['title']}:\n" + "\n".join([f"- {b}" for b in sec['bullets']])
                for sec in sections
            ])
            
            prompt = f"""
            Translate this structured medical policy summary to {LANGUAGE_MAP.get(target_lang, target_lang)}.
            Translate both section titles and bullet points.
            Keep the structure exactly the same.
            Preserve all technical and medical terminology.
            
            Abstract: {text}
            
            Sections:
            {sections_text}
            
            Return JSON format:
            {{
                "abstract": "translated abstract",
                "sections": [
                    {{"title": "translated title", "bullets": ["translated bullet 1", ...]}},
                    ...
                ]
            }}
            """
            
            response = model.generate_content(prompt)
            text_resp = response.text.strip()
            
            if text_resp.startswith("```json"):
                text_resp = text_resp.replace("```json", "").replace("```", "")
            
            data = json.loads(text_resp)
            return data.get("abstract", text), data.get("sections", sections)
            
    except Exception as e:
        print(f"Translation error: {e}")
        return text, sections

# ---------------------- GEMINI IMAGE PROCESSING ---------------------- #

def process_image_with_gemini(image_path: str):
    if not GEMINI_API_KEY:
        return None, "Gemini API Key missing."

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        img = Image.open(image_path)
        
        prompt = """
        Analyze this medical policy document image. Extract text and create a structured summary.
        
        Output JSON:
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

def save_summary_pdf(title: str, abstract: str, sections: List[Dict], out_path: str, tone: str = "academic"):
    c = canvas.Canvas(out_path, pagesize=A4)
    width, height = A4
    margin = 50
    y = height - margin
    
    # Title
    c.setFont("Helvetica-Bold", 18)
    c.setFillColorRGB(0.08, 0.47, 0.44)  # Teal color
    c.drawString(margin, y, title)
    y -= 40
    
    # Abstract
    c.setFont("Helvetica-Bold", 12)
    c.setFillColorRGB(0.2, 0.2, 0.2)
    c.drawString(margin, y, "Executive Summary")
    y -= 20
    
    c.setFont("Helvetica", 10)
    c.setFillColorRGB(0.3, 0.3, 0.3)
    if abstract:
        lines = simpleSplit(abstract, "Helvetica", 10, width - 2*margin)
        for line in lines:
            c.drawString(margin, y, line)
            y -= 14
        y -= 20
    
    # For simple tone, just show abstract
    if tone == "simple":
        c.save()
        return
    
    # Sections (for academic tone)
    for sec in sections:
        if y < 100:
            c.showPage()
            y = height - margin
        
        c.setFont("Helvetica-Bold", 11)
        c.setFillColorRGB(0.08, 0.47, 0.44)
        c.drawString(margin, y, sec["title"])
        y -= 18
        
        c.setFont("Helvetica", 9)
        c.setFillColorRGB(0.3, 0.3, 0.3)
        for b in sec["bullets"]:
            blines = simpleSplit(f"• {b}", "Helvetica", 9, width - 2*margin)
            for l in blines:
                c.drawString(margin + 10, y, l)
                y -= 12
            y -= 6
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
    current_lang = data.get("current_lang", "en")
    
    if not GEMINI_API_KEY:
        return jsonify({"reply": "Gemini API not configured."})
        
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""
        Context from document (first 2000 chars): {doc_text[:2000]}
        
        User Question: {message}
        
        Answer in {LANGUAGE_MAP.get(current_lang, "English")}. Keep response concise and focused on the document content.
        """
        resp = model.generate_content(prompt)
        return jsonify({"reply": resp.text})
    except Exception as e:
        return jsonify({"reply": f"Error: {str(e)}"})

@app.route("/translate_summary", methods=["POST"])
def translate_summary():
    data = request.get_json(force=True, silent=True) or {}
    text = data.get("text", "")
    sections = data.get("sections", [])
    tone = data.get("tone", "academic")
    target_lang = data.get("target_lang", "en")
    
    if not GEMINI_API_KEY:
        return jsonify({"success": False, "error": "Gemini API not configured"})
    
    try:
        translated_text, translated_sections = translate_summary_with_gemini(
            text, sections, target_lang, tone
        )
        
        return jsonify({
            "success": True,
            "translated": {
                "abstract": translated_text,
                "sections": translated_sections
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

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
    tone = request.form.get("tone", "academic")
    language = request.form.get("language", "en")
    length = request.form.get("length", "medium")
    
    # CASE 1: IMAGE -> GEMINI
    if lower_name.endswith(('.png', '.jpg', '.jpeg', '.webp')):
        orig_type = "image"
        used_model = "gemini"
        gemini_data, err = process_image_with_gemini(stored_path)
        if err or not gemini_data:
            abort(500, f"Gemini Image Processing Failed: {err}")
        orig_text = gemini_data.get("extracted_text", "")
        structured_data = gemini_data.get("summary_structure", {})
        
        if "abstract" not in structured_data:
            structured_data["abstract"] = "Summary not generated."
        if "sections" not in structured_data:
            structured_data["sections"] = []

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
        
        # Extract with proper length settings
        sents, _ = summarize_extractive(orig_text, length)
        structured_data = build_structured_summary(sents, tone, language)
    
    # Translate if needed
    if language != "en" and GEMINI_API_KEY:
        try:
            translated_abstract, translated_sections = translate_summary_with_gemini(
                structured_data.get("abstract", ""),
                structured_data.get("sections", []),
                language,
                tone
            )
            structured_data["abstract"] = translated_abstract
            if tone == "academic":
                structured_data["sections"] = translated_sections
        except:
            pass  # Keep original if translation fails
    
    # Generate PDF
    summary_filename = f"{uid}_summary.pdf"
    summary_path = os.path.join(app.config["SUMMARY_FOLDER"], summary_filename)
    save_summary_pdf(
        "Policy Analysis Summary",
        structured_data.get("abstract", ""),
        structured_data.get("sections", []),
        summary_path,
        tone
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
        tone=tone,
        language=language,
        summary_pdf_url=url_for("summary_file", filename=summary_filename),
        used_model=used_model
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
