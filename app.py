import io
import os
import re
import uuid
import time
import json
from collections import defaultdict
from typing import List, Tuple, Dict
from threading import Thread
from flask import (
    Flask,
    request,
    render_template_string,
    abort,
    send_from_directory,
    jsonify,
    url_for,
    Response,
    stream_with_context,
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
import base64

# ---------------------- CONFIG ---------------------- #

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
SUMMARY_FOLDER = os.path.join(BASE_DIR, "summaries")
STATIC_FOLDER = os.path.join(BASE_DIR, "static")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SUMMARY_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SUMMARY_FOLDER"] = SUMMARY_FOLDER

# Store processing status
processing_status = {}

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
  <link rel="stylesheet"
        href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <script>
    tailwind.config = {
      theme: {
        extend: {
          fontFamily: {
            sans: ['Inter', 'sans-serif'],
            mono: ['JetBrains Mono', 'monospace'],
          },
          colors: {
            teal: {
              50: '#f0fdfa', 100: '#ccfbf1', 200: '#99f6e4', 300: '#5eead4',
              400: '#2dd4bf', 500: '#14b8a6', 600: '#0d9488', 700: '#0f766e',
              800: '#115e59', 900: '#134e4a'
            },
            gradient: {
              start: '#06b6d4',
              end: '#10b981'
            }
          },
          animation: {
            'fade-in': 'fadeIn 0.5s ease-in-out',
            'slide-up': 'slideUp 0.3s ease-out',
            'pulse': 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
          },
          keyframes: {
            fadeIn: {
              '0%': { opacity: '0' },
              '100%': { opacity: '1' },
            },
            slideUp: {
              '0%': { transform: 'translateY(10px)', opacity: '0' },
              '100%': { transform: 'translateY(0)', opacity: '1' },
            }
          }
        }
      }
    }
  </script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    body {
      font-family: 'Inter', sans-serif;
      background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
      min-height: 100vh;
    }
    
    .glass-card {
      background: rgba(255, 255, 255, 0.95);
      backdrop-filter: blur(10px);
      border: 1px solid rgba(255, 255, 255, 0.2);
      box-shadow: 0 8px 32px rgba(31, 38, 135, 0.1);
    }
    
    .gradient-bg {
      background: linear-gradient(135deg, #06b6d4 0%, #10b981 100%);
    }
    
    .gradient-text {
      background: linear-gradient(135deg, #06b6d4 0%, #10b981 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }
    
    .file-preview-container {
      min-height: 120px;
      transition: all 0.3s ease;
    }
    
    .progress-bar {
      height: 6px;
      border-radius: 3px;
      background: #e2e8f0;
      overflow: hidden;
      position: relative;
    }
    
    .progress-fill {
      height: 100%;
      background: linear-gradient(90deg, #06b6d4 0%, #10b981 100%);
      border-radius: 3px;
      transition: width 0.3s ease;
      position: relative;
    }
    
    .progress-fill::after {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: linear-gradient(
        90deg,
        transparent 0%,
        rgba(255, 255, 255, 0.4) 50%,
        transparent 100%
      );
      animation: shimmer 1.5s infinite;
    }
    
    @keyframes shimmer {
      0% { transform: translateX(-100%); }
      100% { transform: translateX(100%); }
    }
    
    .blob {
      position: absolute;
      border-radius: 50%;
      filter: blur(40px);
      opacity: 0.3;
      z-index: -1;
    }
    
    .blob-1 {
      width: 500px;
      height: 500px;
      background: linear-gradient(135deg, #06b6d4 0%, #10b981 100%);
      top: -200px;
      right: -200px;
    }
    
    .blob-2 {
      width: 400px;
      height: 400px;
      background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%);
      bottom: -150px;
      left: -150px;
    }
    
    .file-item {
      background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
      border: 2px dashed #cbd5e1;
      border-radius: 12px;
      padding: 20px;
      transition: all 0.3s ease;
    }
    
    .file-item:hover {
      border-color: #06b6d4;
      box-shadow: 0 10px 25px -5px rgba(6, 182, 212, 0.1);
    }
    
    .file-item.dragover {
      border-color: #10b981;
      background: rgba(16, 185, 129, 0.05);
    }
    
    .btn-primary {
      background: linear-gradient(135deg, #06b6d4 0%, #10b981 100%);
      color: white;
      border: none;
      border-radius: 12px;
      padding: 12px 28px;
      font-weight: 600;
      font-size: 14px;
      transition: all 0.3s ease;
      box-shadow: 0 4px 15px rgba(6, 182, 212, 0.2);
    }
    
    .btn-primary:hover {
      transform: translateY(-2px);
      box-shadow: 0 6px 20px rgba(6, 182, 212, 0.3);
    }
    
    .btn-primary:active {
      transform: translateY(0);
    }
    
    .chip {
      display: inline-flex;
      align-items: center;
      padding: 6px 12px;
      background: rgba(6, 182, 212, 0.1);
      border: 1px solid rgba(6, 182, 212, 0.2);
      border-radius: 20px;
      font-size: 12px;
      font-weight: 500;
      color: #0e7490;
    }
    
    .loading-spinner {
      width: 20px;
      height: 20px;
      border: 2px solid rgba(255, 255, 255, 0.3);
      border-top-color: white;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }
    
    @keyframes spin {
      to { transform: rotate(360deg); }
    }
  </style>
</head>
<body class="text-slate-800 relative overflow-x-hidden">

  <!-- Background blobs -->
  <div class="blob blob-1"></div>
  <div class="blob blob-2"></div>

  <!-- NAVBAR -->
  <nav class="fixed w-full z-50 glass-card">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="flex justify-between h-16 items-center">
        <div class="flex items-center gap-3">
          <div class="w-10 h-10 gradient-bg rounded-xl flex items-center justify-center shadow-lg shadow-teal-500/30">
            <i class="fa-solid fa-staff-snake text-white text-lg"></i>
          </div>
          <span class="font-extrabold text-2xl tracking-tight">
            <span class="text-slate-900">Med</span><span class="gradient-text">.AI</span>
          </span>
        </div>
        <div class="hidden md:flex items-center gap-8 text-sm font-semibold">
          <span class="text-slate-500">Unsupervised · TF-IDF · TextRank · MMR</span>
          <a href="#workspace" class="btn-primary text-xs uppercase tracking-wide">
            Open Workspace
          </a>
        </div>
      </div>
    </div>
  </nav>

  <!-- HERO + WORKSPACE -->
  <main class="pt-28 pb-20 px-4">
    <div class="max-w-7xl mx-auto grid lg:grid-cols-12 gap-10 items-start">
      <!-- Left: hero text -->
      <section class="lg:col-span-5 space-y-8 animate-fade-in">
        <div class="space-y-2">
          <div class="chip">
            <i class="fa-solid fa-bolt"></i>
            Primary Healthcare Policy · Extractive NLP
          </div>
          <h1 class="text-5xl lg:text-6xl font-bold leading-tight">
            <span class="text-slate-900">Summarize policy</span><br>
            <span class="gradient-text">briefs for PHC</span>
          </h1>
          <p class="text-slate-600 text-lg font-medium leading-relaxed">
            Upload PDF, text, or capture a photo of a document. Our intelligent engine produces 
            abstract and structured bullet summaries optimized for policy and primary healthcare use-cases.
          </p>
        </div>

        <div class="space-y-4">
          <div class="flex items-center gap-3">
            <div class="w-10 h-10 rounded-full bg-teal-100 flex items-center justify-center">
              <i class="fa-solid fa-bolt text-teal-600"></i>
            </div>
            <div>
              <h3 class="font-semibold text-slate-900">Intelligent Processing</h3>
              <p class="text-sm text-slate-600">TF-IDF + TextRank + MMR algorithms</p>
            </div>
          </div>
          <div class="flex items-center gap-3">
            <div class="w-10 h-10 rounded-full bg-blue-100 flex items-center justify-center">
              <i class="fa-solid fa-camera text-blue-600"></i>
            </div>
            <div>
              <h3 class="font-semibold text-slate-900">Multi-format Support</h3>
              <p class="text-sm text-slate-600">PDF, TXT, and Image capture</p>
            </div>
          </div>
          <div class="flex items-center gap-3">
            <div class="w-10 h-10 rounded-full bg-purple-100 flex items-center justify-center">
              <i class="fa-solid fa-robot text-purple-600"></i>
            </div>
            <div>
              <h3 class="font-semibold text-slate-900">Smart AI Integration</h3>
              <p class="text-sm text-slate-600">Gemini-powered text extraction from images</p>
            </div>
          </div>
        </div>
      </section>

      <!-- Right: upload workspace -->
      <section id="workspace" class="lg:col-span-7">
        <div class="glass-card rounded-2xl p-8 animate-slide-up">
          <div class="mb-8">
            <h2 class="text-2xl font-bold text-slate-900 mb-2">Upload Policy Document</h2>
            <p class="text-slate-600">Supported: PDF · TXT · Image (camera capture)</p>
          </div>

          <form id="upload-form" action="{{ url_for('summarize') }}" method="post" enctype="multipart/form-data" class="space-y-6">
            <!-- File upload area -->
            <div class="file-item" id="drop-area">
              <div class="flex flex-col items-center justify-center text-center py-8">
                <div class="w-16 h-16 gradient-bg rounded-full flex items-center justify-center mb-4">
                  <i class="fa-solid fa-cloud-arrow-up text-white text-2xl"></i>
                </div>
                <p class="text-lg font-semibold text-slate-800 mb-2" id="upload-text">
                  Drop your file here or click to browse
                </p>
                <p class="text-sm text-slate-500 mb-4">
                  Supports PDF, TXT, JPG, PNG up to 10MB
                </p>
                
                <div class="flex flex-col sm:flex-row gap-3">
                  <label class="cursor-pointer">
                    <div class="px-6 py-3 rounded-lg border-2 border-slate-200 hover:border-teal-500 text-slate-700 hover:text-teal-700 font-medium transition-all">
                      <i class="fa-solid fa-folder-open mr-2"></i>
                      Browse Files
                    </div>
                    <input id="file-input" type="file" name="file" 
                           accept=".pdf,.txt,.jpg,.jpeg,.png"
                           class="hidden">
                  </label>
                  <label class="cursor-pointer">
                    <div class="px-6 py-3 gradient-bg text-white rounded-lg font-medium hover:opacity-90 transition-all">
                      <i class="fa-solid fa-camera mr-2"></i>
                      Use Camera
                    </div>
                    <input id="camera-input" type="file" accept="image/*" capture="environment"
                           name="file_camera" class="hidden">
                  </label>
                </div>
              </div>
              
              <!-- File preview -->
              <div id="file-preview" class="hidden mt-4 p-4 bg-slate-50 rounded-xl border border-slate-200">
                <div class="flex items-center justify-between mb-2">
                  <div class="flex items-center gap-3">
                    <div class="w-10 h-10 bg-teal-100 rounded-lg flex items-center justify-center">
                      <i class="fa-solid fa-file text-teal-600"></i>
                    </div>
                    <div>
                      <p id="file-name" class="font-medium text-slate-800"></p>
                      <p id="file-size" class="text-xs text-slate-500"></p>
                    </div>
                  </div>
                  <button type="button" onclick="clearFile()" class="text-slate-400 hover:text-red-500">
                    <i class="fa-solid fa-xmark"></i>
                  </button>
                </div>
                <div id="image-preview" class="hidden mt-3">
                  <img id="preview-image" src="" alt="Preview" class="max-h-40 rounded-lg mx-auto border border-slate-200">
                </div>
              </div>
            </div>

            <!-- Progress Bar -->
            <div id="progress-container" class="hidden">
              <div class="flex justify-between text-sm text-slate-700 mb-2">
                <span>Processing document...</span>
                <span id="progress-percentage">0%</span>
              </div>
              <div class="progress-bar">
                <div id="progress-fill" class="progress-fill" style="width: 0%"></div>
              </div>
              <p id="progress-status" class="text-xs text-slate-500 mt-2">Initializing...</p>
            </div>

            <!-- Options -->
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label class="block text-sm font-medium text-slate-700 mb-3">Summary Length</label>
                <div class="grid grid-cols-3 gap-2">
                  <label class="flex items-center justify-center p-3 border rounded-lg cursor-pointer hover:border-teal-500">
                    <input type="radio" name="length" value="short" class="hidden peer">
                    <span class="peer-checked:text-teal-600 peer-checked:font-semibold">Short</span>
                  </label>
                  <label class="flex items-center justify-center p-3 border rounded-lg cursor-pointer hover:border-teal-500">
                    <input type="radio" name="length" value="medium" checked class="hidden peer">
                    <span class="peer-checked:text-teal-600 peer-checked:font-semibold">Medium</span>
                  </label>
                  <label class="flex items-center justify-center p-3 border rounded-lg cursor-pointer hover:border-teal-500">
                    <input type="radio" name="length" value="long" class="hidden peer">
                    <span class="peer-checked:text-teal-600 peer-checked:font-semibold">Long</span>
                  </label>
                </div>
              </div>
              
              <div>
                <label class="block text-sm font-medium text-slate-700 mb-3">Tone</label>
                <div class="grid grid-cols-2 gap-2">
                  <label class="flex items-center justify-center p-3 border rounded-lg cursor-pointer hover:border-teal-500">
                    <input type="radio" name="tone" value="academic" checked class="hidden peer">
                    <span class="peer-checked:text-teal-600 peer-checked:font-semibold">Academic</span>
                  </label>
                  <label class="flex items-center justify-center p-3 border rounded-lg cursor-pointer hover:border-teal-500">
                    <input type="radio" name="tone" value="easy" class="hidden peer">
                    <span class="peer-checked:text-teal-600 peer-checked:font-semibold">Easy English</span>
                  </label>
                </div>
              </div>
            </div>

            <div class="pt-4 border-t border-slate-200">
              <button type="submit" id="generate-btn" class="btn-primary w-full">
                <i class="fa-solid fa-wand-magic-sparkles mr-2"></i>
                Generate Summary
                <span id="loading-spinner" class="loading-spinner ml-2 hidden"></span>
              </button>
            </div>
          </form>
        </div>
      </section>
    </div>
  </main>

  <script>
    // File upload handling
    const fileInput = document.getElementById('file-input');
    const cameraInput = document.getElementById('camera-input');
    const dropArea = document.getElementById('drop-area');
    const filePreview = document.getElementById('file-preview');
    const fileName = document.getElementById('file-name');
    const fileSize = document.getElementById('file-size');
    const uploadText = document.getElementById('upload-text');
    const imagePreview = document.getElementById('image-preview');
    const previewImage = document.getElementById('preview-image');
    const generateBtn = document.getElementById('generate-btn');
    const progressContainer = document.getElementById('progress-container');
    const progressFill = document.getElementById('progress-fill');
    const progressPercentage = document.getElementById('progress-percentage');
    const progressStatus = document.getElementById('progress-status');
    const loadingSpinner = document.getElementById('loading-spinner');

    let currentFile = null;

    function formatFileSize(bytes) {
      if (bytes === 0) return '0 Bytes';
      const k = 1024;
      const sizes = ['Bytes', 'KB', 'MB', 'GB'];
      const i = Math.floor(Math.log(bytes) / Math.log(k));
      return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    function updateFilePreview(file) {
      currentFile = file;
      fileName.textContent = file.name;
      fileSize.textContent = formatFileSize(file.size);
      uploadText.textContent = 'File ready for processing';
      
      // Show preview for images
      if (file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = (e) => {
          previewImage.src = e.target.result;
          imagePreview.classList.remove('hidden');
        };
        reader.readAsDataURL(file);
      } else {
        imagePreview.classList.add('hidden');
      }
      
      filePreview.classList.remove('hidden');
    }

    function clearFile() {
      currentFile = null;
      fileInput.value = '';
      cameraInput.value = '';
      filePreview.classList.add('hidden');
      imagePreview.classList.add('hidden');
      uploadText.textContent = 'Drop your file here or click to browse';
    }

    // File input change
    fileInput.addEventListener('change', (e) => {
      if (e.target.files.length > 0) {
        updateFilePreview(e.target.files[0]);
      }
    });

    // Camera input change
    cameraInput.addEventListener('change', (e) => {
      if (e.target.files.length > 0) {
        updateFilePreview(e.target.files[0]);
      }
    });

    // Drag and drop
    dropArea.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropArea.classList.add('dragover');
    });

    dropArea.addEventListener('dragleave', () => {
      dropArea.classList.remove('dragover');
    });

    dropArea.addEventListener('drop', (e) => {
      e.preventDefault();
      dropArea.classList.remove('dragover');
      
      if (e.dataTransfer.files.length > 0) {
        updateFilePreview(e.dataTransfer.files[0]);
        fileInput.files = e.dataTransfer.files;
      }
    });

    // Click to browse
    dropArea.addEventListener('click', (e) => {
      if (e.target === dropArea || e.target === uploadText) {
        fileInput.click();
      }
    });

    // Form submission with progress
    document.getElementById('upload-form').addEventListener('submit', async (e) => {
      if (!currentFile) {
        e.preventDefault();
        alert('Please select a file first');
        return;
      }

      e.preventDefault();
      
      // Show progress bar and disable button
      progressContainer.classList.remove('hidden');
      generateBtn.disabled = true;
      loadingSpinner.classList.remove('hidden');
      
      const formData = new FormData(e.target);
      
      try {
        const response = await fetch('{{ url_for("summarize") }}', {
          method: 'POST',
          body: formData
        });
        
        if (response.ok) {
          // Redirect to result page
          window.location.href = response.url;
        } else {
          const error = await response.text();
          alert('Error: ' + error);
          resetProgress();
        }
      } catch (error) {
        alert('Error: ' + error.message);
        resetProgress();
      }
    });

    // Simulate progress updates
    function updateProgress(percentage, status) {
      if (percentage >= 0 && percentage <= 100) {
        progressFill.style.width = percentage + '%';
        progressPercentage.textContent = percentage + '%';
        progressStatus.textContent = status;
        
        if (percentage === 100) {
          progressStatus.textContent = 'Redirecting to results...';
        }
      }
    }

    function resetProgress() {
      progressContainer.classList.add('hidden');
      generateBtn.disabled = false;
      loadingSpinner.classList.add('hidden');
      updateProgress(0, 'Initializing...');
    }

    // Initialize progress simulation (for demo)
    function simulateProgress() {
      let progress = 0;
      const interval = setInterval(() => {
        if (progress >= 100) {
          clearInterval(interval);
          return;
        }
        
        progress += Math.random() * 15;
        if (progress > 100) progress = 100;
        
        let status = 'Initializing...';
        if (progress < 25) status = 'Extracting text...';
        else if (progress < 50) status = 'Analyzing content...';
        else if (progress < 75) status = 'Generating summary...';
        else if (progress < 95) status = 'Finalizing results...';
        else status = 'Almost done...';
        
        updateProgress(progress, status);
      }, 500);
    }

    // Start progress simulation when form is submitted
    document.getElementById('upload-form').addEventListener('submit', simulateProgress);
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
  <link rel="stylesheet"
        href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    body {
      font-family: 'Inter', sans-serif;
      background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    }
    
    .glass-card {
      background: rgba(255, 255, 255, 0.95);
      backdrop-filter: blur(10px);
      border: 1px solid rgba(255, 255, 255, 0.2);
      box-shadow: 0 8px 32px rgba(31, 38, 135, 0.1);
      border-radius: 16px;
    }
    
    .gradient-bg {
      background: linear-gradient(135deg, #06b6d4 0%, #10b981 100%);
    }
    
    .gradient-text {
      background: linear-gradient(135deg, #06b6d4 0%, #10b981 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }
    
    .section-card {
      background: white;
      border-radius: 16px;
      border: 1px solid #e2e8f0;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
      transition: all 0.3s ease;
    }
    
    .section-card:hover {
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
    }
    
    .chat-bubble {
      max-width: 80%;
      padding: 12px 16px;
      border-radius: 20px;
      margin: 8px 0;
    }
    
    .chat-user {
      background: linear-gradient(135deg, #06b6d4 0%, #10b981 100%);
      color: white;
      margin-left: auto;
    }
    
    .chat-assistant {
      background: #f1f5f9;
      color: #334155;
    }
    
    .btn-primary {
      background: linear-gradient(135deg, #06b6d4 0%, #10b981 100%);
      color: white;
      border: none;
      border-radius: 12px;
      padding: 12px 24px;
      font-weight: 600;
      font-size: 14px;
      transition: all 0.3s ease;
      box-shadow: 0 4px 15px rgba(6, 182, 212, 0.2);
    }
    
    .btn-primary:hover {
      transform: translateY(-2px);
      box-shadow: 0 6px 20px rgba(6, 182, 212, 0.3);
    }
    
    .stat-card {
      background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
      border: 2px solid #e2e8f0;
      border-radius: 12px;
      padding: 16px;
    }
    
    .progress-circle {
      width: 80px;
      height: 80px;
      background: conic-gradient(#10b981 var(--progress, 0%), #e2e8f0 0%);
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      position: relative;
    }
    
    .progress-circle::before {
      content: attr(data-progress) '%';
      position: absolute;
      font-size: 14px;
      font-weight: 600;
      color: #0f766e;
    }
  </style>
</head>
<body class="text-slate-800">

  <nav class="fixed w-full z-40 glass-card">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="flex justify-between h-16 items-center">
        <div class="flex items-center gap-3">
          <div class="w-10 h-10 gradient-bg rounded-xl flex items-center justify-center shadow-lg shadow-teal-500/30">
            <i class="fa-solid fa-staff-snake text-white text-lg"></i>
          </div>
          <span class="font-extrabold text-xl tracking-tight">
            <span class="text-slate-900">Med</span><span class="gradient-text">.AI</span>
          </span>
        </div>
        <a href="{{ url_for('index') }}"
           class="btn-primary text-sm">
          <i class="fa-solid fa-plus mr-2"></i>
          New Summary
        </a>
      </div>
    </div>
  </nav>

  <main class="pt-24 pb-12 px-4">
    <div class="max-w-7xl mx-auto">
      <!-- Header -->
      <div class="mb-8">
        <h1 class="text-3xl md:text-4xl font-bold text-slate-900 mb-2">{{ title }}</h1>
        <p class="text-slate-600">Automatic extractive summary with policy-aware heuristics</p>
      </div>

      <div class="grid lg:grid-cols-3 gap-8">
        <!-- Left column - Stats and Abstract -->
        <div class="lg:col-span-2 space-y-8">
          <!-- Stats Cards -->
          <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div class="stat-card">
              <div class="flex items-center justify-between">
                <div>
                  <p class="text-sm text-slate-500">Original Sentences</p>
                  <p class="text-2xl font-bold text-slate-900">{{ stats.original_sentences }}</p>
                </div>
                <div class="text-teal-500">
                  <i class="fa-solid fa-file-lines text-2xl"></i>
                </div>
              </div>
            </div>
            
            <div class="stat-card">
              <div class="flex items-center justify-between">
                <div>
                  <p class="text-sm text-slate-500">Summary Sentences</p>
                  <p class="text-2xl font-bold text-slate-900">{{ stats.summary_sentences }}</p>
                </div>
                <div class="text-blue-500">
                  <i class="fa-solid fa-compress text-2xl"></i>
                </div>
              </div>
            </div>
            
            <div class="stat-card">
              <div class="flex items-center justify-between">
                <div>
                  <p class="text-sm text-slate-500">Compression Ratio</p>
                  <p class="text-2xl font-bold text-slate-900">{{ stats.compression_ratio }}%</p>
                </div>
                <div style="--progress: {{ stats.compression_ratio }};" class="progress-circle"></div>
              </div>
            </div>
          </div>

          <!-- Abstract Section -->
          <div class="section-card p-6">
            <div class="flex items-center gap-3 mb-4">
              <div class="w-10 h-10 gradient-bg rounded-lg flex items-center justify-center">
                <i class="fa-solid fa-quote-left text-white"></i>
              </div>
              <div>
                <h2 class="text-xl font-bold text-slate-900">Abstract</h2>
                <p class="text-sm text-slate-500">Key insights from the document</p>
              </div>
            </div>
            <div class="bg-slate-50 rounded-xl p-4">
              <p class="text-slate-700 leading-relaxed">{{ abstract }}</p>
            </div>
          </div>

          <!-- Structured Summary -->
          {% if sections %}
          <div class="section-card p-6">
            <div class="flex items-center gap-3 mb-6">
              <div class="w-10 h-10 bg-amber-500 rounded-lg flex items-center justify-center">
                <i class="fa-solid fa-list-check text-white"></i>
              </div>
              <div>
                <h2 class="text-xl font-bold text-slate-900">Structured Summary</h2>
                <p class="text-sm text-slate-500">Organized by key themes and topics</p>
              </div>
            </div>
            
            <div class="space-y-4">
              {% for sec in sections %}
              <div class="border border-slate-200 rounded-xl p-4 hover:border-teal-300 transition-colors">
                <div class="flex items-center gap-3 mb-3">
                  <div class="w-6 h-6 rounded-full bg-teal-100 flex items-center justify-center">
                    <span class="text-xs font-bold text-teal-700">{{ loop.index }}</span>
                  </div>
                  <h3 class="font-semibold text-slate-900">{{ sec.title }}</h3>
                </div>
                <ul class="space-y-2">
                  {% for bullet in sec.bullets %}
                  <li class="flex items-start gap-2 text-sm text-slate-700">
                    <i class="fa-solid fa-circle text-teal-500 text-xs mt-1"></i>
                    <span>{{ bullet }}</span>
                  </li>
                  {% endfor %}
                </ul>
              </div>
              {% endfor %}
            </div>
          </div>
          {% endif %}

          <!-- Implementation Points -->
          {% if implementation_points %}
          <div class="section-card p-6">
            <div class="flex items-center gap-3 mb-4">
              <div class="w-10 h-10 bg-emerald-500 rounded-lg flex items-center justify-center">
                <i class="fa-solid fa-road text-white"></i>
              </div>
              <div>
                <h2 class="text-xl font-bold text-slate-900">Implementation Points</h2>
                <p class="text-sm text-slate-500">Key actions and next steps</p>
              </div>
            </div>
            <div class="space-y-3">
              {% for point in implementation_points %}
              <div class="flex items-start gap-3 p-3 bg-emerald-50 rounded-lg">
                <i class="fa-solid fa-check text-emerald-600 mt-0.5"></i>
                <span class="text-sm text-slate-700">{{ point }}</span>
              </div>
              {% endfor %}
            </div>
          </div>
          {% endif %}
        </div>

        <!-- Right column - Document preview and chat -->
        <div class="space-y-8">
          <!-- Document Preview -->
          <div class="section-card p-6">
            <div class="flex items-center justify-between mb-4">
              <div class="flex items-center gap-3">
                <div class="w-10 h-10 bg-slate-800 rounded-lg flex items-center justify-center">
                  <i class="fa-solid fa-file text-white"></i>
                </div>
                <div>
                  <h2 class="text-xl font-bold text-slate-900">Original Document</h2>
                  <p class="text-sm text-slate-500">Source material for analysis</p>
                </div>
              </div>
              <div class="chip bg-slate-100 text-slate-700 text-xs font-medium px-3 py-1 rounded-full">
                {{ file_type }}
              </div>
            </div>
            
            <div class="border border-slate-200 rounded-xl overflow-hidden bg-slate-50 h-[400px]">
              {% if orig_type == 'pdf' %}
                <iframe src="{{ orig_url }}" class="w-full h-full" title="Original PDF"></iframe>
              {% elif orig_type == 'text' %}
                <div class="p-4 h-full overflow-y-auto">
                  <pre class="whitespace-pre-wrap text-sm text-slate-700">{{ orig_text[:2000] }}{% if orig_text|length > 2000 %}...{% endif %}</pre>
                </div>
              {% elif orig_type == 'image' %}
                <div class="flex items-center justify-center h-full p-4">
                  <img src="{{ orig_url }}" alt="Uploaded document image" class="max-h-full max-w-full object-contain rounded-lg">
                </div>
              {% endif %}
            </div>
            
            {% if summary_pdf_url %}
            <div class="mt-4">
              <a href="{{ summary_pdf_url }}" class="btn-primary w-full text-center">
                <i class="fa-solid fa-file-pdf mr-2"></i>
                Download Summary PDF
              </a>
            </div>
            {% endif %}
          </div>

          <!-- Gemini Chat -->
          <div class="section-card p-6">
            <div class="flex items-center justify-between mb-4">
              <div class="flex items-center gap-3">
                <div class="w-10 h-10 bg-purple-500 rounded-lg flex items-center justify-center">
                  <i class="fa-solid fa-robot text-white"></i>
                </div>
                <div>
                  <h2 class="text-xl font-bold text-slate-900">Ask Questions</h2>
                  <p class="text-sm text-slate-500">Powered by Gemini AI</p>
                </div>
              </div>
              <span class="text-xs px-3 py-1 rounded-full bg-purple-100 text-purple-700 font-medium">
                Gemini-connected
              </span>
            </div>
            
            <div id="chat-container" class="border border-slate-200 rounded-xl bg-slate-50 p-4 h-[300px] overflow-y-auto mb-4">
              <div class="chat-bubble chat-assistant">
                <p class="text-sm">Ask me anything about this policy brief — goals, strategies, financing, or implications for primary healthcare.</p>
              </div>
            </div>
            
            <div class="flex gap-2">
              <input id="chat-input" 
                     type="text" 
                     placeholder="Type your question here..."
                     class="flex-1 border border-slate-300 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-transparent">
              <button id="chat-send" class="btn-primary">
                <i class="fa-solid fa-paper-plane"></i>
              </button>
            </div>
            
            <p class="text-xs text-slate-500 mt-2">
              Responses are generated using only this document as context.
            </p>
            
            <textarea id="doc-context" class="hidden">{{ doc_context }}</textarea>
          </div>
        </div>
      </div>
    </div>
  </main>

  <!-- Gemini chat JavaScript -->
  <script>
    (function() {
      const chatContainer = document.getElementById('chat-container');
      const chatInput = document.getElementById('chat-input');
      const chatSend = document.getElementById('chat-send');
      const docContext = document.getElementById('doc-context');
      const docText = docContext.value || "";

      function addMessage(role, text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-bubble ${role === 'user' ? 'chat-user' : 'chat-assistant'}`;
        messageDiv.innerHTML = `<p class="text-sm">${text}</p>`;
        chatContainer.appendChild(messageDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
      }

      async function sendMessage() {
        const message = chatInput.value.trim();
        if (!message) return;
        
        // Add user message
        addMessage('user', message);
        chatInput.value = '';
        
        // Show typing indicator
        const typingDiv = document.createElement('div');
        typingDiv.className = 'chat-bubble chat-assistant';
        typingDiv.innerHTML = '<div class="flex items-center gap-2"><div class="w-2 h-2 bg-slate-400 rounded-full animate-pulse"></div><div class="w-2 h-2 bg-slate-400 rounded-full animate-pulse delay-100"></div><div class="w-2 h-2 bg-slate-400 rounded-full animate-pulse delay-200"></div></div>';
        chatContainer.appendChild(typingDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
        
        try {
          const response = await fetch('{{ url_for("chat") }}', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
              message: message, 
              doc_text: docText 
            })
          });
          
          const data = await response.json();
          
          // Remove typing indicator
          chatContainer.removeChild(typingDiv);
          
          // Add assistant response
          addMessage('assistant', data.reply || 'No response received.');
          
        } catch (error) {
          chatContainer.removeChild(typingDiv);
          addMessage('assistant', 'Error: Unable to connect to the server.');
        }
      }

      chatSend.addEventListener('click', sendMessage);
      chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          sendMessage();
        }
      });
    })();
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
        parts = re.split(r"(?<=[\.\?\!])\s+(?=[A-Z0-9"'\"-])", chunk)
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
    reader = PdfReader(io.BytesIO(raw))
    pages = []
    for pg in reader.pages:
        pages.append(pg.extract_text() or "")
    return "\n".join(pages)


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


def detect_title(raw_text: str) -> str:
    for line in raw_text.splitlines():
        s = line.strip()
        if len(s) < 5:
            continue
        if "content" in s.lower():
            break
        return s
    return "Policy Document"


# ---------------------- GEMINI IMAGE TEXT EXTRACTION ---------------------- #

def extract_text_from_image_gemini(image_bytes: bytes) -> str:
    """Extract text from image using Gemini Vision"""
    if not GEMINI_API_KEY:
        raise ValueError("Gemini API key not configured")
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        image = Image.open(io.BytesIO(image_bytes))
        
        # Prepare prompt for text extraction
        prompt = """Extract all text from this image accurately. 
        Preserve the original formatting, structure, and numbering.
        Return only the extracted text without any additional commentary.
        If there are tables or structured data, convert them to readable text format."""
        
        response = model.generate_content([prompt, image])
        return response.text.strip()
    
    except Exception as e:
        raise Exception(f"Gemini image text extraction failed: {str(e)}")


# ---------------------- GOAL & CATEGORY HELPERS ---------------------- #

GOAL_METRIC_WORDS = [
    "life expectancy",
    "mortality",
    "imr",
    "u5mr",
    "mmr",
    "coverage",
    "immunization",
    "immunisation",
    "incidence",
    "prevalence",
    "%",
    " per ",
    "gdp",
    "reduction",
    "rate",
]

GOAL_VERBS = [
    "reduce",
    "reducing",
    "reduction",
    "increase",
    "increasing",
    "improve",
    "improving",
    "achieve",
    "achieving",
    "eliminate",
    "eliminating",
    "raise",
    "raising",
    "reach",
    "reaching",
    "decrease",
    "decreasing",
    "enhance",
    "enhancing",
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

    if any(
        w in s_lower
        for w in [
            "principle",
            "values",
            "equity",
            "universal access",
            "universal health",
            "right to health",
            "accountability",
            "integrity",
            "patient-centred",
            "patient-centered",
        ]
    ):
        return "policy principles"

    if any(
        w in s_lower
        for w in [
            "primary care",
            "secondary care",
            "tertiary care",
            "health and wellness centre",
            "health & wellness centre",
            "health & wellness center",
            "hospital",
            "service delivery",
            "referral",
            "emergency services",
            "free drugs",
            "free diagnostics",
        ]
    ):
        return "service delivery"

    if any(
        w in s_lower
        for w in [
            "prevention",
            "preventive",
            "promotive",
            "promotion",
            "sanitation",
            "nutrition",
            "tobacco",
            "alcohol",
            "air pollution",
            "road safety",
            "lifestyle",
            "behaviour change",
            "behavior change",
            "swachh",
            "clean water",
        ]
    ):
        return "prevention & promotion"

    if any(
        w in s_lower
        for w in [
            "human resources for health",
            "hrh",
            "health workforce",
            "doctors",
            "nurses",
            "mid-level",
            "medical college",
            "nursing college",
            "public health management cadre",
            "training",
            "capacity building",
        ]
    ):
        return "human resources"

    if any(
        w in s_lower
        for w in [
            "financing",
            "financial protection",
            "insurance",
            "strategic purchasing",
            "public spending",
            "gdp",
            "expenditure",
            "catastrophic",
            "private sector",
            "ppp",
            "reimbursement",
            "fees",
            "empanelment",
        ]
    ):
        return "financing & private sector"

    if any(
        w in s_lower
        for w in [
            "digital health",
            "health information",
            "ehr",
            "electronic health record",
            "telemedicine",
            "information system",
            "surveillance",
            "ndha",
            "health data",
        ]
    ):
        return "digital health"

    if any(w in s_lower for w in ["ayush", "ayurveda", "yoga", "unani", "siddha", "homeopathy"]):
        return "ayush integration"

    if any(
        w in s_lower
        for w in [
            "implementation",
            "way forward",
            "roadmap",
            "action plan",
            "strategy",
            "governance",
            "monitoring",
            "evaluation",
            "framework",
        ]
    ):
        return "implementation"

    return "other"


# ---------------------- TF-IDF + TEXTRANK + MMR ---------------------- #

def build_tfidf(sentences: List[str]):
    vec = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_df=0.9,
        min_df=1,
    )
    mat = vec.fit_transform(sentences)
    return mat


def textrank_scores(sim_mat: np.ndarray, positional_boost: np.ndarray = None) -> Dict[int, float]:
    np.fill_diagonal(sim_mat, 0.0)
    G = nx.from_numpy_array(sim_mat)
    pr = nx.pagerank(G, max_iter=200, tol=1e-6)
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


# ---------------------- EXTRACTIVE SUMMARIZER ---------------------- #

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
        summary_sentences = sentences
        summary_text = " ".join(summary_sentences)
        stats = {
            "original_sentences": n,
            "summary_sentences": len(summary_sentences),
            "original_chars": len(cleaned),
            "summary_chars": len(summary_text),
            "compression_ratio": 100,
        }
        return summary_sentences, stats

    if length_choice == "short":
        ratio, max_s = 0.10, 6
    elif length_choice == "long":
        ratio, max_s = 0.30, 20
    else:
        ratio, max_s = 0.20, 12

    target = min(max(1, int(round(n * ratio))), max_s, n)

    tfidf = build_tfidf(sentences)
    sim = cosine_similarity(tfidf)

    pos_boost = np.zeros(n, dtype=float)
    sec_first_idx: Dict[int, int] = {}
    for idx, sec_idx in enumerate(sent_to_section):
        sec_first_idx.setdefault(sec_idx, None)
        if sec_first_idx[sec_idx] is None:
            sec_first_idx[sec_idx] = idx
    num_sections = max(sent_to_section) + 1 if sent_to_section else 1
    for sec_idx, first_idx in sec_first_idx.items():
        if first_idx is not None:
            if num_sections > 1:
                weight = 0.06 * (1.0 - (sec_idx / (num_sections - 1)))
            else:
                weight = 0.06
            pos_boost[first_idx] += weight

    tr_scores = textrank_scores(sim, positional_boost=pos_boost)

    sec_scores = defaultdict(float)
    for i, sec_idx in enumerate(sent_to_section):
        row = tfidf[i]
        sec_scores[sec_idx] += float(np.linalg.norm(row.toarray()))

    for sec_idx, (title, _) in enumerate(sections):
        t = title.lower()
        boost = 1.0
        if any(w in t for w in ["goal", "objective"]):
            boost *= 1.5
        if "principle" in t:
            boost *= 1.2
        sec_scores[sec_idx] *= boost

    sorted_secs = sorted(sec_scores.items(), key=lambda x: -x[1])
    num_secs = len(sorted_secs)
    per_section_quota = [0] * num_secs

    if target >= num_secs:
        for i in range(min(target, num_secs)):
            per_section_quota[i] = 1
        remaining = target - sum(per_section_quota)
        idx = 0
        while remaining > 0 and num_secs > 0:
            per_section_quota[idx % num_secs] += 1
            idx += 1
            remaining -= 1
    else:
        for i in range(target):
            per_section_quota[i] = 1

    selected_idxs: List[int] = []
    sec_to_global = defaultdict(list)
    sec_order = [s for s, _ in sorted_secs]
    for g_idx, s_idx in enumerate(sent_to_section):
        sec_to_global[s_idx].append(g_idx)

    for rank_pos, sec_idx in enumerate(sec_order):
        quota = per_section_quota[rank_pos] if rank_pos < len(per_section_quota) else 0
        if quota <= 0:
            continue
        candidates = sec_to_global.get(sec_idx, [])
        if not candidates:
            continue
        local_index_map = {g: i for i, g in enumerate(candidates)}
        local_sim = sim[np.ix_(candidates, candidates)]
        local_scores = {i: tr_scores[g] for g, i in local_index_map.items()}
        local_picks = mmr(local_scores, local_sim, min(quota, len(candidates)), lambda_param=0.75)
        for lp in local_picks:
            selected_idxs.append(candidates[lp])

    if len(selected_idxs) < target:
        remaining = target - len(selected_idxs)
        already = set(selected_idxs)
        ranked_global = sorted(range(n), key=lambda i: -tr_scores.get(i, 0.0))
        cand = [i for i in ranked_global if i not in already]
        local_scores = {idx: tr_scores.get(idx, 0.0) for idx in cand}
        local_sim = sim[np.ix_(candidates, candidates)]
        global_picks = mmr(local_scores, local_sim, min(remaining, len(cand)), lambda_param=0.7)
        for p in global_picks:
            selected_idxs.append(cand[p])

    # force-in goal sentences
    goal_indices = [i for i, s in enumerate(sentences) if is_goal_sentence(s)]
    goal_indices_sorted = sorted(goal_indices, key=lambda i: tr_scores.get(i, 0.0), reverse=True)

    if goal_indices_sorted:
        max_goal = max(1, min(3, int(0.25 * target)))
        forced_goal = goal_indices_sorted[:max_goal]
    else:
        forced_goal = []

    combined = set(selected_idxs) | set(forced_goal)
    if len(combined) > target:
        goal_set = set(forced_goal)
        non_goal = [i for i in combined if i not in goal_set]
        non_goal_sorted = sorted(non_goal, key=lambda i: tr_scores.get(i, 0.0), reverse=True)
        keep_non_goal = non_goal_sorted[: max(0, target - len(goal_set))]
        combined = goal_set | set(keep_non_goal)

    selected_idxs = sorted(combined)

    summary_sentences = [sentences[i].strip() for i in selected_idxs][:target]
    summary_text = " ".join(summary_sentences)
    stats = {
        "original_sentences": n,
        "summary_sentences": len(summary_sentences),
        "original_chars": len(cleaned),
        "summary_chars": len(summary_text),
        "compression_ratio": int(round(100.0 * len(summary_sentences) / max(1, n))),
    }
    return summary_sentences, stats


# ---------------------- STRUCTURED SUMMARY ---------------------- #

def simplify_for_easy_english(s: str) -> str:
    s = re.sub(r"\([^)]{1,30}\)", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def build_structured_summary(summary_sentences: List[str], tone: str):
    if tone == "easy":
        processed = [simplify_for_easy_english(s) for s in summary_sentences]
    else:
        processed = summary_sentences[:]

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
            unique_bullets = []
            for b in bullets:
                if b not in seen:
                    seen.add(b)
                    unique_bullets.append(b)
            sections.append({"title": title, "bullets": unique_bullets})

    category_counts = {k: len(v) for k, v in category_to_sentences.items()}
    implementation_points = category_to_sentences.get("implementation", [])

    return {
        "abstract": abstract,
        "sections": sections,
        "category_counts": category_counts,
        "implementation_points": implementation_points,
    }


# ---------------------- SUMMARY PDF GENERATION ---------------------- #

def save_summary_pdf(title: str, abstract: str, sections: List[Dict], out_path: str):
    c = canvas.Canvas(out_path, pagesize=A4)
    width, height = A4
    margin_x = 40
    margin_y = 40
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
    for line in simpleSplit(title, "Helvetica-Bold", 14, max_width):
        c.drawString(margin_x, y, line)
        y -= 18

    y -= 10
    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin_x, y, "Abstract")
    y -= 14
    y = draw_paragraph(abstract, y)

    y -= 8
    for sec in sections:
        if y < margin_y + 40:
            c.showPage()
            y = height - margin_y
        c.setFont("Helvetica-Bold", 11)
        c.drawString(margin_x, y, sec["title"])
        y -= 14
        for bullet in sec["bullets"]:
            bullet_text = "• " + bullet
            y = draw_paragraph(bullet_text, y)
            y -= 2
        y -= 6

    c.showPage()
    c.save()


# ---------------------- GEMINI CHAT ---------------------- #

def gemini_answer(user_message: str, doc_text: str) -> str:
    if not GEMINI_API_KEY:
        return "Gemini API key is not configured on the server."

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = (
            "You are an AI assistant helping a student understand a healthcare policy document.\n"
            "Answer concisely and only using information from the document.\n\n"
            "DOCUMENT:\n"
            f"{doc_text[:120000]}\n\n"
            f"USER QUESTION: {user_message}\n\n"
            "ANSWER:"
        )
        resp = model.generate_content(prompt)
        return (resp.text or "").strip() or "No response generated."
    except Exception as e:
        return f"Error while contacting Gemini API: {e}"


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
        return jsonify({"reply": "Please type a question first."})
    reply = gemini_answer(message, doc_text)
    return jsonify({"reply": reply})


@app.route("/progress/<task_id>")
def get_progress(task_id):
    """Get progress updates for a task"""
    progress = processing_status.get(task_id, {"progress": 0, "status": "Starting..."})
    return jsonify(progress)


@app.route("/summarize", methods=["POST"])
def summarize():
    # Get file from either field
    f = request.files.get("file")
    if not f or (f and f.filename == ""):
        f = request.files.get("file_camera")

    if not f or f.filename == "":
        abort(400, "No file uploaded. Please upload a PDF, text file, or image.")

    filename = f.filename or "document"
    safe_name = secure_filename(filename)
    raw_bytes = f.read()
    
    if not raw_bytes:
        abort(400, "Uploaded document appears to be empty or unreadable.")

    # Save original file
    uid = uuid.uuid4().hex
    stored_name = f"{uid}_{safe_name}"
    stored_path = os.path.join(app.config["UPLOAD_FOLDER"], stored_name)
    with open(stored_path, "wb") as out:
        out.write(raw_bytes)

    # Detect file type and extract text
    lower_name = filename.lower()
    orig_type = "unknown"
    orig_text = ""
    raw_text = ""
    file_type = ""
    
    try:
        if lower_name.endswith(".pdf"):
            orig_type = "pdf"
            file_type = "PDF"
            raw_text = extract_text_from_pdf_bytes(raw_bytes)
        elif lower_name.endswith(".txt"):
            orig_type = "text"
            file_type = "TXT"
            raw_text = raw_bytes.decode("utf-8", errors="ignore")
            orig_text = raw_text
        else:
            # Image file - use Gemini for text extraction
            orig_type = "image"
            file_type = "Image"
            
            # Try Gemini first if API key is available
            if GEMINI_API_KEY:
                try:
                    raw_text = extract_text_from_image_gemini(raw_bytes)
                except Exception as gemini_error:
                    # Fall back to pytesseract if Gemini fails
                    print(f"Gemini extraction failed, falling back to OCR: {gemini_error}")
                    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                    raw_text = pytesseract.image_to_string(img)
            else:
                # Use pytesseract if no Gemini API key
                img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                raw_text = pytesseract.image_to_string(img)
            
            orig_text = raw_text
    
    except Exception as e:
        abort(500, f"Error while extracting text from the uploaded file: {e}")

    if not raw_text or len(raw_text.strip()) < 50:
        abort(400, "Could not extract enough text from the uploaded document.")

    length_choice = request.form.get("length", "medium").lower()
    tone = request.form.get("tone", "academic").lower()

    # Generate summary
    try:
        summary_sentences, stats = summarize_extractive(raw_text, length_choice=length_choice)
        structured = build_structured_summary(summary_sentences, tone=tone)
    except Exception as e:
        abort(500, f"Error during summarization: {e}")

    # Generate title
    doc_title_raw = detect_title(raw_text)
    doc_title = doc_title_raw.strip()
    if doc_title:
        page_title = f"{doc_title} — Summary"
    else:
        page_title = "Policy Brief Summary"

    doc_context = raw_text[:8000]

    # Generate summary PDF
    summary_pdf_url = None
    try:
        summary_filename = f"{uid}_summary.pdf"
        summary_path = os.path.join(app.config["SUMMARY_FOLDER"], summary_filename)
        save_summary_pdf(
            page_title,
            structured["abstract"],
            structured["sections"],
            summary_path,
        )
        summary_pdf_url = url_for("summary_file", filename=summary_filename)
    except Exception:
        summary_pdf_url = None

    # Prepare data for template
    category_counts = structured["category_counts"]
    if category_counts:
        labels = list(category_counts.keys())
        values = [category_counts[k] for k in labels]
    else:
        labels, values = [], []

    # For text preview
    if orig_type == "text" and not orig_text:
        orig_text = raw_text[:20000]

    return render_template_string(
        RESULT_HTML,
        title=page_title,
        abstract=structured["abstract"],
        sections=structured["sections"],
        stats=stats,
        implementation_points=structured["implementation_points"],
        category_counts=category_counts,
        category_labels=labels,
        category_values=values,
        orig_type=orig_type,
        orig_url=url_for("uploaded_file", filename=stored_name),
        orig_text=orig_text,
        doc_context=doc_context,
        summary_pdf_url=summary_pdf_url,
        file_type=file_type,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
