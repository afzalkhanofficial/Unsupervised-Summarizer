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
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import simpleSplit

import google.generativeai as genai

# Try importing googletrans, handle if missing
try:
    from googletrans import Translator
    HAS_TRANSLATE = True
except ImportError:
    HAS_TRANSLATE = False

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
                        'scan': 'scan 3s linear infinite',
                    },
                    keyframes: {
                        'pulse-opacity': {
                            '0%, 100%': { opacity: 0.2, transform: 'scale(1)' },
                            '50%': { opacity: 0.5, transform: 'scale(1.1)' },
                        },
                        'float': {
                            '0%, 100%': { transform: 'translateY(0)' },
                            '50%': { transform: 'translateY(-10px)' },
                        },
                        'scan': {
                            '0%': { top: '0%', opacity: 0 },
                            '10%': { opacity: 1 },
                            '90%': { opacity: 1 },
                            '100%': { top: '100%', opacity: 0 },
                        }
                    }
                },
            },
        };
    </script>

    <style>
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #0D0D0F; }
        ::-webkit-scrollbar-thumb { background: #374151; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #4b5563; }

        body { background-color: #0D0D0F; color: #F3F4F6; }

        .perspective-1000 { perspective: 1000px; }
        .transform-style-3d { transform-style: preserve-3d; }
        
        .glossary-card {
            border-radius: 0.75rem;
            padding: 2rem;
            position: relative;
            overflow: hidden;
            transition: transform 0.1s ease-out, box-shadow 0.3s ease;
            transform-style: preserve-3d;
            background-color: #161b22;
            border: 1px solid #374151;
        }

        .glossary-card:hover {
            box-shadow: 0 0 30px -5px rgba(140, 79, 255, 0.15);
            z-index: 10;
            border-color: rgba(140, 79, 255, 0.5);
        }

        .fade-up {
            opacity: 0;
            transform: translateY(20px);
            transition: opacity 0.6s ease-out, transform 0.6s ease-out;
        }
        
        .upload-zone {
            background-image: url("data:image/svg+xml,%3csvg width='100%25' height='100%25' xmlns='http://www.w3.org/2000/svg'%3e%3crect width='100%25' height='100%25' fill='none' rx='12' ry='12' stroke='%23374151FF' stroke-width='2' stroke-dasharray='12%2c 12' stroke-dashoffset='0' stroke-linecap='square'/%3e%3c/svg%3e");
            transition: all 0.3s ease;
        }
        .upload-zone:hover {
            background-image: url("data:image/svg+xml,%3csvg width='100%25' height='100%25' xmlns='http://www.w3.org/2000/svg'%3e%3crect width='100%25' height='100%25' fill='none' rx='12' ry='12' stroke='%238C4FFF' stroke-width='2' stroke-dasharray='12%2c 12' stroke-dashoffset='0' stroke-linecap='square'/%3e%3c/svg%3e");
            background-color: rgba(22, 27, 34, 0.8);
        }

        /* Custom Styles for Redesigned Cards */
        .preview-terminal {
            background: #050505;
            border: 1px solid #333;
            box-shadow: inset 0 0 20px rgba(0,0,0,0.8);
        }
        .scan-line {
            position: absolute;
            left: 0; 
            width: 100%; 
            height: 2px;
            background: linear-gradient(to right, transparent, #8C4FFF, transparent);
            box-shadow: 0 0 10px #8C4FFF;
            animation: scan 2s linear infinite;
            z-index: 20;
            pointer-events: none;
        }
        
        .chat-bubble-ai {
            background: #1f2937;
            border-bottom-left-radius: 0;
            border: 1px solid #374151;
        }
        .chat-bubble-user {
            background: #8C4FFF;
            border-bottom-right-radius: 0;
            color: white;
        }
    </style>
"""

COMMON_SCRIPTS = """
<script>
document.addEventListener('DOMContentLoaded', () => {
    // 3D Tilt Effect (Only for elements with data-tilt attribute)
    const glossaryCards = document.querySelectorAll('[data-tilt]');
    glossaryCards.forEach(card => {
        card.addEventListener('mousemove', (e) => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top; 
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;
            const rotateX = ((y - centerY) / centerY) * -5;
            const rotateY = ((x - centerX) / centerX) * 5;
            card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) scale(1.02)`;
        });
        card.addEventListener('mouseleave', () => {
            card.style.transform = 'perspective(1000px) rotateX(0) rotateY(0) scale(1)';
        });
    });

    // Fade Up Animation
    const observerOptions = { threshold: 0.1, rootMargin: "0px 0px -20px 0px" };
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    const fadeElements = document.querySelectorAll('.fade-up');
    fadeElements.forEach(el => { observer.observe(el); });
    
    // Fallback for animations
    setTimeout(() => {
        fadeElements.forEach(el => {
            if(getComputedStyle(el).opacity === '0') {
                el.style.opacity = '1';
                el.style.transform = 'translateY(0)';
            }
        });
    }, 2000);
});
</script>
"""

INDEX_HTML = """
<!DOCTYPE html>
<html class="dark" lang="en">
<head>
    <meta charset="utf-8"/>
    <meta content="width=device-width, initial-scale=1.0" name="viewport"/>
    <title>Workspace | Med.AI</title>
    {COMMON_HEAD}
</head>
<body class="font-sans antialiased text-gray-300 bg-background-dark overflow-x-hidden selection:bg-afzal-purple selection:text-white">

<nav class="border-b border-gray-800 sticky top-0 z-50 bg-background-dark/80 backdrop-blur-md h-16">
    <div class="w-full h-full">
        <div class="grid grid-cols-[auto_1fr_auto] lg:grid-cols-3 items-center h-full">
            <div class="flex items-center h-full pl-6 lg:pl-8">
                <a href="/" class="flex items-center space-x-2.5 mr-8 shrink-0">
                    <div class="w-8 h-8 bg-white rounded-full relative overflow-hidden flex items-center justify-center">
                        <i class="fa-solid fa-staff-snake text-black text-lg"></i>
                    </div>
                    <span class="font-bold text-2xl tracking-tight text-white font-sans">Med.AI</span>
                </a>
            </div>
            <div class="hidden lg:flex justify-center">
                 <span class="text-afzal-purple font-mono text-xs uppercase tracking-widest border border-afzal-purple/30 bg-afzal-purple/10 px-3 py-1 rounded">Workspace v2.1</span>
            </div>
            <div class="flex items-center justify-end h-full pr-6 lg:pr-8">
                 <a href="/about.html" class="text-sm font-medium text-gray-400 hover:text-white transition-colors">Documentation</a>
            </div>
        </div>
    </div>
</nav>

<header class="relative pt-20 pb-12 overflow-hidden bg-background-dark">
    <div class="absolute inset-0 bg-grid-pattern-dark bg-[size:50px_50px] opacity-40"></div>
    <div class="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[500px] bg-radial-glow blur-3xl pointer-events-none"></div>

    <div class="max-w-4xl mx-auto px-4 text-center relative z-10 fade-up">
        <h1 class="text-5xl md:text-7xl font-semibold text-white mb-6 leading-tight tracking-tight">
            Intelligent <br>
            <span class="text-transparent bg-clip-text bg-gradient-to-r from-afzal-purple via-white to-afzal-blue">Policy Analysis.</span>
        </h1>
        <p class="text-xl text-gray-400 leading-relaxed max-w-2xl mx-auto font-light">
            Upload healthcare policy briefs. Our NLP engine extracts entities, categorizes clauses, and generates structured summaries in your preferred language.
        </p>
    </div>
</header>

<section class="py-12 relative z-10 bg-background-dark">
    <div class="max-w-4xl mx-auto px-4 fade-up delay-100">
        
        <div class="glossary-card bg-surface-dark border border-gray-800 p-8 rounded-2xl shadow-2xl" data-tilt>
            <form id="uploadForm" action="{{ url_for('summarize') }}" method="post" enctype="multipart/form-data" class="space-y-8">
                
                <div class="upload-zone relative w-full h-64 rounded-xl flex flex-col items-center justify-center cursor-pointer group" id="drop-zone">
                    <input id="file-input" type="file" name="file" accept=".pdf,.txt,.jpg,.jpeg,.png,.webp" multiple class="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-20">
                    
                    <div id="upload-prompt" class="text-center space-y-4 transition-all duration-300 group-hover:scale-105 transform-style-3d">
                        <div class="w-16 h-16 bg-surface-dark border border-gray-700 rounded-full flex items-center justify-center mx-auto text-afzal-purple text-2xl group-hover:border-afzal-purple group-hover:bg-afzal-purple/10 transition-colors shadow-lg">
                            <i class="fa-solid fa-cloud-arrow-up transform translate-z-[10px]"></i>
                        </div>
                        <div>
                            <p class="text-lg font-bold text-white">Upload Files</p>
                            <p class="text-sm text-gray-500 font-mono mt-1">PDF / TXT (Single) or Images (Multiple)</p>
                        </div>
                    </div>

                    <div id="file-preview" class="hidden absolute inset-0 bg-surface-dark z-10 flex flex-col items-center justify-center p-6 text-center rounded-xl">
                        <div id="preview-icon" class="mb-4 text-4xl text-afzal-purple drop-shadow-[0_0_10px_rgba(140,79,255,0.5)]"></div>
                        <div id="preview-count" class="text-2xl font-bold text-white mb-2"></div>
                        <p id="filename-display" class="font-mono text-gray-400 text-sm break-all max-w-md bg-black/30 px-4 py-2 rounded border border-gray-800"></p>
                        <button type="button" id="change-file-btn" class="mt-4 text-xs text-afzal-blue hover:text-white font-bold tracking-wide uppercase transition-colors z-30 relative">Change</button>
                    </div>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div class="bg-black/20 rounded-lg p-4 border border-gray-800">
                        <label class="block text-xs font-bold text-gray-500 uppercase tracking-widest mb-3">Length</label>
                        <div class="flex gap-1">
                            <label class="flex-1 cursor-pointer">
                                <input type="radio" name="length" value="short" class="peer hidden">
                                <span class="block text-center py-2 text-[10px] font-mono font-bold text-gray-500 bg-surface-dark rounded border border-gray-700 peer-checked:bg-afzal-purple peer-checked:text-white peer-checked:border-afzal-purple transition-all">SHORT</span>
                            </label>
                            <label class="flex-1 cursor-pointer">
                                <input type="radio" name="length" value="medium" checked class="peer hidden">
                                <span class="block text-center py-2 text-[10px] font-mono font-bold text-gray-500 bg-surface-dark rounded border border-gray-700 peer-checked:bg-afzal-purple peer-checked:text-white peer-checked:border-afzal-purple transition-all">MED</span>
                            </label>
                            <label class="flex-1 cursor-pointer">
                                <input type="radio" name="length" value="long" class="peer hidden">
                                <span class="block text-center py-2 text-[10px] font-mono font-bold text-gray-500 bg-surface-dark rounded border border-gray-700 peer-checked:bg-afzal-purple peer-checked:text-white peer-checked:border-afzal-purple transition-all">LONG</span>
                            </label>
                        </div>
                    </div>

                    <div class="bg-black/20 rounded-lg p-4 border border-gray-800">
                        <label class="block text-xs font-bold text-gray-500 uppercase tracking-widest mb-3">Output Language</label>
                        <select name="language" class="w-full bg-surface-dark border border-gray-700 text-gray-300 text-xs rounded py-2 px-3 focus:outline-none focus:border-afzal-purple focus:ring-1 focus:ring-afzal-purple">
                            <option value="en" selected>English (Default)</option>
                            <option value="hi">Hindi (हिंदी)</option>
                            <option value="te">Telugu (తెలుగు)</option>
                            <option value="ta">Tamil (தமிழ்)</option>
                            <option value="bn">Bengali (বাংলা)</option>
                            <option value="es">Spanish (Español)</option>
                            <option value="fr">French (Français)</option>
                        </select>
                    </div>

                    <div class="bg-black/20 rounded-lg p-4 border border-gray-800">
                        <label class="block text-xs font-bold text-gray-500 uppercase tracking-widest mb-3">Tone</label>
                        <div class="flex gap-2">
                            <label class="flex-1 cursor-pointer">
                                <input type="radio" name="tone" value="academic" checked class="peer hidden">
                                <span class="block text-center py-2 text-[10px] font-mono font-bold text-gray-500 bg-surface-dark rounded border border-gray-700 peer-checked:bg-afzal-blue peer-checked:text-white peer-checked:border-afzal-blue transition-all">TECH</span>
                            </label>
                            <label class="flex-1 cursor-pointer">
                                <input type="radio" name="tone" value="easy" class="peer hidden">
                                <span class="block text-center py-2 text-[10px] font-mono font-bold text-gray-500 bg-surface-dark rounded border border-gray-700 peer-checked:bg-afzal-blue peer-checked:text-white peer-checked:border-afzal-blue transition-all">EASY</span>
                            </label>
                        </div>
                    </div>
                </div>

                <button type="submit" class="w-full relative z-[1] flex items-center justify-center bg-white text-black h-14 font-semibold text-sm uppercase tracking-widest hover:bg-gray-200 transition-colors rounded">
                    Generate Analysis
                </button>
            </form>
        </div>

    </div>
</section>

<div id="progress-overlay" class="fixed inset-0 bg-background-dark/95 backdrop-blur-md z-50 hidden flex-col items-center justify-center">
    <div class="w-full max-w-md px-6 text-center space-y-8 fade-up">
        
        <div class="relative w-24 h-24 mx-auto">
            <div class="absolute inset-0 rounded-full border-4 border-gray-800"></div>
            <div class="absolute inset-0 rounded-full border-4 border-afzal-purple border-t-transparent animate-spin"></div>
            <div class="absolute inset-0 flex items-center justify-center text-white font-mono text-xl font-bold" id="progress-text">0%</div>
        </div>

        <div class="space-y-3">
            <h3 class="text-2xl font-light text-white" id="progress-stage">Initializing...</h3>
            <p class="text-sm text-gray-500 font-mono uppercase tracking-widest">Processing secure document</p>
        </div>

        <div class="w-full h-1 bg-gray-800 rounded-full overflow-hidden relative">
            <div id="progress-bar" class="h-full bg-gradient-to-r from-afzal-purple to-afzal-blue w-0 transition-all duration-300 ease-out shadow-[0_0_15px_rgba(140,79,255,0.5)]"></div>
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

    fileInput.addEventListener('change', function(e) {
      if (this.files && this.files.length > 0) {
        
        uploadPrompt.classList.add('hidden');
        filePreview.classList.remove('hidden');
        
        if (this.files.length === 1) {
            const file = this.files[0];
            filenameDisplay.textContent = file.name;
            previewCount.innerHTML = '';
            
            if (file.type.startsWith('image/')) {
                previewIcon.innerHTML = '<i class="fa-regular fa-image"></i>';
            } else if (file.type === 'application/pdf') {
                previewIcon.innerHTML = '<i class="fa-regular fa-file-pdf"></i>';
            } else {
                previewIcon.innerHTML = '<i class="fa-regular fa-file-lines"></i>';
            }
        } else {
            // Multiple files
            previewIcon.innerHTML = '<i class="fa-solid fa-layer-group"></i>';
            filenameDisplay.textContent = this.files[0].name + " + " + (this.files.length - 1) + " others";
            previewCount.innerHTML = this.files.length + " <span class='text-sm font-normal text-gray-500'>Images Selected</span>";
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
        const totalDuration = 8000; 
        const intervalTime = 100;
        const step = 100 / (totalDuration / intervalTime);

        const interval = setInterval(() => {
            if (width >= 95) {
                clearInterval(interval);
                progressStage.textContent = "Finalizing Output...";
            } else {
                width += step;
                if(Math.random() > 0.5) width += 0.5;
                
                progressBar.style.width = width + '%';
                progressText.textContent = Math.round(width) + '%';

                if (width < 30) {
                    progressStage.textContent = "Uploading & Encrypting...";
                } else if (width < 60) {
                    progressStage.textContent = "Extracting Entities...";
                } else if (width < 85) {
                    progressStage.textContent = "Translating & Summarizing...";
                } else {
                    progressStage.textContent = "Structuring Data...";
                }
            }
        }, intervalTime);
    });
</script>
{COMMON_SCRIPTS}
</body>
</html>
"""

RESULT_HTML = """
<!DOCTYPE html>
<html class="dark" lang="en">
<head>
    <meta charset="utf-8"/>
    <meta content="width=device-width, initial-scale=1.0" name="viewport"/>
    <title>{{ title }} | Med.AI</title>
    {COMMON_HEAD}
</head>
<body class="font-sans antialiased text-gray-300 bg-background-dark overflow-x-hidden selection:bg-afzal-purple selection:text-white">

<nav class="border-b border-gray-800 sticky top-0 z-50 bg-background-dark/80 backdrop-blur-md h-16">
    <div class="w-full h-full max-w-7xl mx-auto px-6">
        <div class="flex items-center justify-between h-full">
            <div class="flex items-center gap-4">
                <a href="/" class="flex items-center space-x-2.5">
                    <div class="w-8 h-8 bg-white rounded-full flex items-center justify-center">
                        <i class="fa-solid fa-staff-snake text-black text-lg"></i>
                    </div>
                    <span class="font-bold text-xl text-white">Med.AI</span>
                </a>
                <span class="text-gray-600">/</span>
                <span class="text-sm font-mono text-afzal-purple">Analysis Report</span>
            </div>
            <a href="{{ url_for('index') }}" class="group relative z-[1] inline-flex items-center cursor-pointer transition-colors text-xs font-bold uppercase tracking-widest text-white hover:text-afzal-purple border border-gray-700 hover:border-afzal-purple px-4 py-2 rounded">
                <i class="fa-solid fa-plus mr-2"></i> New
            </a>
        </div>
    </div>
</nav>

<main class="py-12 px-4 relative">
    <div class="fixed top-20 left-10 w-64 h-64 bg-afzal-purple/10 rounded-full blur-[100px] pointer-events-none"></div>

    <div class="max-w-7xl mx-auto grid lg:grid-cols-12 gap-8 relative z-10">
       
      <section class="lg:col-span-7 space-y-6 fade-up">
        
        <div class="glossary-card bg-surface-dark border border-gray-800 p-8 rounded-2xl shadow-lg">
           <div class="flex flex-wrap items-start justify-between gap-4 mb-6 border-b border-gray-700 pb-6">
             <div>
                <div class="flex items-center gap-2 mb-3">
                    <span class="px-2 py-1 rounded text-[10px] font-bold uppercase tracking-wide bg-gray-800 text-gray-400 border border-gray-600">
                        {{ orig_type }} Source
                    </span>
                    {% if language != 'en' %}
                    <span class="px-2 py-1 rounded text-[10px] font-bold uppercase tracking-wide bg-green-900/30 text-green-400 border border-green-800">
                        Translated to {{ language_name }}
                    </span>
                    {% endif %}
                    
                    {% if used_model == 'gemini' %}
                    <span class="px-2 py-1 rounded text-[10px] font-bold uppercase tracking-wide bg-afzal-purple/10 text-afzal-purple border border-afzal-purple/20">
                        <i class="fa-solid fa-wand-magic-sparkles mr-1"></i> Gemini 1.5 Flash
                    </span>
                    {% else %}
                    <span class="px-2 py-1 rounded text-[10px] font-bold uppercase tracking-wide bg-blue-500/10 text-blue-400 border border-blue-500/20">
                        BERTSum + TextRank
                    </span>
                    {% endif %}
                </div>
                <h1 class="text-3xl font-light text-white leading-tight">Executive Summary</h1>
             </div>
             {% if summary_pdf_url %}
             <a href="{{ summary_pdf_url }}" class="flex items-center justify-center w-10 h-10 rounded-full bg-white text-black hover:bg-afzal-purple hover:text-white transition-all shadow-[0_0_15px_rgba(255,255,255,0.2)]">
                <i class="fa-solid fa-file-arrow-down"></i>
             </a>
             {% endif %}
           </div>

           <div class="mb-8">
              <h2 class="text-xs font-bold text-afzal-blue uppercase tracking-widest mb-4 flex items-center gap-2">
                 <i class="fa-solid fa-layer-group"></i> Abstract
              </h2>
              <div class="p-6 rounded-lg bg-black/30 border border-gray-800 text-sm leading-relaxed text-gray-300 font-light">
                 {{ abstract }}
              </div>
           </div>

           {% if simple_text %}
           <div class="mb-8">
              <h2 class="text-xs font-bold text-gray-500 uppercase tracking-widest mb-4">Detailed Text</h2>
              <div class="text-sm leading-7 text-gray-400 text-justify">
                 {{ simple_text }}
              </div>
           </div>
           {% endif %}

           {% if sections %}
           <div class="space-y-8">
              {% for sec in sections %}
              <div class="relative pl-6 border-l border-gray-800 hover:border-afzal-purple transition-colors duration-300 group">
                 <h3 class="text-lg font-medium text-white mb-3 group-hover:text-afzal-purple transition-colors font-mono">
                    {{ sec.title }}
                 </h3>
                 <ul class="space-y-3">
                    {% for bullet in sec.bullets %}
                    <li class="flex items-start gap-3 text-sm text-gray-400">
                       <i class="fa-solid fa-angle-right mt-1 text-gray-600 group-hover:text-afzal-purple transition-colors text-xs"></i>
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

      <section class="lg:col-span-5 space-y-6 fade-up delay-100">
         
        <div class="rounded-xl shadow-lg border border-gray-800 overflow-hidden bg-[#0a0a0c]">
            <div class="flex justify-between items-center px-4 py-3 bg-[#111] border-b border-gray-800">
                 <div class="flex gap-2">
                    <div class="w-2.5 h-2.5 rounded-full bg-red-500/80"></div>
                    <div class="w-2.5 h-2.5 rounded-full bg-yellow-500/80"></div>
                    <div class="w-2.5 h-2.5 rounded-full bg-green-500/80"></div>
                 </div>
                 <div class="text-[10px] font-mono text-gray-500 uppercase tracking-widest">DOC_PREVIEW.JSX</div>
            </div>
            <div class="h-[300px] relative w-full bg-[#050505] preview-terminal overflow-hidden">
                 <div class="scan-line"></div>
                 <div class="absolute inset-0 bg-[linear-gradient(rgba(18,16,16,0)_50%,rgba(0,0,0,0.25)_50%),linear-gradient(90deg,rgba(255,0,0,0.06),rgba(0,255,0,0.02),rgba(0,0,255,0.06))] z-10 bg-[length:100%_2px,3px_100%] pointer-events-none"></div>
                 
                 <div class="relative z-0 h-full w-full">
                     {% if orig_type == 'pdf' %}
                       <iframe src="{{ orig_url }}" class="w-full h-full opacity-80 border-0" title="Original PDF"></iframe>
                     {% elif orig_type == 'text' %}
                       <div class="p-6 overflow-y-auto h-full text-xs font-mono text-green-500/80 leading-relaxed">
                           <span class="block mb-2 text-gray-500"># START OF DOCUMENT STREAM</span>
                           {{ orig_text }}
                           <span class="block mt-2 text-gray-500 animate-pulse">_</span>
                       </div>
                     {% elif orig_type == 'image' %}
                       <img src="{{ orig_url }}" class="w-full h-full object-contain bg-black">
                     {% endif %}
                 </div>
            </div>
        </div>

        <div class="rounded-xl border border-gray-800 bg-surface-dark overflow-hidden flex flex-col h-[500px] shadow-2xl">
          <div class="p-4 bg-black/40 border-b border-gray-800 flex items-center justify-between backdrop-blur-md">
            <div class="flex items-center gap-3">
                <div class="relative">
                    <div class="w-10 h-10 rounded-full bg-gradient-to-tr from-afzal-purple to-blue-600 flex items-center justify-center text-white shadow-lg">
                        <i class="fa-solid fa-robot"></i>
                    </div>
                    <div class="absolute bottom-0 right-0 w-3 h-3 bg-green-500 border-2 border-black rounded-full"></div>
                </div>
                <div>
                    <h2 class="text-sm font-bold text-white">Med.AI Assistant</h2>
                    <p class="text-[10px] text-gray-400">Online • Context Loaded</p>
                </div>
            </div>
            <button class="text-gray-500 hover:text-white transition"><i class="fa-solid fa-ellipsis-vertical"></i></button>
          </div>
           
          <div id="chat-panel" class="flex-1 overflow-y-auto p-4 space-y-4 custom-scrollbar bg-[#0d0d0f]">
             <div class="flex flex-col gap-1 items-start max-w-[85%] fade-up">
                <div class="p-3.5 rounded-2xl chat-bubble-ai text-xs text-gray-300 leading-relaxed shadow-md">
                   Analysis complete. I have processed the document content. I can answer specific questions regarding figures, dates, or compliance requirements.
                </div>
                <span class="text-[9px] text-gray-600 ml-1">Just now</span>
             </div>
          </div>

          <div class="p-4 bg-surface-dark border-t border-gray-800">
             <div class="relative flex items-center gap-2">
                 <input type="text" id="chat-input" class="w-full pl-4 pr-12 py-3.5 rounded-full bg-[#050505] border border-gray-700 text-sm text-white focus:outline-none focus:border-afzal-purple focus:ring-1 focus:ring-afzal-purple transition placeholder-gray-600" placeholder="Ask a question...">
                 <button id="chat-send" class="absolute right-2 top-1/2 -translate-y-1/2 w-8 h-8 flex items-center justify-center bg-afzal-purple text-white rounded-full hover:bg-afzal-blue transition shadow-lg">
                    <i class="fa-solid fa-paper-plane text-xs"></i>
                 </button>
             </div>
          </div>
          <textarea id="doc-context" class="hidden">{{ doc_context }}</textarea>
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
        const wrapper = document.createElement('div');
        wrapper.className = `flex flex-col gap-1 max-w-[85%] fade-up ${role === 'user' ? 'items-end ml-auto' : 'items-start'}`;
        
        const bubble = document.createElement('div');
        bubble.className = `p-3.5 rounded-2xl text-xs leading-relaxed shadow-md ${role === 'user' ? 'chat-bubble-user' : 'chat-bubble-ai text-gray-300'}`;
        bubble.textContent = text;

        const time = document.createElement('span');
        time.className = "text-[9px] text-gray-600 mx-1";
        time.textContent = "Just now";

        wrapper.appendChild(bubble);
        wrapper.appendChild(time);
        panel.appendChild(wrapper);
        panel.scrollTop = panel.scrollHeight;
        
        // Trigger animation
        setTimeout(() => { 
            wrapper.style.opacity = 1; 
            wrapper.style.transform = 'translateY(0)'; 
        }, 50);
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
            addMsg('assistant', "Connection interrupted. Please try again.");
        }
    }

    sendBtn.onclick = sendMessage;
    input.onkeypress = (e) => { if(e.key === 'Enter') sendMessage(); }
</script>
{COMMON_SCRIPTS}
</body>
</html>
"""

# Inject Common Parts
INDEX_HTML = INDEX_HTML.replace("{COMMON_HEAD}", COMMON_HEAD).replace("{COMMON_SCRIPTS}", COMMON_SCRIPTS)
RESULT_HTML = RESULT_HTML.replace("{COMMON_HEAD}", COMMON_HEAD).replace("{COMMON_SCRIPTS}", COMMON_SCRIPTS)


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
    abbreviations = { "Dr.": "Dr<DOT>", "Mr.": "Mr<DOT>", "Ms.": "Ms<DOT>", "Mrs.": "Mrs<DOT>", "Fig.": "Fig<DOT>", "No.": "No<DOT>", "Vol.": "Vol<DOT>", "approx.": "approx<DOT>", "vs.": "vs<DOT>", "e.g.": "e<DOT>g<DOT>", "i.e.": "i<DOT>e<DOT>" }
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
        if len(p) < 15: continue
        if re.match(r'^[0-9\.]+$', p): continue
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

POLICY_KEYWORDS = {
    "key goals": ["aim", "goal", "objective", "target", "achieve", "reduce", "increase", "coverage", "mortality", "rate", "%", "2025", "2030", "vision", "mission", "outcome", "expectancy", "eliminate"],
    "policy principles": ["principle", "equity", "universal", "right", "access", "accountability", "transparency", "inclusive", "patient-centered", "quality", "ethics", "value", "integrity", "holistic"],
    "service delivery": ["hospital", "primary care", "secondary care", "tertiary", "referral", "clinic", "health center", "wellness", "ambulance", "emergency", "drug", "diagnostic", "infrastructure", "bed", "supply chain", "logistics"],
    "prevention & promotion": ["prevent", "sanitation", "nutrition", "immunization", "vaccine", "tobacco", "alcohol", "hygiene", "awareness", "lifestyle", "pollution", "water", "screening", "diet", "exercise", "community"],
    "human resources": ["doctor", "nurse", "staff", "training", "workforce", "recruit", "medical college", "paramedic", "salary", "incentive", "capacity building", "skill", "hrh", "deployment", "specialist"],
    "financing & private sector": ["fund", "budget", "finance", "expenditure", "cost", "insurance", "private", "partnership", "ppp", "out-of-pocket", "reimbursement", "allocation", "spending", "gdp", "tax", "claim"],
    "digital health": ["digital", "technology", "data", "record", "ehr", "emr", "telemedicine", "mobile", "app", "information system", "cyber", "interoperability", "portal", "online", "software", "ai"],
    "ayush integration": ["ayush", "ayurveda", "yoga", "unani", "siddha", "homeopathy", "traditional", "naturopathy", "alternative medicine", "integrative"]
}

def score_sentence_categories(sentence: str) -> str:
    s_lower = sentence.lower()
    scores = {cat: 0 for cat in POLICY_KEYWORDS}
    for cat, keywords in POLICY_KEYWORDS.items():
        for kw in keywords:
            if kw in s_lower: scores[cat] += 2
    if '%' in s_lower or re.search(r'\b20[2-5][0-9]\b', s_lower): scores['key goals'] += 2
    best_cat = max(scores, key=scores.get)
    if scores[best_cat] == 0: return "other"
    return best_cat

# ---------------------- ML SUMMARIZER ---------------------- #

def build_tfidf(sentences: List[str]):
    return TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True).fit_transform(sentences)

def textrank_scores(sim_mat: np.ndarray, doc_len: int) -> Dict[int, float]:
    np.fill_diagonal(sim_mat, 0.0)
    G = nx.from_numpy_array(sim_mat)
    try:
        pr = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-4)
    except:
        pr = {i: 0.0 for i in range(sim_mat.shape[0])}
    return pr

def summarize_extractive(raw_text: str, length_choice: str = "medium"):
    cleaned = normalize_whitespace(raw_text)
    sentences = sentence_split(cleaned)
    n = len(sentences)
    if n <= 3: return sentences, {}
    
    if length_choice == "short": target_sentences = 12
    elif length_choice == "long": target_sentences = 90
    else: target_sentences = 45
    if target_sentences > n: target_sentences = n
    
    tfidf_mat = build_tfidf(sentences)
    sim_mat = cosine_similarity(tfidf_mat)
    tr_scores = textrank_scores(sim_mat, n)
    
    selected_idxs = []
    if target_sentences > 0:
        bucket_size = n / target_sentences
        for i in range(target_sentences):
            start_idx = int(i * bucket_size)
            end_idx = int((i + 1) * bucket_size)
            if i == target_sentences - 1: end_idx = n
            best_in_bucket_idx = -1
            best_in_bucket_score = -1.0
            if start_idx >= end_idx:
                if start_idx < n: selected_idxs.append(start_idx)
                continue
            for j in range(start_idx, end_idx):
                score = tr_scores.get(j, 0.0)
                if score > best_in_bucket_score:
                    best_in_bucket_score = score
                    best_in_bucket_idx = j
            if best_in_bucket_idx != -1: selected_idxs.append(best_in_bucket_idx)
    selected_idxs.sort()
    final_sents = [sentences[i] for i in selected_idxs]
    return final_sents, {}

def build_structured_summary(summary_sentences: List[str], tone: str):
    if tone == "easy":
        text_block = " ".join(summary_sentences)
        text_block = re.sub(r'\([^)]*\)', '', text_block)
        text_block = re.sub(r'\s+', ' ', text_block)
        return {
            "abstract": summary_sentences[0] if summary_sentences else "No abstract generated.",
            "sections": [],
            "simple_text": text_block,
            "category_counts": {}
        }
    
    cat_map = defaultdict(list)
    for s in summary_sentences:
        category = score_sentence_categories(s)
        cat_map[category].append(s)
    
    section_titles = {
        "key goals": "Key Goals & Targets", "policy principles": "Policy Principles & Vision",
        "service delivery": "Healthcare Delivery Systems", "prevention & promotion": "Prevention & Wellness",
        "human resources": "Workforce (HR)", "financing & private sector": "Financing & Costs",
        "digital health": "Digital Interventions", "ayush integration": "AYUSH / Traditional Medicine",
        "other": "Other Key Observations"
    }
    
    sections = []
    def clean_bullet(txt): return re.sub(r'\[[\d,\-\s]+\]', '', txt).strip()

    for k, title in section_titles.items():
        if cat_map[k]:
            unique = list(dict.fromkeys([clean_bullet(s) for s in cat_map[k]]))
            unique = [u for u in unique if len(u) > 10]
            if unique: sections.append({"title": title, "bullets": unique})
            
    abstract_candidates = cat_map['key goals'] + cat_map['policy principles'] + summary_sentences
    abstract_cleaned = [clean_bullet(s) for s in abstract_candidates]
    abstract_cleaned = [s for s in abstract_cleaned if len(s) > 10]
    abstract = " ".join(list(dict.fromkeys(abstract_cleaned))[:3])
    
    return {
        "abstract": abstract,
        "sections": sections,
        "simple_text": None,
        "category_counts": {k: len(v) for k, v in cat_map.items()}
    }

# ---------------------- TRANSLATION UTILS ---------------------- #

def translate_content(structured_data, target_lang):
    if not HAS_TRANSLATE or target_lang == 'en':
        return structured_data
    
    try:
        translator = Translator()
        
        # Helper to translate list or string
        def safe_translate(text):
            if not text: return text
            try:
                return translator.translate(text, dest=target_lang).text
            except:
                return text

        # Translate Abstract
        if structured_data.get("abstract"):
            structured_data["abstract"] = safe_translate(structured_data["abstract"])
            
        # Translate Simple Text
        if structured_data.get("simple_text"):
            structured_data["simple_text"] = safe_translate(structured_data["simple_text"])
            
        # Translate Sections
        if structured_data.get("sections"):
            for sec in structured_data["sections"]:
                sec["title"] = safe_translate(sec["title"])
                new_bullets = []
                for b in sec["bullets"]:
                    new_bullets.append(safe_translate(b))
                sec["bullets"] = new_bullets
                
    except Exception as e:
        print(f"Translation failed: {e}")
        
    return structured_data

# ---------------------- GEMINI IMAGE PROCESSING ---------------------- #

def process_images_with_gemini(image_paths: List[str]):
    if not GEMINI_API_KEY:
        return None, "Gemini API Key missing."

    try:
        model = genai.GenerativeModel("gemini-2.5-flash") # Updated to 1.5 Flash for multi-modal
        
        content_parts = ["Analyze these policy document images. Extract text and create a structured summary. Output strict JSON: { \"extracted_text\": \"...\", \"summary_structure\": { \"abstract\": \"...\", \"sections\": [ { \"title\": \"...\", \"bullets\": [\"...\"] } ] } }"]
        
        for path in image_paths:
            img = Image.open(path)
            content_parts.append(img)
            
        response = model.generate_content(content_parts)
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
    
    # Needs a font that supports unicode for translations, but standard ReportLab only supports basic latin.
    # For this demo, we stick to Helvetica, which means non-latin chars might look broken in PDF.
    # In a real app, register a TTF font like NotoSans.
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, title)
    y -= 30
    
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Abstract")
    y -= 15
    
    c.setFont("Helvetica", 10)
    if abstract:
        # Simple cleanup for PDF generation to avoid crashing on weird chars if font missing
        safe_abstract = abstract.encode('latin-1', 'replace').decode('latin-1')
        lines = simpleSplit(safe_abstract, "Helvetica", 10, width - 2*margin)
        for line in lines:
            c.drawString(margin, y, line)
            y -= 12
    y -= 10
    
    if simple_text:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y, "Full Summary")
        y -= 15
        c.setFont("Helvetica", 10)
        safe_text = simple_text.encode('latin-1', 'replace').decode('latin-1')
        lines = simpleSplit(safe_text, "Helvetica", 10, width - 2*margin)
        for line in lines:
            if y < 50: c.showPage(); y = height - margin
            c.drawString(margin, y, line)
            y -= 12
    else:
        for sec in sections:
            if y < 100: c.showPage(); y = height - margin
            c.setFont("Helvetica-Bold", 11)
            safe_title = sec["title"].encode('latin-1', 'replace').decode('latin-1')
            c.drawString(margin, y, safe_title)
            y -= 15
            c.setFont("Helvetica", 10)
            for b in sec["bullets"]:
                safe_bullet = b.encode('latin-1', 'replace').decode('latin-1')
                blines = simpleSplit(f"• {safe_bullet}", "Helvetica", 10, width - 2*margin)
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
    files = request.files.getlist("file")
    if not files or files[0].filename == "":
        abort(400, "No file uploaded")
    
    # 1. Determine Input Type (Multiple Images vs Single PDF/Text)
    is_multi_image = False
    valid_images = []
    first_file = files[0]
    
    # Check if user uploaded multiple files (assuming images)
    if len(files) > 1:
        # Strict check: all must be images
        for f in files:
            fname = secure_filename(f.filename).lower()
            if not fname.endswith(('.png', '.jpg', '.jpeg', '.webp')):
                abort(400, "Multiple file upload is only allowed for Images.")
        is_multi_image = True
    else:
        # Single file check
        fname = secure_filename(first_file.filename).lower()
        if fname.endswith(('.png', '.jpg', '.jpeg', '.webp')):
            is_multi_image = True # Treat single image as list of 1
        
    # 2. Process Files
    uid = uuid.uuid4().hex
    stored_name = "" # For display URL
    
    structured_data = {}
    orig_text = ""
    orig_type = "unknown"
    used_model = "ml"
    
    if is_multi_image:
        orig_type = "image"
        used_model = "gemini"
        image_paths = []
        
        for f in files:
            fname = secure_filename(f.filename)
            s_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{uid}_{fname}")
            f.save(s_path)
            image_paths.append(s_path)
            
        stored_name = f"{uid}_{secure_filename(files[0].filename)}" # Just use first for preview url logic
        
        gemini_data, err = process_images_with_gemini(image_paths)
        if err or not gemini_data:
            abort(500, f"Gemini Processing Failed: {err}")
            
        orig_text = gemini_data.get("extracted_text", "")
        structured_data = gemini_data.get("summary_structure", {})
        
    else:
        # PDF / Text Logic
        f = first_file
        fname = secure_filename(f.filename)
        stored_name = f"{uid}_{fname}"
        s_path = os.path.join(app.config["UPLOAD_FOLDER"], stored_name)
        f.save(s_path)
        used_model = "ml"
        
        with open(s_path, "rb") as f_in:
            raw_bytes = f_in.read()
            
        if fname.lower().endswith(".pdf"):
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

    # 3. Translation
    target_lang = request.form.get("language", "en")
    lang_names = {'en': 'English', 'hi': 'Hindi', 'te': 'Telugu', 'ta': 'Tamil', 'bn': 'Bengali', 'es': 'Spanish', 'fr': 'French'}
    
    if target_lang != 'en':
        structured_data = translate_content(structured_data, target_lang)

    # Defaults
    if "abstract" not in structured_data: structured_data["abstract"] = "Summary not generated."
    if "sections" not in structured_data: structured_data["sections"] = []

    # 4. Generate PDF
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
        used_model=used_model,
        language=target_lang,
        language_name=lang_names.get(target_lang, target_lang)
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
