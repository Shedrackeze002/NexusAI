# =====================================================================
# CONFIGURATION - Environment variables, paths, and shared imports
# =====================================================================
# This cell loads the .env file and exposes all API keys, directory
# paths, and standard-library imports used by downstream cells.
# It runs first so every later cell can reference these constants.
# =====================================================================
import os
import sys
import datetime
import time
import json
import logging
import urllib.parse
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Literal
from dotenv import load_dotenv

# Load environment variables from the .env file in the project root.
# This file contains API keys (Gemini, Slack, LangSmith) and email
# credentials.  It is NOT committed to version control.
load_dotenv()

# --- API Keys ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")

# --- Email Credentials ---
# EMAIL_PASSWORD should be a Gmail App Password (not the account password).
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS", "shedrackcmu@gmail.com")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")  # App Password
EMAIL_USER_TARGET = os.getenv("EMAIL_USER_TARGET", "shedrackeze002@gmail.com")

# --- Paths (Adjusted for Notebook Environment) ---
# The notebook runs from the project root which contains Templates/ and outputs/
BASE_DIR = os.getcwd()
TEMPLATES_DIR = os.path.join(BASE_DIR, "Templates")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Startup Diagnostics ---
print(f"Configuration Loaded.")
print(f"Templates Dir: {TEMPLATES_DIR}")
print(f"Outputs Dir: {OUTPUT_DIR}")
if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not found in .env")

