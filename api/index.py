"""
api/index.py — Vercel Serverless Entrypoint for ExamGuard AI FastAPI
"""

import sys
import os
from pathlib import Path

# Add project root and 'new website' to path so all imports resolve seamlessly
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "new website"))

from server import app

# Vercel natively supports ASGI FastAPI directly via `app` variable
