import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

MOTHERDUCK_TOKEN = os.getenv("MOTHERDUCK_TOKEN", "")
MOTHERDUCK_DB = os.getenv("MOTHERDUCK_DB", "Massive")

ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://localhost:3001,https://backtester-psi.vercel.app",
    ).split(",")
    if o.strip()
]
