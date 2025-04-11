# app.py
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access environment variables
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")
