import os
import requests
# pyrefly: ignore [missing-import]
from python_dotenv import load_dotenv
load_dotenv()

groqapikey= os.getenv('GROQ_API_KEY')
groqfallbackapikey=os.getenv('GROQ_FALLBACK_API_KEY')

