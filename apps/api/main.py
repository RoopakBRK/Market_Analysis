import os
from python_dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')
groq_fallback_api_key = os.getenv('GROQ_FALLBACK_API_KEY')

def main():
    print("Market Intelligence API Server")

if __name__ == "__main__":
    main()
