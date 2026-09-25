import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("API key not found")
    exit()

model = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=api_key
)

try:
    response = model.invoke("say hello to my RAG project in one sentence")
    print(response.content)

except Exception as e:
    print("Error while communicating with Gemini:")
    print(e)