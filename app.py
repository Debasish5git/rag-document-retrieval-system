import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_community.document_loaders import PyPDFLoader

pdf_path = "data/sample.pdf"

try:
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    if not documents:
        print("PDF contains no pages.")
    else:
        total_characters = sum(
            len(document.page_content.strip())
            for document in documents
        )

        if total_characters == 0:
            print("PDF contains no extractable text.")
        else:
            print("PDF loaded successfully")
            print("Number of pages:", len(documents))
            print("Total extracted characters:", total_characters)

except Exception as e:
    print("Error while loading PDF:")
    print(e)