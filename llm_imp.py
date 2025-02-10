from dotenv import load_dotenv
import os
from llama_index.llms.gemini import Gemini
# Load environment variables from .env file
load_dotenv()




llm = Gemini(
    model="models/gemini-1.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY")
)


if __name__=="__main__":
    resp = llm.complete("Write a poem about a magic backpack")
    print(resp)