# movie_recommender_agent.py

from transformers import pipeline
from huggingface_hub import login
from dotenv import load_dotenv
import os
import requests
import json

# ========== LOAD ENVIRONMENT VARIABLES ==========
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
OMDB_KEY = os.getenv("OMDB_KEY")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env file.")
if not OMDB_KEY:
    raise ValueError("OMDB_KEY not found in .env file.")

# ========== LOGIN & INITIALIZE CHAT MODEL ==========
login(token=HF_TOKEN, new_session=False)
chat_model = pipeline("text-generation", model="meta-llama/Llama-3.1-8B-Instruct")

# ========== FUNCTION TO QUERY OMDB ==========
def query_omdb(title):
    """Fetch movie info from OMDB by title."""
    url = f"http://www.omdbapi.com/?apikey={OMDB_KEY}&t={title}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"OMDB request failed: {response.status_code}")
    return response.json()

# ========== FUNCTION TO GENERATE QUERY AND EXPLANATION ==========
def generate_recommendations(user_movie):
    """Use the LLM to suggest related movies and explain why."""
    
    # 1. Ask the LLM to come up with an intelligent OMDB query plan
    query_prompt = f"""You are a movie expert. The user liked '{user_movie}'.
Generate 5 related movie titles that they might enjoy based on genre, tone, and popularity.
Return them as a simple comma-separated list (no extra commentary)."""
    
    llm_query_output = chat_model(query_prompt, max_new_tokens=100)[0]["generated_text"]
    print("\n[DEBUG] Model-generated movie list:\n", llm_query_output)
    
    # Extract movie titles from the text (rough split)
    recommended_titles = [t.strip() for t in llm_query_output.split(",") if len(t.strip()) > 0]
    
    # 2. Fetch basic info for each title via OMDB
    movie_data = []
    for title in recommended_titles:
        data = query_omdb(title)
        if data.get("Response") == "True":
            movie_data.append({
                "Title": data.get("Title"),
                "Year": data.get("Year"),
                "Genre": data.get("Genre"),
                "Plot": data.get("Plot")
            })
    
    # 3. Ask LLM to summarize and explain why these were picked
    explanation_prompt = f"""The user liked the movie '{user_movie}'. 
Here are some recommended movies:
{json.dumps(movie_data, indent=2)}
Explain briefly why these movies are good recommendations for them."""
    
    explanation = chat_model(explanation_prompt, max_new_tokens=300)[0]["generated_text"]
    
    # 4. Return results
    return {
        "recommendations": movie_data,
        "explanation": explanation
    }

# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    user_input = input("Enter a movie you liked: ")
    results = generate_recommendations(user_input)

    print("\n🎬 RECOMMENDED MOVIES 🎬")
    for m in results["recommendations"]:
        print(f"- {m['Title']} ({m['Year']}) | {m['Genre']}")
    print("\n💡 WHY THESE? 💡")
    print(results["explanation"])
