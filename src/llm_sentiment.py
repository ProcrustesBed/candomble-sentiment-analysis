from openai import OpenAI
import pandas as pd
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Kommentare laden
df = pd.read_csv("/Users/Shared/candomble_sentiment_analysis/data/raw/vader_instagram_comments.csv", sep=";")

# Analysefunktion mit starkem Prompt
def analyze_with_llm(text):
    if pd.isna(text) or text.strip() == "":
        return 0.0
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",   # günstig & stark genug
            temperature=0,          # deterministisch für Reproduzierbarkeit
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a sentiment analysis tool similar to VADER. "
                        "Your only task is to return a sentiment score between -1 (very negative) and +1 (very positive). "
                        "The output must always be a single number with a dot as decimal separator."
                    )
                },
                {
                    "role": "user",
                    "content": f"""
Analyze the following Instagram comment. 
Context: It comes from the Afro-Brazilian religious community (Candomblé) on Instagram. 
You must consider that some terms and emojis have specific cultural meanings.

Rules:
- Yoruba and Candomblé terms (e.g., "Axé", "Asé", "Oxalá", "Xangô", "Oxóssi", "Iemanjá", "Ogum", "Nanã", "Iansã", "Exú", "Omolu") 
  are always positive or sacred, so they should increase the sentiment score.
- Greetings like "Motumbá", "Laroyê", "Atotô", "Kabiesi" are positive.
- Emojis like ❤️ 💕 💖 💜 🤍 🕊️ 🏹 ⚡ 🌊 🐚 🔥 👑 ✨ 🎊 🙌 🎉 should be treated as positive. 
- Emojis like 💔 😢 😡 are negative.
- If the comment contains ONLY emojis, interpret them the same way (e.g., "❤️❤️❤️" is strongly positive).
- Neutral words or factual comments should remain close to 0.0.
- Always return a single number between -1 and +1 (example: 0.85).

Comment: {text}
"""
                }
            ]
        )
        # Antwort extrahieren und in float umwandeln
        sentiment_score = float(response.choices[0].message.content.strip())
        return sentiment_score
    except Exception as e:
        print("Error:", e)
        return 0.0

print("✅ LLM-Sentimentanalyse abgeschlossen!")