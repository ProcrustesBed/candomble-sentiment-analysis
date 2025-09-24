from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from leia import SentimentIntensityAnalyzer as LeiaAnalyzer
import pandas as pd

analyzer_en = SentimentIntensityAnalyzer()
analyzer_pt = LeiaAnalyzer()

# VADER Emoji-Updates
vader_emoji_direct = {
    "✨": 1.0,      # dizzy
    "💔": -4.0,     # broken heart
    "🔥": 1.5,      # fire
    "🤍": 2.0,      # white heart
}

# VADER Emoji-Updates hinzufügen
for emoji, value in vader_emoji_direct.items():
    analyzer_en.lexicon[emoji] = value
    analyzer_en.lexicon["🔥"] = 1.5
    analyzer_en.lexicon["fire"] = 1.5
    analyzer_en.lexicon["💔"] = -4.0
    analyzer_en.lexicon["broken_heart"] = -4.0

# LeIA Emoji-zu-Portugiesisch Mapping
emoji_to_portuguese = {
    '❤️': 'amor',
    '💖': 'amor', 
    '💕': 'carinho',
    '😍': 'apaixonado',
    '😊': 'feliz',
    '🥰': 'apaixonado',
    '🙏': 'oracao',
    '🙌': 'celebracao',
    '✨': 'brilhante',
    '🌟': 'estrela',
    '🎉': 'festa',
    '🎊': 'celebracao',
    '🏹': 'oxossi',
    '⚡': 'xango',
    '🌊': 'iemanja',
    '🔥': 'fogo',
    '🐍': 'serpente',
    '🕊️': 'pomba',
    '😢': 'triste',
    '😡': 'raiva',
    '💔': 'coracao_partido',
    '👑': 'coroa',
    '🐚': 'concha',
    '💎': 'pedra_preciosa',
    '🌈': 'arco_iris'
}

# LeIA Portugiesische Sentiment-Werte
portuguese_sentiment = {
    'amor': 1.5,
    'carinho': 1.5,
    'apaixonado': 1.8,
    'feliz': 1.3,
    'oracao': 1.0,
    'celebracao': 1.5,
    'brilhante': 1.8,
    'estrela': 1.2,
    'festa': 1.2,
    'oxossi': 2.0,
    'xango': 2.0,
    'iemanja': 2.0,
    'fogo': 1.5,
    'serpente': 2.0,
    'pomba': 2.0,
    'triste': -2.1,
    'raiva': -2.8,
    'coracao_partido': -4.0,
    'coroa': 1.5,
    'concha': 2.0,
    'pedra_preciosa': 1.5,
    'arco_iris': 2.0
}

# Update des Wörterbuchs mit Yoruba- und Candomblé-Begriffen
yoruba_terms = {
    "axé": 2.5,
    "axe": 2.5,
    "asé": 2.5,
    "àsé": 2.5,
    "asè": 2.5,
    "ase": 2.5,
    "àse": 2.5,
    "orixás": 3.0,
    "orixas": 3.0,
    "orixá": 3.0,
    "oxalá": 3.0,
    "iemanjá": 3.0,
    "xangô": 3.0,
    "xango": 3.0,
    "sàngó": 3.0,
    "kaô": 2.0,
    "kaó": 2.0,
    "kao": 2.0,
    "kabiesisi": 2.0,
    "kabiesi'lè": 2.0,
    "kàbiésilé": 2.0,
    "kabiesile": 2.0,
    "kabiesi": 2.0,
    "kabecile": 2.0,
    "nanã": 3.0,
    "oxóssi": 3.0,
    "arolê": 2.0,
    "okê arôlê": 2.0,
    "okê arolê": 2.0,
    "okê aro": 2.0,
    "okê arô": 2.0,
    "oke aro": 2.0,
    "okê aró": 2.0,
    "oxum": 3.0,
    "iansã": 3.0,
    "ogum": 3.0,
    "omolu": 3.0,
    "atôtô": 2.0,
    "atoto": 2.0,
    "atotô": 2.0,
    "atotó": 2.0,
    "exú": 2.0,
    "laroyê": 2.0,
    "laroye": 2.0,
    "terreiro": 2.0,
    "orixá": 2.5,
    "ebó": 1.5,
    "oferenda": 2.0,
    "de santo": 1.5,
    "ìyá": 1.5,
    "iya": 1.5,
    "iyá": 1.5,
    "yalorixá": 1.5,
    "ya": 1.5,
    "yá": 1.5,
    "babalorixá": 2.0,
    "adupé": 1.5,
    "motumbá": 1.5,
    "terreiro": 1.0,
    "terreiros": 1.0,
}

for term, value in yoruba_terms.items():
    analyzer_en.lexicon[term] = value
    analyzer_pt.lexicon[term] = value

for term, value in portuguese_sentiment.items():
    analyzer_pt.lexicon[term] = value

# Emoji-Preprocessing für LeIA
def preprocess_emojis_for_leia(text):
    if pd.isna(text):
        return text
    processed = str(text)
    for emoji, portuguese in emoji_to_portuguese.items():
        processed = processed.replace(emoji, f' {portuguese} ')
    return processed

# Analyse der auf Englisch übersetzten Kommentare
def analyze_en(text):
    df_analyzed = analyzer_en.polarity_scores(text)['compound'] 
    return df_analyzed

# Analyse der portugiesischen Originalkommentare (mit Emoji-Preprocessing)
def analyze_pt(text):
    processed_text = preprocess_emojis_for_leia(text)
    df_analyzed = analyzer_pt.polarity_scores(processed_text)['compound'] 
    return df_analyzed

# Analyse aller Kommentare
def analyze_all(df, use_llm=True):
    df["sentiment_vader_en"] = df["comment_en"].apply(analyze_en)
    df["sentiment_vader_pt"] = df["comment_pt"].apply(analyze_pt)
    
    # Analyse mit LLM-Sentiment (optional)
    if use_llm:
        from src.llm_sentiment import analyze_with_llm
        df["sentiment_llm"] = df["comment_pt"].apply(analyze_with_llm)

    return df