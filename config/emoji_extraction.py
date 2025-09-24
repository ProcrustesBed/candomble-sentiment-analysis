import pandas as pd
import regex as re

df = pd.read_csv('/Users/Shared/candomble_sentiment_analysis/data/raw/vader_instagram_comments.csv', sep=';')

original_comments = df['comment_pt'].tolist()

all_emojis = set()

for comment in original_comments:
    emojis_found = re.findall(r'\p{Emoji}', str(comment))
    all_emojis.update(emojis_found)

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

emoji_values = {}

for emoji in all_emojis:
    score = analyzer.polarity_scores(emoji)['compound']
    emoji_values[emoji] = score

print(emoji_values)