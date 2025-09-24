import pandas as pd

df = pd.read_csv('/Users/Shared/candomble_sentiment_analysis/data/raw/vader_instagram_comments.csv', sep=';')

sample_rows = df.sample(200, random_state=42)

sample_en = sample_rows["comment_en"]
sample_pt = sample_rows["comment_pt"]

manual_en = pd.DataFrame({
    "comment_en": sample_en,
    "evaluation": [" " for _ in range(len(sample_en))],
})

manual_en.to_excel("/Users/Shared/candomble_sentiment_analysis/data/processed/manual_labels_en.xlsx", index=False)

manual_pt = pd.DataFrame({
    "comment_en": sample_pt,
    "evaluation": [" " for _ in range(len(sample_pt))],
})

manual_pt.to_excel("/Users/Shared/candomble_sentiment_analysis/data/processed/manual_labels_pt.xlsx", index=False)