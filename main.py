import pandas as pd
from src.sentiment_analysis import analyze_all
from src.llm_sentiment import analyze_with_llm
from src.visualization import sentiment_plotter

def main():
    df = pd.read_csv('/Users/Shared/candomble_sentiment_analysis/data/raw/vader_instagram_comments.csv', sep=';')

    df = analyze_all(df, use_llm=True)

    df.to_csv('/Users/Shared/candomble_sentiment_analysis/data/processed/vader_leia_instagram_comments_final.csv', sep=';', index=False, decimal = ",")

    # Durchschnittswerte der Analyse berechnen
    mean_en = round(df["sentiment_vader_en"].mean(), 2)
    mean_pt = round(df["sentiment_vader_pt"].mean(), 2)
    mean_llm = round(df["sentiment_llm"].mean(), 2)

    mean_en_purged = round(df[df["sentiment_vader_en"] != 0]["sentiment_vader_en"].mean(), 2)
    mean_pt_purged = round(df[df["sentiment_vader_pt"] != 0]["sentiment_vader_pt"].mean(), 2)
    mean_llm_purged = round(df[df["sentiment_llm"] != 0]["sentiment_llm"].mean(), 2)

    # Standardabweichung berechnen
    std_en = round(df["sentiment_vader_en"].std(), 2)
    std_pt = round(df["sentiment_vader_pt"].std(), 2)
    std_llm = round(df["sentiment_llm"].std(), 2)

    std_en_purged = round(df[df["sentiment_vader_en"] != 0]["sentiment_vader_en"].std(), 2)
    std_pt_purged = round(df[df["sentiment_vader_pt"] != 0]["sentiment_vader_pt"].std(), 2)
    std_llm_purged = round(df[df["sentiment_llm"] != 0]["sentiment_llm"].std(), 2)

    averages = pd.DataFrame({
        "metric": ["mean_vader", "mean_vader_purged", "std_vader", "std_vader_purged", "mean_leia", "mean_leia_purged", "std_leia", "std_leia_purged", "mean_llm", "mean_llm_purged", "std_llm", "std_llm_purged"],
        "value": [mean_en, mean_en_purged, std_en, std_en_purged, mean_pt, mean_pt_purged, std_pt, std_pt_purged, mean_llm, mean_llm_purged, std_llm, std_llm_purged]
        })

    averages.to_csv("/Users/Shared/candomble_sentiment_analysis/data/processed/sentiment_averages_3.csv", index=False, sep=";")

    print("✅ Analyse abgeschlossen und Ergebnisse gespeichert.")

    plot = sentiment_plotter(averages, file_path="/Users/Shared/candomble_sentiment_analysis/data/processed/sentiment_analysis_plot.png")
    
    print("✅ Plot gespeichert.")

    return

if __name__ == "__main__":
    main()