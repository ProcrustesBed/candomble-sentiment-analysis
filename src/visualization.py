import matplotlib.pyplot as plt
import numpy as np

def sentiment_plotter(averages, file_path=None):
    
    def get_val(metric_name):
        return averages.loc[averages["metric"] == metric_name, "value"].iloc[0]
        
    mean_en = get_val("mean_vader")
    mean_pt = get_val("mean_leia")
    mean_llm = get_val("mean_llm")
    
    mean_en_purged = get_val("mean_vader_purged")
    mean_pt_purged = get_val("mean_leia_purged")
    mean_llm_purged = get_val("mean_llm_purged")
    
    std_en = get_val("std_vader")
    std_pt = get_val("std_leia")
    std_llm = get_val("std_llm")
    
    std_en_purged = get_val("std_vader_purged")
    std_pt_purged = get_val("std_leia_purged")
    std_llm_purged = get_val("std_llm_purged")

    models = ("VADER (EN)", "LeIA (PT)", "LLM")
    
    means_all = [mean_en, mean_pt, mean_llm]
    means_purged = [mean_en_purged, mean_pt_purged, mean_llm_purged]
    
    stds_all = [std_en, std_pt, std_llm]
    stds_purged = [std_en_purged, std_pt_purged, std_llm_purged]
    
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(layout='constrained', figsize=(10, 6))

    bars1 = ax.bar(x - width/2, means_all, width, yerr=stds_all, 
                   capsize=5, label='Alle Kommentare', alpha=0.8)
    bars2 = ax.bar(x + width/2, means_purged, width, yerr=stds_purged, 
                   capsize=5, label='Ohne Null-Werte', alpha=0.8)

    ax.bar_label(bars1, padding=3, fmt='%.2f')
    ax.bar_label(bars2, padding=3, fmt='%.2f')

    ax.set_ylabel('Sentiment-Score')
    ax.set_title('Durchschnittliche Sentiment-Werte der drei Analyse-Tools')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, 1.0)

    if file_path:
        plt.savefig(file_path, bbox_inches='tight')

    plt.show()