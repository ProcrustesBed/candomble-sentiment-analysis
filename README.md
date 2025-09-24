Candomblé Instagram Sentiment Analysis

This project analyzes comments on Instagram pages of Candomblé terreiros.  
The goal is to use sentiment analysis tools (VADER, LeiA, optionally LLMs) to find out  
how public interaction is represented in the comments.

Structure

- `src/` – Main code (`sentiment_analysis.py`, `main.py`, `llm_sentiment.py`)
- `data/example/` – Sample file with fake comments for demonstration purposes  
  *(Original data is not included for privacy reasons)*  
- `requirements.txt` – Required Python libraries  
- `.gitignore` – Ensures that sensitive data is not uploaded  

Usage

1. Clone the repository:
```bash
  git clone https://github.com/<your-username>/<repo-name>.git
  cd <repo-name>

2. Install dependencies:
pip install -r requirements.txt

3. Run example:
python main.py

Note:
Original data (comments) are not included.

Only a small sample file is available at data/example/example_comments.csv.

The project is intended for research purposes as part of a master's thesis.