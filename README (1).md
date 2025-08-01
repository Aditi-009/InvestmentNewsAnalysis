
# 📘 Investment Bank News Analysis Pipeline

This project processes financial news to extract, filter, and analyze sentiment about investment banks.

## 🔰 Features
- Named entity and dependency-based extraction
- Relevance filtering (bank as subject/object)
- Accuracy scoring (rule-based & GPT)
- Sentiment analysis via OpenAI GPT-3.5
- Model comparison (spaCy sm/md/trf)

## 📁 Structure
```
├── pipeinvest.py
├── bankllm.py
├── task24.py
├── setup.py
├── sample_investment_news.csv
└── .env
```

## ⚙️ Setup
```bash
python setup.py
```

## 🧠 Core Functions

- `task24.py`: Entity extraction, relevance, rule-based accuracy
- `bankllm.py`: GPT sentiment, fallback sentiment, LLM accuracy scoring
- `pipeinvest.py`: Pipeline runner and model comparison

## 🧪 Running the Pipeline

```bash
python pipeinvest.py                        # Default (trf model + new rule)
python pipeinvest.py --compare              # Compare all models/rules
python pipeinvest.py --single md false      # Run specific config
```

## 📊 Output Columns

- `text`, `subject`, `object`, `relevant`, `accuracy_AP`, `Accuracy SA`
- `target_sentiment_score`, `target_sentiment`, `target_bank`
- `model_used`, `new_rule_used`

## 📦 Dependencies

- Python 3.8+
- `spaCy`, `openai`, `pandas`, `dotenv`, `transformers`

## 💡 Notes

- Use `en_core_web_trf` for best accuracy (needs GPU & RAM)
- GPT sentiment uses OpenAI API key from `.env`
