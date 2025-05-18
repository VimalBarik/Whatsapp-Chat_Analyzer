## WhatsApp Chat Analyzer

This project is a full pipeline for analyzing WhatsApp chat exports. It processes conversations to extract insights like sentiment trends, personality traits, and conversation topics, with rich visualizations.

---

## Features

- **Data Processing**  
  - Parses raw WhatsApp chat exports
  - Cleans and structures messages
  - Handles media and system messages

- **Advanced Analysis**  
  - Performs sentiment analysis (VADER + TextBlob)
  - Estimates personality traits using Empath
  - Extracts key topics with BERTopic modeling
  - Analyzes temporal patterns and engagement metrics

- **Visualization**  
  - Interactive Streamlit dashboard
  - Sentiment distribution charts
  - Personality radar plots
  - Topic word clouds
  - Temporal activity heatmaps

> **Currently supports English-language chats only.**

---

## How to Use

### Web Application
1. Install dependencies: `pip install -r requirements.txt`
2. Run the app: `streamlit run app.py`
3. Upload your WhatsApp chat export (.txt) via the sidebar

### Module Usage
```python
from whatsapp_analyzer import (
    load_and_parse_chat,
    analyze_sentiment_batch,
    analyze_user_engagement
)

# Load and analyze chat
df = load_and_parse_chat("chat.txt")
results = analyze_user_engagement(df)
```

> **Privacy Notice**  
> This repository contains no chat data. All processing happens locally on your machine.

---

## Project Structure

```
whatsapp-chat-analyzer/
│
├── whatsapp_analyzer/          # Core analysis package
│   ├── __init__.py             # Package initialization
│   ├── analyzer.py             # Analysis functions
│   ├── config.py               # Configuration settings
│   ├── data_loader.py          # Chat parsing utilities
│   ├── parallel_utils.py       # Multiprocessing handlers
│   ├── text_processor.py       # Text cleaning functions
│   └── visualizer.py           # Visualization components
│
├── app.py                      # Streamlit application entry point
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
└── README.md                   # Project documentation
```

Key Notes:
- The core functionality is packaged in `whatsapp_analyzer/`
- `app.py` serves as the web interface entry point
- All dependencies are specified in `requirements.txt`


---

## Requirements

```text
# Core Analysis
pandas>=1.5.0
numpy>=1.23.0
nltk>=3.7
textblob>=0.17.1
vaderSentiment>=3.3.2
empath>=0.89
bertopic>=0.9.4

# Visualization
matplotlib>=3.6.0
seaborn>=0.12.0
wordcloud>=1.8.2
streamlit>=1.22.0

# Utilities
tqdm>=4.64.0
emoji>=2.2.0
```

Install with:  
`pip install -r requirements.txt`

---

## Output Samples


---

## Roadmap

### Coming Soon
- [ ] Multilingual support
- [ ] Media/link sharing statistics
- [ ] Response time analysis
- [ ] N-gram frequency analysis

### Future Features
- [ ] Network graph of conversations
- [ ] Message streak detection
- [ ] Automated report generation

---

## Privacy Commitment

- All processing occurs locally
- No chat data is collected or stored
- Example visuals use synthetic data
