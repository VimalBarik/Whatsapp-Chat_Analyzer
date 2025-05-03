#  WhatsApp Chat Analyzer

This project is a full pipeline for analyzing WhatsApp chat exports. It processes conversations to extract insights like sentiment trends, personality traits, and conversation topics, with rich visualizations.

---

##  Features

-  Parses raw WhatsApp chat exports
-  Cleans and structures messages
-  Performs sentiment analysis
-  Estimates Big Five personality traits
-  Extracts key topics with topic modeling
-  Visualizes insights using Seaborn & Matplotlib

> **Currently supports English-language chats only.**

---

##  How to Use

1. **Export your WhatsApp chat** (without media) as a `.txt` file  
   - WhatsApp → Chat → More → Export Chat → Choose “Without Media”
2. **Rename the file** to: `chat.txt`
3. **Place it in the root directory** (same level as `whatsapp_chat_analyzer.ipynb`)
4. **Open `whatsapp_chat_analyzer.ipynb`** and run the cells one by one

>  **Do not upload personal chat data to GitHub.**  
> This project is configured to ignore any `.txt` or `.csv` files by default.

---

##  Project Structure

```
Whatsapp-Chat-Analyzer/
│
├── whatsapp_chat_analyzer.ipynb              # Main analysis notebook
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
├── .gitignore                 # Files to exclude from version control
```

---

##  Installation

Install required packages:

```bash
pip install -r requirements.txt
```

---

##  Parameters

Adjust these values at the top of the notebook:

| Parameter           | Description                          | Default |
|---------------------|--------------------------------------|---------|
| `POSITIVE_THRESHOLD` | Sentiment score threshold            | `0.05`  |
| `NEGATIVE_THRESHOLD` | Negative sentiment lower bound       | `-0.05` |
| `NUM_TOPICS`         | Number of topics for LDA             | `5`     |
| `TOP_N_WORDS`        | Words per topic                      | `15`    |
| `MIN_MESSAGE_LENGTH` | Short message cutoff                 | `3`     |
| `SAMPLE_SIZE`        | Max number of messages to sample     | `5000`  |

---

##  Output

- Sentiment charts
- Word clouds
- Personality radar plots
- Top topics and dominant message themes

---

##  Language Support

This version supports **only English**. Multilingual support is not implemented yet but could be added in future updates.

---

##  Privacy First

This repository **does not** contain any chat logs or message content. Users are expected to provide their own exported WhatsApp chats.

---

##  Requirements

```txt
streamlit==1.22.0
pandas==1.5.0
numpy==1.23.0
nltk==3.7
textblob==0.17.1
vaderSentiment==3.3.2
empath==0.89
matplotlib==3.6.0
seaborn==0.12.0
wordcloud==1.8.2.2
tqdm==4.64.0
emoji==2.2.0
scikit-learn==1.2.0

# Specific versions for BERTopic compatibility
huggingface-hub==0.4.0
sentence-transformers==2.2.2
bertopic==0.9.4
```
## Outputs

All generated plots and visualizations can be found in the [Images](./Images) folder.

##  Upcoming Features

- [ ]  Count of media files shared (total & per user)
- [ ]  Count of links shared (total & per user)
- [ ]  Emoji usage stats (total & per user)
- [ ]  User response time analysis
- [ ]  Words used per user
- [ ]  Most commonly used word
- [ ]  Most frequently used phrases (n-grams)
- [ ]  Unique words per user
- [ ]  Longest message sender
- [ ]  Word clouds (overall & per user)
- [ ]  Messaging streaks (consecutive active days)

