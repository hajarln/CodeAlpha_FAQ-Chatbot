# 🍏 Vitality Healthy Core AI — FAQ Chatbot

Vitality Healthy Core AI is an intelligent chatbot designed to provide instant answers to questions regarding nutrition, fitness, sleep, and a balanced lifestyle. This project combines Natural Language Processing (NLP), text vectorization, and a modern user interface to deliver a smooth and interactive user experience.

---

## Key Features

- **Knowledge Base (FAQ):** Integration of 30 in-depth, specialized question-and-answer pairs focused on health and wellness.
- **Advanced NLP Preprocessing:** Automated text cleaning including lowercasing, punctuation removal, stop words filtering, and lemmatization powered by **SpaCy**.
- **Smart Matching Engine:** Uses **TF-IDF** vectorization combined with **Cosine Similarity** computation to understand user intent and accurately map queries to the best matching answer.
- **Modern Chat Interface:** A dynamic and responsive UI built with **Streamlit**, enhanced with custom CSS styling (clean light mode, hover effects, and smooth visual gradients).
- **Session History Management:** Maintains chat context throughout the user session using `st.session_state`.

---

## Technologies Used

- **Language:** Python 3
- **User Interface:** Streamlit
- **Natural Language Processing (NLP):** SpaCy (`en_core_web_sm`)
- **Machine Learning & Vectorization:** Scikit-Learn (`TfidfVectorizer`, `cosine_similarity`)

---

## Installation & Setup

### 1. Clone the project
```bash
git clone [https://github.com/hajarln/CodeAlpha_FAQ-Chatbot.git](https://github.com/hajarln/CodeAlpha_FAQ-Chatbot.git)
cd CodeAlpha_FAQ-Chatbot