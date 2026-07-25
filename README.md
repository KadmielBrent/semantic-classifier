# 🇵🇭 Semantic Frame Classifier

**Live Demo:** [Click here to open the app](https://semantic-classifier-8gqzd5njprfgrreoktt3bu.streamlit.app)

BS Data Analytics Thesis Proposed System | University of the Cordilleras

## 📊 About

A Filipino Fake News Frame Classifier that detects semantic frames in news articles using fine-tuned BERT models.

## 🚀 Features

- 🔴 **Tagalog Model (FNF1)** - Analyzes Filipino news articles
- 🔵 **English Model (FNF2)** - Analyzes English news articles  
- 🎯 **8 Semantic Frames** - Political Conflict, Scandal/Corruption, Conspiracy, etc.
- 📊 **Interactive UI** - Simple toggle between languages and adjust thresholds

## 📁 Model Repositories

- [Tagalog Model](https://huggingface.co/KadmielBrent/tagalog_model)
- [English Model](https://huggingface.co/KadmielBrent/english_model)

## 📝 How to Use

1. Select your preferred language (Tagalog or English)
2. Paste or type a news article
3. Click "Analyze"
4. View detected semantic frames with confidence scores

## 🧠 Semantic Frames Detected

| Frame | Description |
|-------|-------------|
| Political Conflict | Binary struggle between opposing political groups |
| Scandal/Corruption | Wrongdoing, unethical behavior, or violations |
| Conspiracy | Secret plots by influential figures |
| Victimization | Powerful figures as misunderstood victims |
| Heroism | Protagonists as saviors or champions |
| Threat/Danger | Risks to social order, security, or stability |
| Economic Impact | Financial consequences and national wealth |
| Nationalism/Patriotism | Appeals to national identity |

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Models:** Fine-tuned BERT (multilingual-cased)
- **Deployment:** Streamlit Cloud
- **Model Hosting:** Hugging Face

## 🏗️ Project Structure
