#  Mindful — Bilingual Mental Health Support Chatbot

A domain-specific AI chatbot for mental health support, supporting both **English and Arabic** users. Built using TF-IDF retrieval and fine-tuned DistilGPT2.

##  Deployed Chatbot ui
 [https://huggingface.co/spaces/Evanice4/mindful-chatbot](https://huggingface.co/spaces/Evanice4/mindful-chatbot)

##  Features
-  Bilingual support — auto-detects English and Arabic
-  TF-IDF + Cosine Similarity retrieval engine (15,000 features, bigrams)
-  Fine-tuned DistilGPT2 LLM on mental health conversations
-  Intent classification — Linear SVM (95.51% accuracy)
-  Crisis detection with emergency helpline resources
-  Gradio UI deployed on Hugging Face Spaces

##  Dataset
- **Name:** combined_english_arabic_dataset.csv
- **Size:** 12,137 rows × 5 columns
- **Languages:** English (90.5%), Arabic (9.4%)
- **Topics:** Anxiety, depression, stress, sleep, relationships, general wellbeing

##  Model
- **Retrieval:** TF-IDF Vectoriser (sklearn) + Cosine Similarity
- **Generative LLM:** DistilGPT2 (82M parameters) fine-tuned via Hugging Face Trainer API
- **Fine-tuning config:** LR=2e-5, Epochs=3, Batch=4, Gradient Accumulation=2
- **Evaluation metric:** Perplexity (lower = better)

##  How to Run
1. Open `mental_health_chatbot.ipynb` in Google Colab
2. Go to **Runtime → Change Runtime Type → GPU**
3. Run all cells from top to bottom
4. The Gradio UI will launch automatically

##  Requirements
```
gradio
scikit-learn
nltk
langdetect
transformers
torch
wordcloud
matplotlib
seaborn
pandas
numpy
plotly
```

## Results

| Model | Metric | Score |
|---|---|---|
| Linear SVM | Intent Accuracy | 95.51% |
| Logistic Regression | Intent Accuracy | 85.83% |
| Naive Bayes | Intent Accuracy | 70.05% |
| DistilGPT2 (fine-tuned) | Perplexity | Lower than base  |

## Author
Nice Eva Karabaranga | ALU | February 2026