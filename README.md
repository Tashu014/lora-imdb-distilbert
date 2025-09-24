---
base_model: distilbert-base-uncased
library_name: peft
tags:
- base_model:adapter:distilbert-base-uncased
- lora
- transformers
---

# LoRA Fine-Tuning Practice on IMDB with DistilBERT

This repository contains a **practice project** demonstrating parameter-efficient fine-tuning (PEFT) using **LoRA adapters** on the IMDB dataset for sentiment classification.  
The main goal was to compare predictions **before and after fine-tuning** using a small dataset subset and understand how LoRA reduces trainable parameters while still adapting the model.

---

## 📌 Project Overview
- **Base Model:** `distilbert-base-uncased`  
- **Dataset:** IMDB reviews (subset for faster experimentation)  
- **Fine-Tuning Method:** LoRA (Low-Rank Adapters)  
- **Frameworks:** Hugging Face Transformers + PEFT  

---

## 🚀 How to Run

1. Clone the repo and install dependencies:
   ```bash
   git clone <your-repo-url>
   cd <your-repo>
   pip install -r requirements.txt
2. Run the training script:
   ```bash
   python main.py
3. Observe predictions before and after fine-tuning in the console.
