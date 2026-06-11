
#  MediQ — AI Medical Advising using LLM

###  Retrieval-Augmented Generation (RAG) for Safe & Personalized Healthcare

##  Overview

**MediQ** is an AI-powered medical advising system that leverages **Large Language Models (LLMs)** combined with a **Retrieval-Augmented Generation (RAG)** pipeline to deliver **accurate, safe, and personalized medical responses**.

Unlike traditional chatbots, MediQ integrates:

*  **Hybrid Retrieval (BM25 + Vector Search)**
*  **LLaMA 3.1 8B (via Groq API)**
*  **Safety-Aware Emergency Detection**
*  **Patient Personalization**

>  Goal: Reduce hallucinations and provide **context-grounded, reliable medical advice**

##  System Architecture

```
User Query
   ↓
Safety Detection (Emergency Check)
   ↓
Hybrid Retrieval (BM25 + ChromaDB)
   ↓
Context Construction
   ↓
LLaMA 3.1 (Groq API)
   ↓
Response + Confidence + Urgency
```

##  Key Features

###  Hybrid Retrieval Engine

* Combines:

  * **BM25 (keyword-based search)**
  * **ChromaDB (semantic vector search)**
* Improves accuracy and coverage of medical responses


###  Safety-Aware Emergency Detection

* Detects critical symptoms like:

  * Chest pain
  * Stroke
  * Breathing issues
* Classifies into 4 levels:

  * `INFO`, `LOW`, `MODERATE`, `HIGH`\
    
### Confidence Scoring

* Based on semantic distance:

```math
confidence = max(0, (2 - d)/2 * 100)
```

* Provides **explainability** for model outputs

---

### Patient Personalization

* Uses:

  * Age
  * Gender
  * Medical conditions
* Generates context-aware responses

---

###  LLM Integration

* Model: **LLaMA 3.1 8B**
* Platform: **Groq API**
* Benefits:

  *  Low latency (<2s)
  *  Context-aware reasoning
  *  Reduced hallucination via RAG

---

## Dataset

*  **BioASQ Biomedical Q&A Dataset**
* 707 medical Q&A pairs
* 🔹 898 processed chunks
* 🔹 Optimized for biomedical retrieval

---

##  Performance Highlights

| Metric              | Value |
| ------------------- | ----- |
| Accuracy            | ~87%  |
| Coverage            | ~94%  |
| Relevance Score     |  5/5 |
| Response Time       | < 2s  |
| Emergency Detection | ~95%  |

---

##  Tech Stack

### 🔹 Backend

* Flask (Python)
* Groq API (LLM inference)

### 🔹 Retrieval

* BM25 (rank-bm25)
* ChromaDB (vector database)
* Sentence Transformers (`all-MiniLM-L6-v2`)

### 🔹 Frontend

* HTML / CSS / JavaScript

---

##  Setup Instructions (OSC)

### 1. Create Environment

```bash
module load miniconda3/24.1.2-py310
conda create -n medical_llm python=3.9 -y
conda activate medical_llm
```

### 2. Install Dependencies

```bash
pip install groq chromadb rank-bm25 sentence-transformers flask
```

### 3. Add API Key

```python
GROQ_API_KEY = "your_key_here"
```

### 4. Run Application

```bash
python app.py
```

### 5. Open in Browser (OSC)

```
https://ondemand.osc.edu/rnode/<NODE_NAME>/5000/
```

##  Demo Features

*  Chat-based medical Q&A
*  Confidence score visualization
*  Urgency classification
*  Retrieved source suggestions

---

##  Limitations

*  Not a replacement for professional medical advice
*  Limited to BioASQ dataset knowledge
*  No real-time clinical data
*  English-only support

---

##  Ethical Considerations

* Patient data privacy
* Transparent AI decisions
* Bias mitigation
* Responsible medical AI usage


## Author
Bhanu Prasad Dharavathu


