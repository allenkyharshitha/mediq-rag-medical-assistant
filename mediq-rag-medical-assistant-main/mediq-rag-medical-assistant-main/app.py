from flask import Flask, request, jsonify, render_template, session
import sys, os, pickle
sys.path.insert(0, '/users/PGS0411/pdhanish/.conda/envs/medical_llm/lib/python3.9/site-packages')

from groq import Groq
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from langchain.text_splitter import RecursiveCharacterTextSplitter

app = Flask(__name__)
app.secret_key = "medical_llm_secret"

# Config
GROQ_API_KEY = ""
GROQ_MODEL = "llama-3.1-8b-instant"
BASE_DIR = os.path.expanduser("~/Final project")
NODE_NAME = "p0220"  
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vector_db")
BM25_DIR = os.path.join(BASE_DIR, "bm25_index")

# Load components
client = Groq(api_key=GROQ_API_KEY)
chroma_client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
ef = embedding_functions.DefaultEmbeddingFunction()
collection = chroma_client.get_or_create_collection("medical_qa", embedding_function=ef)

with open(os.path.join(BM25_DIR, "bm25_index.pkl"), "rb") as f:
    bm25, texts = pickle.load(f)

# Safety keywords
EMERGENCY_KEYWORDS = [
    "chest pain", "heart attack", "can't breathe", "difficulty breathing",
    "stroke", "unconscious", "severe bleeding", "overdose", "suicidal",
    "seizure", "not breathing", "choking", "severe allergic"
]

def check_safety(query):
    query_lower = query.lower()
    for keyword in EMERGENCY_KEYWORDS:
        if keyword in query_lower:
            return True, keyword
    return False, None

SYMPTOM_KEYWORDS = [
    "pain", "fever", "hurt", "ache", "swelling", "bleeding", "vomit",
    "nausea", "dizzy", "tired", "fatigue", "rash", "cough", "breath",
    "symptom", "feeling", "sick", "infection", "cold", "flu"
]
def generate_followup_questions(symptom):
    prompt = f"""You are a clinical intake assistant.
A patient says: "{symptom}"

Generate exactly 3 short follow-up questions to better understand this specific symptom.
Each question must be directly relevant to "{symptom}" — do not use generic questions.
The last question should ask about any other symptoms present and allow multiple selections.
First two questions must have 3 clear single-choice options each.

Return ONLY a valid JSON array, nothing else:
[
  {{"question": "...", "options": ["...", "...", "..."]}},
  {{"question": "...", "options": ["...", "...", "..."]}},
  {{"question": "...", "options": ["...", "...", "..."], "multi": true}}
]"""
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400
    )
    import json
    text = response.choices[0].message.content.strip()
    start = text.find("[")
    end = text.rfind("]") + 1
    text = text[start:end]
    return json.loads(text)

def get_medical_response_with_context(symptom, answers):
    retrieved_docs, confidence = hybrid_search(symptom, n_results=5)
    urgency, urgency_reason = get_urgency_level(symptom, confidence)
    answers_text = "\n".join([f"- {q}: {a}" for q, a in answers.items()])
    prompt = f"""You are MediQ, a friendly medical assistant.
A patient described: "{symptom}"
Additional information collected:
{answers_text}

Give a direct, practical 2-sentence response based on what the patient told you.
Use "you" and "your". No greetings. No disclaimers. No medical jargon.
Answer:"""
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )
    return {
        "response": response.choices[0].message.content,
        "confidence": confidence,
        "urgency": urgency,
        "urgency_reason": urgency_reason,
        "sources": retrieved_docs[:2]
    }
def get_urgency_level(query, confidence):
    is_emergency, keyword = check_safety(query)
    if is_emergency:
        return "HIGH", f"Emergency keyword: {keyword}"
    
    query_lower = query.lower()
    is_symptom_query = any(word in query_lower for word in SYMPTOM_KEYWORDS)
    
    if not is_symptom_query:
        return "INFO", "This is an informational query — no medical action needed"
    elif confidence > 60:
        return "LOW", "Home care likely sufficient"
    elif confidence > 30:
        return "MODERATE", "Visit doctor if symptoms worsen"
    else:
        return "MODERATE", "Symptoms unclear — monitor and visit doctor if worsens"

def hybrid_search(query, n_results=5):
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_idx = sorted(range(len(bm25_scores)),
                          key=lambda i: bm25_scores[i], reverse=True)[:n_results]
    bm25_results = [texts[i] for i in top_bm25_idx]
    semantic_results = collection.query(query_texts=[query], n_results=n_results)
    semantic_docs = semantic_results['documents'][0]
    distance = semantic_results['distances'][0][0]
    confidence = round(max(0, (2 - distance) / 2 * 100), 1)
    combined = list(dict.fromkeys(bm25_results + semantic_docs))[:n_results]
    return combined, confidence

def get_medical_response(query, patient_profile=None):
    is_emergency, keyword = check_safety(query)
    if is_emergency:
        return {
            "response": f"EMERGENCY: You mentioned '{keyword}'. Please call 911 or go to the nearest emergency room immediately.",
            "confidence": 0,
            "urgency": "HIGH",
            "urgency_reason": f"Emergency keyword: {keyword}",
            "sources": []
        }
    retrieved_docs, confidence = hybrid_search(query, n_results=5)
    urgency, urgency_reason = get_urgency_level(query, confidence)
    profile_text = ""
    if patient_profile:
        profile_text = f"Patient: {patient_profile.get('name')}, Age: {patient_profile.get('age')}, Gender: {patient_profile.get('gender')}, Conditions: {patient_profile.get('conditions')}"
    context = "\n\n".join(retrieved_docs[:3])
    prompt = f"""You are MediQ, an AI medical assistant.
If the user sends a greeting like 'hi', 'hello', 'hey', respond with just a friendly one-line greeting.
Otherwise answer the medical question in 2-3 short sentences. First sentence: direct answer. Second sentence: one key detail.
No greetings for medical questions. No disclaimers. No bullet points. Plain sentences only.
If it is a research or scientific question, just explain what it is — do not give home care advice.

{profile_text}
Medical Context: {context}
Patient Question: {query}
Response:"""
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400
    )
    return {
        "response": response.choices[0].message.content,
        "confidence": confidence,
        "urgency": urgency,
        "urgency_reason": urgency_reason,
        "sources": retrieved_docs[:2]
    }

@app.route("/")
def index():
    return render_template("index.html", node_name=NODE_NAME)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    query = data.get("query", "")
    patient_profile = data.get("profile", None)
    result = get_medical_response(query, patient_profile)
    return jsonify(result)

@app.route("/health")
def health():
    return jsonify({"status": "ok"})
@app.route("/start_followup", methods=["POST"])
def start_followup():
    data = request.json
    symptom = data.get("symptom", "")
    is_emergency, keyword = check_safety(symptom)
    if is_emergency:
        return jsonify({
            "emergency": True,
            "response": f"EMERGENCY: You mentioned '{keyword}'. Please call 911 or go to the nearest emergency room immediately.",
            "confidence": 0,
            "urgency": "HIGH",
            "urgency_reason": f"Emergency keyword: {keyword}",
            "sources": []
        })

    # Ask LLM if this is a symptom or a direct question
    import json
    check_prompt = f"""Is this message from a patient describing a medical symptom that needs clarification?
Message: "{symptom}"
Reply with ONLY: {{"is_symptom": true}} or {{"is_symptom": false}}
Direct questions like "can I take aspirin", "yes", "no", greetings are NOT symptoms."""

    check_response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": check_prompt}],
        max_tokens=20
    )
    check_text = check_response.choices[0].message.content.strip()
    start = check_text.find("{")
    end = check_text.rfind("}") + 1
    try:
        is_symptom = json.loads(check_text[start:end]).get("is_symptom", False)
    except:
        is_symptom = False

    if not is_symptom:
        return jsonify({"emergency": False, "questions": []})

    try:
        questions = generate_followup_questions(symptom)
        return jsonify({"emergency": False, "questions": questions})
    except:
        return jsonify({"emergency": False, "questions": []})

@app.route("/answer_followup", methods=["POST"])
def answer_followup():
    data = request.json
    symptom = data.get("symptom", "")
    answers = data.get("answers", {})
    result = get_medical_response_with_context(symptom, answers)
    return jsonify(result)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)