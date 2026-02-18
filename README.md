
Real-time multimodal emotion recognition system combining speech and text models with Dynamic Sliding Window tracking, confidence-weighted fusion, and acoustic-aware correction.


# 🎭 Real-Time Multimodal Emotion Recognition System  
### With Dynamic Sliding Window & Acoustic-Aware Correction

---

## 🚀 Overview

This project presents a **real-time multimodal emotion recognition system** designed to detect, interpret, and respond to user emotions using both **speech** and **text** inputs.

Unlike traditional emotion classifiers that operate on isolated predictions, this system introduces **adaptive reasoning mechanisms** to improve emotional continuity, robustness, and real-time interaction quality.

The architecture combines deep learning models with temporal context tracking, confidence-based fusion, and signal-level correction to produce more stable and human-aligned emotional interpretations.

---

## ✨ Key Capabilities

- 🎤 **Real-time Speech Emotion Recognition** (Wav2Vec 2.0)  
- 💬 **Text Emotion Classification** (BERT)  
- 🔁 **Confidence-Weighted Multimodal Fusion**  
- 🧠 **Dynamic Sliding Window (DSW)** for temporal emotion tracking  
- 🎚️ **Acoustic-Aware Emotion Correction**  
- 🤖 **Emotionally Adaptive AI Responses** (OpenChat via Ollama)  
- 🔊 **Emotion-Modulated Text-to-Speech Feedback**  
- 📊 **Emotion Logging & Confusion Matrix Evaluation**

---

## 🧠 Key Innovations

### **Dynamic Sliding Window (DSW)**
A novel adaptive temporal mechanism that:

- Tracks recent emotion predictions
- Dynamically resizes based on emotional volatility
- Stabilises low-confidence outputs
- Smooths erratic classification shifts

This allows the system to model **emotional trajectories**, rather than treating each input independently.

---

### **Confidence-Weighted Fusion**
Speech and text predictions are merged using softmax confidence scores:

\[
P_{fused} = \frac{w_s P_s + w_t P_t}{w_s + w_t}
\]

Where:

- \(P_s\), \(P_t\) = model probability outputs  
- \(w_s\), \(w_t\) = confidence weights  

---

### **Acoustic-Aware Correction**
Low-confidence speech predictions are refined using:

- Pitch mean / variance  
- RMS volume  
- Speaking rate  

This signal-level reasoning improves robustness under:

✔ Noisy input  
✔ Monotone speech  
✔ Ambiguous emotional delivery  

---

### **LLM-Based Emotional Response**
Final emotion predictions guide:

- Emotion-specific prompt templates  
- OpenChat response generation  
- Tone-adaptive feedback  

Enabling **emotionally coherent dialogue**, not just classification.

---

## 🛠️ Tech Stack

### **Languages**
Python

---

### **Machine Learning Models**
- Wav2Vec 2.0 (Speech Emotion Recognition)
- BERT (Text Emotion Classification)

---

### **Core Libraries**
PyTorch • Transformers • Librosa • NLTK • Pandas • NumPy

---

### **Speech & Audio**
SpeechRecognition • PyAudio • gTTS • pyttsx3

---

### **Visualisation & Evaluation**
Matplotlib • Seaborn

---

## ⚙️ System Workflow

1️⃣ User provides **speech or text input**  
2️⃣ Speech → Wav2Vec 2.0  
3️⃣ Text / Transcript → BERT  
4️⃣ Predictions fused via **confidence weighting**  
5️⃣ DSW updates emotional context  
6️⃣ Low-confidence → correction logic  
7️⃣ Final emotion → OpenChat prompt  
8️⃣ AI response generated  
9️⃣ Output vocalised via **TTS modulation**

---

## ▶️ Running the Project

### **1️⃣ Install Dependencies**

```bash
pip install -r requirements.txt
