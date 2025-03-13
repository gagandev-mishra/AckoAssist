# **AckoAssist 🚀**  
### **AI-Powered Chatbot for Acko Health Insurance Queries**  

## **🔹 Overview**  
AckoAssist is an AI-powered chatbot designed to simplify Acko Health Insurance policy understanding. Built using Retrieval-Augmented Generation (RAG), this chatbot enables users to query policy details, coverage, and conditions in natural language, making insurance information more accessible and reducing manual effort.  

## **🛠️ Tech Stack**  
- **Python** 🐍  
- **LangChain** 🔗  
- **LLM (Google Gemini)** 🤖  
- **HuggingFace Transformers** 🧠  
- **Retrieval-Augmented Generation (RAG)** 📚  
- **FAISS (Facebook AI Similarity Search)** ⚡  
- **Streamlit** 🌐  

## **✨ Features**  
**Conversational AI** – Users can ask insurance-related questions in natural language.  
**RAG-based Retrieval** – Ensures accurate and contextually relevant responses.  
**FAISS-powered Search** – Enables **fast** document retrieval for policy details.  
**LLM Integration (Google Gemini, HuggingFace)** – Provides intelligent and concise answers.  
**User-Friendly Interface** – Built with **Streamlit** for seamless interaction.  

## **📌 How It Works?**  
1. **User Input** – Queries related to insurance policies.  
2. **FAISS Search** – Retrieves relevant policy documents.  
3. **LangChain Processing** – Sends relevant data to the LLM for response generation.  
4. **Response Generation** – AI formulates a clear and concise answer.  
5. **User Interaction** – Response is displayed in a user-friendly Streamlit UI.  

## **🚀 Getting Started**  

### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/gagandev-mishra/AckoAssist
cd AckoAssist
```

### **2️⃣ Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **3️⃣ Run the Streamlit App**  
```bash
streamlit run insurance_helper.py
```

## **📖 Future Improvements**  
🔹 Expand support for multiple insurance providers.  
🔹 Enhance response accuracy with fine-tuned LLM models.  
🔹 Implement voice-based interaction for hands-free queries.  

## **🤝 Contributing**  
We welcome contributions! Feel free to **fork this repo**, submit issues, or create pull requests to improve AckoAssist.  