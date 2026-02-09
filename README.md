**Insurance Query Engine – Project Overview**

An LLM-powered intelligent query–retrieval system designed to answer user questions from large, unstructured insurance documents such as policies, contracts, endorsements, and emails.

**Purpose**
Enable fast, accurate, and contextual responses to insurance-related queries without manual document searching.

**Core Functionality**

* Document ingestion (PDF/Word/Text)
* Text chunking and semantic embedding
* Vector database creation for similarity search
* Retrieval-Augmented Generation (RAG) using LLMs
* Context-aware, explainable answers

**Architecture**

* **Backend**: FastAPI / Flask
* **LLM**: Gemini API
* **Vector DB**: FAISS / Chroma
* **Database**: MongoDB
* **Frontend**: React (Vite)

**Key Features**

* Accurate insurance policy Q&A
* Multi-document querying
* Reduced hallucinations via grounded retrieval
* Modular and scalable design
* API-driven backend with web-based UI

---

### **Setup Instructions**

**Backend**

1. Run:

   ```
   pip install -r requirements.txt
   ```
2. Download required **spaCy** language model.
3. Run the backend:

   ```
   python main.py
   ```
4. To verify vector database creation:

   ```
   python test.py
   ```

**Frontend**

1. Navigate to frontend directory.
2. Install dependencies:

   ```
   npm install
   ```
3. Start the development server:

   ```
   npm run dev
   ```

**Environment Configuration (`.env`)**

```
MONGO_URI="your mongo uri"
GEMINI_API_KEY="your gemini api key"
```

**Outcome**

* Faster insurance query resolution
* Improved policy understanding for users and agents
* Practical application of LLMs + RAG in real-world insurance workflows

