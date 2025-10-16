# RAG System Features Overview

---

### 1. Simple RAG
✅ **Present**  
**The Basic Tool:**  
Like asking a librarian to find relevant books (retrieval) and then reading them to answer your question (generation). The foundation of everything else.

---

### 2. Memory
✅ **Present**  
**The Conversation Partner:**  
Like having a friend who remembers what you talked about earlier, allowing for follow-up questions ("What did you say about X again?") without repeating everything.

---

### 3. Adaptive RAG
✅ **Present**  
**The Smart Librarian:**  
This librarian first figures out if your question is simple ("What is X?") or complex ("Compare X and Y considering Z"). For complex questions, they might look up more information or use different search strategies.

---

### 4. Corrective RAG (CRAG)
✅ **Present**  
**The Fact-Checker:**  
Before giving you the final answer, this system double-checks if the information it found actually answers your question well. If not, it goes back and searches again or adjusts its approach.

---

### 5. Self-RAG
✅ **Present**  
**The Self-Reflective Expert:**  
After forming an initial answer, this expert pauses and asks, "Does this make sense? Is it supported by the evidence I found?" If not, it iterates, perhaps retrieving more information or refining its answer.

---

### 6. HyDe (Hypothetical Document Embedding)
✅ **Present**  
**The Idea Generator:**  
Instead of just searching for your exact words, it first imagines a detailed answer to your question and then searches for documents that match the *meaning* of that imagined answer, potentially finding relevant info even if it uses different terms.

---

### 7. Branched RAG
✅ **Added**  
**The Specialist Network:**  
Imagine a team of experts in different fields (Legal, Medical, Finance, Tech). This system first figures out which field your question belongs to and then asks the *right* expert for the answer, using their specific knowledge base.

---

### 8. Agentic RAG
✅ **Added**  
**The Team Leader:**  
Instead of one person answering, this acts like a manager who breaks down your question, asks multiple *document-specific agents* ("agents" are like tiny experts for individual documents) for their input, and then combines their answers into a final, well-rounded response.
