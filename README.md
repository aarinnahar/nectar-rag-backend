<div align="center">
  <img width="1633" height="863" alt="nectar-evaluation" src="https://github.com/user-attachments/assets/2bcc4e51-7df3-4239-bacc-5539eddc911d" />
<!-- 📸 PLACEHOLDER: Insert a high-res screenshot of your Step 1 UI or the glowing radar loading modal here -->

  # Nectar RAG Evaluation Platform

  **A production-grade, zero-cost retrieval-augmented generation (RAG) diagnostic engine.**  
  Quantify chunking strategies, track precision-recall drift, and test multi-agent architectures using local vector indices.

  [![Live Demo](https://img.shields.io/badge/Live_Demo-nectar--rag.vercel.app-000000?style=for-the-badge&logo=vercel)]([YOUR_LIVE_LINK_HERE](https://nectar-rag.vercel.app/))
  [![Video Demo](https://img.shields.io/badge/Video_Demo-Watch_on_LinkedIn-0A66C2?style=for-the-badge&logo=linkedin)](YOUR_LINKEDIN_VIDEO_LINK_HERE)

  ![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
  ![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C?style=for-the-badge)
  ![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
  ![AWS](https://img.shields.io/badge/AWS-232F3E?style=for-the-badge&logo=amazonaws)
</div>

<br />

## ⚡ Core Architecture & Engineering Highlights

Most developers guess their RAG hyperparameters. Nectar RAG eliminates the guesswork by validating document chunking against a JSON Golden Dataset, generating automated HTML benchmark reports.

*   **Latency Optimized (76% Speedup):** Collapsed total pipeline execution time from 5 minutes to 72 seconds. Bypassed redundant CPU embedding passes by injecting an in-memory `FastEmbedWrapper` caching dictionary and pure NumPy vectorized math for FAISS vector storage.
*   **Zero-Cost Infrastructure Engineering:** Engineered to run heavy local ONNX embedding models on a 1GB AWS `t3.micro` instance. Completely bypassed 512MB container OOM kernel crashes by tuning a 2GB Linux swap file (`swappiness=10`).
*   **100% Tenant Isolation & Privacy:** Safeguards concurrent multi-user web traffic on a single vCPU via an `asyncio.Lock()` queue. Every document upload is securely sandboxed into a temporary `uuid.uuid4()` workspace with immediate automated garbage collection.
*   **Real-Time SSE Streaming:** Utilizes FastAPI `StreamingResponse` and LangGraph's `graph.astream()` to push sub-second node execution updates directly to the React interface, entirely removing the need for external WebSocket libraries.
*   **Bypassing Edge Proxy Limitations:** Escapes Vercel's free-tier payload-stripping on `multipart/form-data` uploads by routing the frontend directly to the AWS EC2 backend via a permanent, static **ngrok** HTTPS tunnel.

---

## 📸 Output & Diagnostics

<!-- 📸 PLACEHOLDER: Insert a screenshot of the generated chunking_report.html showing the metrics and tables -->
![Nectar HTML Evaluation Report](docs/nectar-report-preview.png)

The pipeline dynamically generates a downloadable HTML evaluation report tracking:
*   Total execution latency and chunk volume.
*   Retrieved context faithfulness.
*   Cost analysis and token usage.

---

## 🛠️ Technology Stack

**Frontend (Client)**
*   **React + Vite:** Ultra-fast HMR and optimized production builds.
*   **Tailwind CSS:** Custom animations, gradient glowing modals, and modern UI/UX.
*   **Lucide React:** Lightweight SVG iconography.

**Backend (Serverless / AWS)**
*   **FastAPI:** Asynchronous Python web framework for API routing and Server-Sent Events.
*   **LangGraph & LangChain:** Multi-node autonomous pipeline orchestration and state management.
*   **FastEmbed & FAISS:** Local ONNX embedding models and in-memory similarity search vectors.
*   **PyMuPDF:** Sub-second binary XML extraction for strict 10MB/100-page document limit enforcement.

---

## 🚀 Local Installation & Setup

### Prerequisites
* Python 3.11+
* Node.js 18+
* [Ollama](https://ollama.ai/) (Optional, for local inference)

### 1. Clone the Repository
```bash
git clone [https://github.com/aarin/nectar-rag.git](https://github.com/aarin/nectar-rag.git)
cd nectar-rag
