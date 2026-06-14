# How to Run the Two-Stage RAG Application

> ⚠️ **These are the original macOS notes** (hardcoded `/Users/brajeswarghosh/...` paths).
> For the current, verified **Linux** setup — Node 20, CPU Torch, and the dep/lockfile
> gotchas — use **[../RUN_GUIDE.md](../RUN_GUIDE.md)** instead.

This project consists of three parts that need to be running simultaneously to work end-to-end:
1. **Python FastAPI Backend** (Handles indexing, embedding, retrieval, reranking, and LLM communication)
2. **Node.js Express Proxy** (Proxies requests and real-time WebSocket connections)
3. **React Vite Frontend** (The interactive user interface)

## Prerequisites

1. Ensure you have Python 3.9+ and Node.js installed on your Mac.
2. Ensure you have your `.env` file configured in the root `two_stage_rag` directory with your Google API key:
   ```env
   GOOGLE_API_KEY=AQ.Ab8RN6KmYQX...
   ```

---

## Step-by-Step Running Instructions

You will need to open **three separate terminal windows/tabs**, one for each service.

### Terminal 1: Run the Python Backend
This is the core engine for the AI RAG pipeline.
```bash
# 1. Navigate to the project folder
cd /Users/brajeswarghosh/Documents/GitHub/master/IITP_workshop/two_stage_rag

# 2. Activate the virtual environment
source venv/bin/activate

# 3. Start the FastAPI server on port 8000
uvicorn api:app --reload --port 8000
```
*(Wait until you see `Application startup complete.`)*

### Terminal 2: Run the Node.js Proxy
This handles routing the API and streaming WebSocket upgrades between the frontend and the Python backend.
```bash
# 1. Navigate to the backend folder
cd /Users/brajeswarghosh/Documents/GitHub/master/IITP_workshop/two_stage_rag/backend

# 2. Install dependencies (only needed the first time)
npm install

# 3. Start the proxy server on port 3001
npm run dev
```
*(Wait until you see `Server is running on port 3001`)*

### Terminal 3: Run the React Frontend
This serves the beautiful web interface you interact with.
```bash
# 1. Navigate to the frontend folder
cd /Users/brajeswarghosh/Documents/GitHub/master/IITP_workshop/two_stage_rag/frontend

# 2. Install dependencies (only needed the first time)
npm install

# 3. Start the development server
npm run dev
```
*(Wait until it gives you the local URL, usually `http://localhost:5173`)*

---

## Accessing the Application

Once all three servers are running successfully without errors, open your web browser and navigate to:
**[http://localhost:5173](http://localhost:5173)**

## How to use the Web App:
1. **Upload Documents**: Drop a PDF or TXT file into the "Drop files here" zone in the left sidebar.
2. **Wait for Indexing**: Wait a moment for it to be processed (you will see it listed below "Ingested Documents").
3. **Ask Questions**: Type a question about the document in the bottom chat bar and press Enter.
4. **Watch it Work**: You will see the pipeline stages update in real-time, followed by the answer streaming word-by-word into the chat interface.

## Stopping the application
To stop the servers when you are finished, simply go to each of the three terminal windows and press `Ctrl + C` on your keyboard.
