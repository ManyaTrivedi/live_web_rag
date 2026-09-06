import time
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from ddgs import DDGS

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize Free Local Embeddings & LLM
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = Ollama(model="llama3")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request, "index.html", {})

@app.post("/chat")
async def chat_endpoint(request: Request):
    start_time = time.time()
    data = await request.json()
    query = data.get("question")
    
    if not query:
        return {"error": "No question provided"}

    # --- ADVANCED FEATURE: QUERY REWRITING ---
    # We ask Llama 3 to turn the user's chat question into an optimized web search query
    rewrite_prompt = f"Convert this user question into a concise, keyword-rich search engine query. Return ONLY the search query, no extra text.\nUser Question: {query}\nSearch Query:"
    search_query = llm.invoke(rewrite_prompt).strip().replace('"', '')
    print(f"Original Query: {query} | Optimized Search Query: {search_query}")
    
    # 1. Search the web using the optimized query
    search_start = time.time()
    urls = []
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(search_query, max_results=3)]
            for r in results:
                urls.append(r['href'])
    except Exception as e:
        print(f"Search error: {e}")
    search_time = round(time.time() - search_start, 2)

    if not urls:
        return {"answer": "I couldn't find any relevant live web results for that question.", "metrics": "0s"}

    # 2. Scrape web pages
    try:
        loader = WebBaseLoader(urls)
        docs = loader.load()
    except Exception as e:
        return {"answer": f"Error scraping web pages: {str(e)}", "metrics": "0s"}

    # 3. Chunk & Index locally
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 4. Retrieve Chunks
    retrieved_docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    sources = list(set([doc.metadata.get('source', 'Unknown') for doc in retrieved_docs]))

    # 5. Generate Final Answer
    gen_start = time.time()
    prompt = f"""You are a helpful AI assistant. Answer the user's question accurately using ONLY the context provided below. If you cannot answer based on the context, say "I cannot answer this based on the web results."

Context:
{context}

Question: {query}
Answer:"""

    answer = llm.invoke(prompt)
    gen_time = round(time.time() - gen_start, 2)
    total_time = round(time.time() - start_time, 2)

    metrics = f"Search: {search_time}s | Gen: {gen_time}s | Total: {total_time}s"

    return {
        "answer": answer,
        "sources": sources,
        "metrics": metrics
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)