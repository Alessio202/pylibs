import os
import sys
import pdfplumber

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# =========================
# CONFIGURAZIONE
# =========================

PDF_PATH = "./ml/ow_test"
CHROMA_DIR = "./chroma_db_nemo"

LLM_MODEL = "mistral-nemo"
EMBED_MODEL = "mxbai-embed-large"

CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
TOP_K = 10

# Se True, cancella e ricrea il DB Chroma
FORCE_REBUILD = False

# =========================
# UTILS LOG
# =========================

def log(msg: str):
    print(f"[INFO] {msg}")

def warn(msg: str):
    print(f"[WARN] {msg}", file=sys.stderr)

def error(msg: str):
    print(f"[ERROR] {msg}", file=sys.stderr)

# =========================
# 1. CARICAMENTO PDF CON METADATA
# =========================

def load_all_pdfs(directory: str):
    documents = []

    if not os.path.exists(directory):
        warn(f"La cartella PDF '{directory}' non esiste. La creo, ma è vuota.")
        os.makedirs(directory, exist_ok=True)
        return documents

    pdf_files = [f for f in os.listdir(directory) if f.lower().endswith(".pdf")]
    if not pdf_files:
        warn(f"Nessun PDF trovato in '{directory}'.")
        return documents

    for filename in pdf_files:
        path = os.path.join(directory, filename)
        log(f"Processing file: {filename}")

        try:
            with pdfplumber.open(path) as pdf:
                total_pages = len(pdf.pages)
                pages_with_text = 0

                for page_number, page in enumerate(pdf.pages):
                    text = page.extract_text()

                    if text and text.strip():
                        pages_with_text += 1
                        documents.append(
                            Document(
                                page_content=text,
                                metadata={
                                    "source": filename,
                                    "page": page_number + 1
                                }
                            )
                        )
                    else:
                        warn(f"No text extracted from {filename} page {page_number + 1}")

                log(f"{filename}: {pages_with_text}/{total_pages} pages with text")

        except Exception as e:
            error(f"Errore leggendo '{filename}': {e}")

    log(f"Totale documenti (pagine con testo) caricati: {len(documents)}")
    return documents

# =========================
# 2. INIZIALIZZAZIONE EMBEDDINGS
# =========================

log(f"Initializing embeddings model: {EMBED_MODEL}")
embeddings = OllamaEmbeddings(model=EMBED_MODEL)

# =========================
# 3. CREAZIONE / CARICAMENTO VECTOR DB
# =========================

def build_or_load_vectordb():
    db_exists = os.path.exists(CHROMA_DIR) and len(os.listdir(CHROMA_DIR)) > 0

    if db_exists and not FORCE_REBUILD:
        log("Existing database found. Loading...")
        vectordb = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings
        )
        return vectordb

    if db_exists and FORCE_REBUILD:
        log("FORCE_REBUILD=True, cancello il DB esistente...")
        for root, dirs, files in os.walk(CHROMA_DIR, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))

    log("Building new database from PDFs...")
    documents = load_all_pdfs(PDF_PATH)

    if not documents:
        error("Nessun documento leggibile trovato. Esco.")
        sys.exit(1)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    log("Splitting documents into chunks...")
    chunks = splitter.split_documents(documents)
    log(f"Totale chunks creati: {len(chunks)}")

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    log(f"Database created and saved in '{CHROMA_DIR}'")
    return vectordb

vectordb = build_or_load_vectordb()

# =========================
# 4. CONFIGURAZIONE LLM E RETRIEVER
# =========================

log(f"Starting model: {LLM_MODEL}")

llm = ChatOllama(
    model=LLM_MODEL,
    temperature=0
)

retriever = vectordb.as_retriever(
    search_type="mmr", 
    search_kwargs={"k": TOP_K, "fetch_k": TOP_K * 3}
)

# =========================
# 5. PROMPT RAG
# =========================

template = """
You are a document assistant.

Use ONLY the provided context to answer the question.
Be concise and answer in Italian.
If the answer is not in the context, say clearly that you do not know.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
Include the list of sources you used.
"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    formatted = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "?")
        formatted.append(f"[Source: {src} - Page {page}]\n{d.page_content}")
    return "\n\n".join(formatted)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# =========================
# 6. TEST RAPIDO DEL RETRIEVER (OPZIONALE)
# =========================

def debug_retriever(sample_query: str):
    log(f"Testing retriever with query: {sample_query!r}")
    docs = retriever.get_relevant_documents(sample_query)
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "?")
        print(f"  [{i}] {src} - page {page}")

# Decommenta se vuoi fare un test iniziale
# debug_retriever("test")

# =========================
# 7. INTERFACCIA CHAT
# =========================

log("System ready. Type 'exit' to quit.")

while True:
    try:
        query = input("\nQuestion: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye.")
        break

    if not query:
        continue

    if query.lower() in {"exit", "quit"}:
        print("Bye.")
        break

    print("Generating answer...")

    try:
        response = rag_chain.invoke(query)
        print("\nAnswer:\n")
        print(response)
    except Exception as e:
        error(f"Error during RAG invocation: {e}")
