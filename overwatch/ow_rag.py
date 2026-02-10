import os
import pdfplumber

# Modern LangChain 2026 Imports
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- CONFIGURAZIONE ---
PDF_PATH = "./ml/ow_docs"       
CHROMA_DIR = "./chroma_db_qwen"
LLM_MODEL = "qwen2.5"
EMBED_MODEL = "nomic-embed-text"

# --- 1. FUNZIONE CARICAMENTO PDF ---
def load_all_pdfs(directory: str) -> str:
    all_text = ""
    if not os.path.exists(directory):
        os.makedirs(directory)
        return ""
    
    files = [f for f in os.listdir(directory) if f.lower().endswith(".pdf")]
    if not files:
        return ""

    for filename in files:
        path = os.path.join(directory, filename)
        print(f"📄 Elaborazione nuovo file: {filename}")
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    all_text += text + "\n"
    return all_text

# --- 2. INIZIALIZZAZIONE EMBEDDINGS ---
embeddings = OllamaEmbeddings(model=EMBED_MODEL)

# --- 3. GESTIONE DATABASE (LOGICA PERSISTENTE) ---
# Controlliamo se il database esiste già
if os.path.exists(CHROMA_DIR) and len(os.listdir(CHROMA_DIR)) > 0:
    print("✅ Database trovato! Caricamento dati esistenti...")
    vectordb = Chroma(
        persist_directory=CHROMA_DIR, 
        embedding_function=embeddings
    )
else:
    print("🔍 Database non trovato o vuoto. Analizzo i PDF...")
    raw_text = load_all_pdfs(PDF_PATH)
    
    if not raw_text.strip():
        print("❌ Errore: Nessun PDF trovato in './ow_docs' o file illeggibili.")
        exit()

    # Splitting
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(raw_text)

    # Creazione e salvataggio database
    vectordb = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    print(f"💾 Database creato e salvato in '{CHROMA_DIR}'")

# --- 4. PIPELINE RAG (LCEL) ---
print(f"🤖 Avvio {LLM_MODEL}...")
llm = ChatOllama(model=LLM_MODEL, temperature=0)
retriever = vectordb.as_retriever(search_kwargs={"k": 4})

template = """
Usa il seguente contesto per rispondere alla domanda. 
Sii preciso e rispondi in inglese.
Se non sai la risposta, ammettilo onestamente.

CONTESTO:
{context}

DOMANDA: 
{question}

RISPOSTA:
"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Catena di esecuzione
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- 5. INTERFACCIA CHAT ---
print("\n✨ Sistema pronto! Digita 'esci' per terminare.")

while True:
    query = input("\n🤔 Domanda: ")

    if query.lower() in {"exit", "quit", "esci"}:
        break

    if not query.strip():
        continue

    print("⏳ Sto pensando...")
    try:
        response = rag_chain.invoke(query)
        print(f"\n📢 Risposta:\n{response}")
    except Exception as e:
        print(f"❌ Errore: {e}")