import os
import re
import logging
from typing import Tuple
from langchain_ollama import ChatOllama

# =========================================================
# CONFIGURAZIONE
# =========================================================

LLM_MODEL = "deepseek-coder:6.7b"
TEMPERATURE = 0.3  # ← Un po' di creatività
TOP_P = 0.9
MAX_RETRIES = 3
STRICT_MODE = True

OUTPUT_FOLDER = "test_generati"
CLASS_NAME = "Result"

FORBIDDEN_PATTERNS = [
    "WebDriverManager",
    "System.setProperty",
    "io.github.bonigarcia"
]

# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    filename="generator.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# =========================================================
# INIZIALIZZAZIONE LLM
# =========================================================

try:
    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        num_predict=4096  # ← Più spazio per creatività
    )
except Exception as e:
    logging.critical(f"Errore connessione Ollama: {e}")
    raise SystemExit(f"Errore connessione Ollama: {e}")

# =========================================================
# FUNZIONI UTILI
# =========================================================

def extract_java_code(text: str) -> str:
    """Estrae codice Java, gestendo sia markdown che testo grezzo"""
    
    # Prova con markdown
    pattern = r"```(?:java)?\s*(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    
    if matches:
        code = "\n\n".join(m.strip() for m in matches).strip()
        logging.info(f"Estratto da markdown: {len(code)} caratteri")
        return code
    
    # Prova a trovare codice Java diretto
    # Cerca da "package" o "import" o "public class" fino alla fine logica
    import_match = re.search(r'((?:package|import|public\s+class)\s+.*)', text, re.DOTALL)
    if import_match:
        code = import_match.group(1).strip()
        logging.info(f"Estratto da testo grezzo: {len(code)} caratteri")
        return code
    
    # Ultimo tentativo: tutto il testo se contiene elementi Java
    if any(keyword in text for keyword in ["public class", "@Test", "WebDriver"]):
        logging.warning("Usando tutto il testo come codice")
        return text.strip()
    
    logging.error("Nessun codice Java trovato")
    return ""

def validate_code(code: str) -> Tuple[bool, str]:
    """Verifica requisiti essenziali, senza essere troppo rigido"""
    if not code or len(code) < 100:
        return False, "Codice troppo corto o vuoto"

    if f"public class {CLASS_NAME}" not in code:
        return False, f"Deve contenere 'public class {CLASS_NAME}'"

    if "@Test" not in code:
        return False, "Deve contenere almeno un metodo @Test"

    # Verifica pattern vietati
    for forbidden in FORBIDDEN_PATTERNS:
        if forbidden in code:
            return False, f"NON usare {forbidden} (usa Selenium Manager integrato)"

    # Screenshot extension è richiesto ma diamo più flessibilità
    if "Screenshot" not in code:
        return False, "Deve includere meccanismo per screenshot (es. ScreenshotExtension)"

    return True, ""

def build_prompt(description: str, previous_error: str = "") -> str:
    """Prompt chiaro ma non soffocante"""
    
    prompt = f"""Genera un test Selenium 4 in Java per questo requisito:

**REQUISITO**: {description}

**REGOLE TECNICHE**:
- Classe principale: `public class {CLASS_NAME}`
- Framework: JUnit 5 (@Test, @BeforeEach, @AfterEach)
- WebDriver: `new ChromeDriver()` (Selenium 4 ha driver manager integrato)
- Wait espliciti: `WebDriverWait` con `Duration.ofSeconds(10)`
- Screenshot: crea una classe `ScreenshotExtension implements TestWatcher` nello stesso file
  - Cattura screenshot sia su successo che su fallimento
  - Usa @ExtendWith(ScreenshotExtension.class) sulla classe {CLASS_NAME}

**DIVIETI ASSOLUTI**:
- NON usare WebDriverManager, System.setProperty, o io.github.bonigarcia
- (Selenium 4 gestisce i driver automaticamente)

**OUTPUT**:
Genera il codice Java completo pronto per essere salvato in Result.java.
Puoi usare blocchi ```java``` oppure scrivere il codice direttamente.
Commenti in italiano.
"""

    if previous_error:
        prompt += f"""

**CORREZIONE RICHIESTA**:
Il tentativo precedente aveva questo problema: {previous_error}
Rigenera il codice completo correggendo l'errore.
"""

    return prompt

# =========================================================
# GENERAZIONE CON RETRY INTELLIGENTE
# =========================================================

def generate_java_test(description: str) -> str:
    previous_error = ""
    best_attempt = ""
    best_score = 0

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\n[AI] 🤖 Tentativo {attempt}/{MAX_RETRIES}...")

        try:
            prompt = build_prompt(description, previous_error)
            response = llm.invoke(prompt).content
            
            # Salva risposta grezza per debug
            debug_file = os.path.join(OUTPUT_FOLDER, f"debug_attempt_{attempt}.txt")
            os.makedirs(OUTPUT_FOLDER, exist_ok=True)
            with open(debug_file, "w", encoding="utf-8") as f:
                f.write(response)
            
            java_code = extract_java_code(response)
            
            if not java_code:
                previous_error = "Nessun codice Java rilevato nella risposta"
                logging.warning(f"Tentativo {attempt}: {previous_error}")
                print(f"   ⚠️ {previous_error}")
                continue
            
            # Calcola "score" del codice
            score = len(java_code)  # Codice più lungo è generalmente più completo
            if score > best_score:
                best_attempt = java_code
                best_score = score
            
            valid, error = validate_code(java_code)

            if valid:
                logging.info(f"✅ Codice valido al tentativo {attempt}")
                print(f"   ✅ Codice valido ({len(java_code)} caratteri)")
                return java_code

            logging.warning(f"Tentativo {attempt} non valido: {error}")
            print(f"   ⚠️ {error}")
            previous_error = error

        except Exception as e:
            logging.error(f"Errore LLM tentativo {attempt}: {e}")
            print(f"   ❌ Errore: {e}")
            previous_error = f"Errore tecnico: {str(e)}"

    # Se abbiamo almeno qualcosa, usa il miglior tentativo
    if best_attempt:
        logging.warning(f"Uso miglior tentativo (score: {best_score})")
        print(f"\n⚠️ Nessun tentativo perfetto, uso il migliore disponibile")
        return best_attempt
    
    # Totale fallimento
    logging.error("Tutti i tentativi falliti, nessun codice generato")
    return ""

# =========================================================
# SALVATAGGIO FILE
# =========================================================

def save_result_file(code: str) -> str:
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    path = os.path.join(OUTPUT_FOLDER, f"{CLASS_NAME}.java")
    
    if code:
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"   💾 Salvato: {path}")
    else:
        print(f"   ❌ Nessun codice da salvare")
        return None
    
    return path

# =========================================================
# MAIN LOOP
# =========================================================

def main():
    print("=" * 60)
    print("  🤖 Java Selenium Test Generator")
    print(f"  Model: {LLM_MODEL}")
    print(f"  Temperature: {TEMPERATURE} (creatività)")
    print(f"  Max Retries: {MAX_RETRIES}")
    print("=" * 60)
    print("\nQuesto tool usa AI per generare test automation DIVERSI ogni volta.")
    print("Più descrittivo sei, migliori saranno i test generati!\n")

    while True:
        desc = input("📝 Descrivi il test (o 'esci'):\n> ").strip()

        if desc.lower() in ["esci", "exit", "quit", "q"]:
            print("\n👋 Arrivederci!")
            break
        
        if not desc:
            print("⚠️  Inserisci una descrizione\n")
            continue

        print(f"\n{'='*60}")
        print(f"🎯 Generazione in corso per: {desc[:60]}...")
        print(f"{'='*60}")
        
        code = generate_java_test(desc)

        if not code:
            print("\n❌ Generazione fallita completamente.")
            print("   Suggerimenti:")
            print("   - Verifica che Ollama sia attivo")
            print("   - Prova una descrizione più semplice")
            print("   - Controlla generator.log per dettagli")
            print(f"   - Guarda {OUTPUT_FOLDER}/debug_attempt_*.txt per vedere le risposte raw")
            continue

        path = save_result_file(code)
        
        if path:
            lines = len(code.splitlines())
            print(f"\n✅ Generazione completata!")
            print(f"   📄 File: {path}")
            print(f"   📊 Righe: {lines}")
            print(f"   📏 Caratteri: {len(code)}")
            
            # Quick preview
            if "@Test" in code:
                test_count = code.count("@Test")
                print(f"   🧪 Test methods: {test_count}")
            if "Screenshot" in code:
                print(f"   📸 Screenshot: ✓")

if __name__ == "__main__":
    main()
