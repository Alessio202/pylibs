import os
import re
import logging
from typing import Tuple
from langchain_ollama import ChatOllama

# =========================================================
# CONFIGURAZIONE
# =========================================================

LLM_MODEL = "qwen2.5-coder:7b"
TEMPERATURE = 0.1
TOP_P = 0.9
MAX_RETRIES = 4

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
        num_predict=6000
    )
    logging.info(f"LLM inizializzato: {LLM_MODEL}")
except Exception as e:
    logging.critical(f"Errore connessione Ollama: {e}")
    raise SystemExit(f"Errore connessione Ollama: {e}")

# =========================================================
# ESTRAZIONE CODICE
# =========================================================

def extract_java_code(text: str) -> str:
    """Estrae codice Java da qualsiasi formato"""
    
    # Rimuovi possibili preamble/postamble
    text = text.strip()
    
    # Pattern 1: Blocchi markdown
    markdown_pattern = r"```(?:java)?\s*(.*?)```"
    matches = re.findall(markdown_pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        code = "\n\n".join(m.strip() for m in matches)
        logging.info(f"Estratto da markdown: {len(code)} char")
        return code
    
    # Pattern 2: Trova inizio codice Java
    java_starts = []
    for pattern in [r'^package\s+', r'^import\s+', r'^public\s+class\s+']:
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            java_starts.append(match.start())
    
    if java_starts:
        start_pos = min(java_starts)
        code = text[start_pos:].strip()
        
        # Rimuovi testo dopo l'ultima chiusura di classe
        # Conta le graffe per trovare dove finisce il codice
        brace_count = 0
        last_closing = -1
        
        for i, char in enumerate(code):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    last_closing = i
        
        if last_closing > 0:
            code = code[:last_closing + 1]
        
        logging.info(f"Estratto da testo grezzo: {len(code)} char")
        return code
    
    # Pattern 3: Se contiene elementi Java, prova a usare tutto
    if any(keyword in text for keyword in ["public class", "@Test", "WebDriver", "import"]):
        logging.warning("Usando tutto il testo come codice")
        return text.strip()
    
    logging.error("Nessun codice Java trovato")
    return ""

# =========================================================
# VALIDAZIONE
# =========================================================

def validate_code(code: str) -> Tuple[bool, str]:
    """Validazione con feedback specifico per correzione autonoma"""
    
    if not code:
        return False, "Risposta vuota"
    
    if len(code) < 200:
        return False, "Codice troppo corto - deve includere classe test completa + ScreenshotExtension"
    
    # Check 1: Nome classe
    if f"public class {CLASS_NAME}" not in code:
        found_classes = re.findall(r'public class (\w+)', code)
        if found_classes:
            return False, f"Classe trovata: {found_classes[0]}, ma deve chiamarsi esattamente '{CLASS_NAME}'"
        return False, f"Manca 'public class {CLASS_NAME}'"
    
    # Check 2: Annotazioni JUnit
    if "@Test" not in code:
        return False, "Nessun metodo @Test trovato - il test non può eseguire"
    
    if "@BeforeEach" not in code:
        return False, "Manca @BeforeEach per setup del WebDriver"
    
    if "@AfterEach" not in code:
        return False, "Manca @AfterEach per cleanup del WebDriver"
    
    # Check 3: Selenium components
    if "WebDriver" not in code:
        return False, "Manca dichiarazione WebDriver"
    
    if "ChromeDriver()" not in code and "new ChromeDriver" not in code:
        return False, "Manca inizializzazione ChromeDriver"
    
    if "WebDriverWait" not in code:
        return False, "Manca WebDriverWait per wait espliciti"
    
    # Check 4: Duration syntax (errore comune)
    wrong_wait_pattern = r'WebDriverWait\s*\(\s*\w+\s*,\s*\d+\s*\)'
    if re.search(wrong_wait_pattern, code):
        return False, "WebDriverWait syntax errata: usa 'new WebDriverWait(driver, Duration.ofSeconds(10))' non 'new WebDriverWait(driver, 10)'"
    
    if "Duration.ofSeconds" not in code and "WebDriverWait" in code:
        return False, "WebDriverWait richiede Duration.ofSeconds(X), non un intero diretto"
    
    # Check 5: Screenshot Extension
    if "ScreenshotExtension" not in code:
        return False, "Manca classe ScreenshotExtension"
    
    if "implements TestWatcher" not in code:
        return False, "ScreenshotExtension deve implementare TestWatcher"
    
    if "@ExtendWith" not in code or "ScreenshotExtension.class" not in code:
        return False, f"Manca @ExtendWith(ScreenshotExtension.class) sulla classe {CLASS_NAME}"
    
    # Check 6: TestWatcher methods - rileva allucinazioni
    hallucinated = [
        "TestSuccessfulEvent", "TestFailedEvent", "TestAbortedEvent",
        "__TestSuccessfulEvent__", "__event__", "TestDisabledEvent"
    ]
    for fake in hallucinated:
        if fake in code:
            return False, f"API inesistente '{fake}' rilevata. TestWatcher usa firme: testSuccessful(ExtensionContext context) e testFailed(ExtensionContext context, Throwable cause)"
    
    # Check 7: Pattern vietati
    for forbidden in FORBIDDEN_PATTERNS:
        if forbidden in code:
            return False, f"Pattern vietato '{forbidden}' - Selenium 4 ha driver manager integrato"
    
    # Check 8: Static driver se usa ScreenshotExtension
    if "class ScreenshotExtension" in code:
        # Cerca se driver è accessibile
        if "private WebDriver driver" in code:
            # Verifica se è static
            driver_lines = [line for line in code.split('\n') if 'WebDriver driver' in line and 'private' in line]
            if driver_lines and not any('static' in line for line in driver_lines):
                return False, "WebDriver driver deve essere 'private static WebDriver driver' per accesso da ScreenshotExtension"
    
    # Check 9: Screenshot capture logic
    if "TakesScreenshot" not in code:
        return False, "Manca logica per catturare screenshot (TakesScreenshot)"
    
    if "getScreenshotAs" not in code:
        return False, "Manca chiamata getScreenshotAs(OutputType.FILE)"
    
    # All checks passed
    return True, ""

# =========================================================
# PROMPT CONSTRUCTION
# =========================================================

def build_prompt(description: str, previous_error: str = "") -> str:
    """Prompt SENZA boilerplate - solo specifiche tecniche precise"""
    
    prompt = f"""Genera un test automation completo in Java per Selenium 4.20+ e JUnit 5.

**REQUISITO FUNZIONALE**:
{description}

**SPECIFICHE TECNICHE OBBLIGATORIE**:

1. STRUTTURA CLASSE TEST:
   - Nome classe: public class {CLASS_NAME}
   - Annotation: @ExtendWith(ScreenshotExtension.class)
   - Field: private static WebDriver driver (DEVE essere static)
   - Field: private WebDriverWait wait
   - Metodo: @BeforeEach void setUp() - inizializza driver e wait
   - Metodo: @Test void testXXX() - implementa il test richiesto
   - Metodo: @AfterEach void tearDown() - chiude driver

2. INIZIALIZZAZIONE SELENIUM:
   - Driver: driver = new ChromeDriver();
   - Wait: wait = new WebDriverWait(driver, Duration.ofSeconds(10));
   - NON usare WebDriverManager, System.setProperty, o io.github.bonigarcia
   - Selenium 4 gestisce i driver automaticamente

3. TEST LOGIC:
   - Usa wait.until(ExpectedConditions.XXX) per wait espliciti
   - Usa By.id(), By.name(), By.xpath(), ecc. per locator
   - Implementa la logica richiesta nel requisito funzionale

4. SCREENSHOT EXTENSION (SECONDA CLASSE NELLO STESSO FILE):
   - Nome: class ScreenshotExtension implements TestWatcher
   - Metodo: public void testSuccessful(ExtensionContext context)
   - Metodo: public void testFailed(ExtensionContext context, Throwable cause)
   - Metodo privato: void takeScreenshot(String status)
   - Cattura screenshot usando: ((TakesScreenshot) {CLASS_NAME}.driver).getScreenshotAs(OutputType.FILE)
   - Salva con FileUtils.copyFile() nella cartella screenshots/

5. API CORRETTE - NON INVENTARE:
   - TestWatcher methods: testSuccessful(ExtensionContext context) e testFailed(ExtensionContext context, Throwable cause)
   - NON esistono: TestSuccessfulEvent, TestFailedEvent, TestAbortedEvent
   - WebDriverWait constructor: new WebDriverWait(driver, Duration.ofSeconds(10))
   - NON usare: new WebDriverWait(driver, 10) [sintassi vecchia]

6. IMPORT NECESSARI:
   - java.io.File, java.time.Duration
   - org.apache.commons.io.FileUtils
   - org.junit.jupiter.api.* (Test, BeforeEach, AfterEach, extension.*)
   - org.openqa.selenium.* (By, WebDriver, OutputType, TakesScreenshot)
   - org.openqa.selenium.chrome.ChromeDriver
   - org.openqa.selenium.support.ui.* (WebDriverWait, ExpectedConditions)

7. FORMATO OUTPUT:
   - Restituisci SOLO il file Result.java completo
   - Due classi: {CLASS_NAME} e ScreenshotExtension nello stesso file
   - Commenti in italiano
   - Codice production-ready

**REGOLE DI QUALITÀ**:
- Codice compilabile senza errori
- Nomi metodi descrittivi
- Wait espliciti, mai Thread.sleep()
- Gestione eccezioni in screenshot logic
- Screenshot salvati con timestamp univoci
"""

    if previous_error:
        prompt += f"""

**CORREZIONE RICHIESTA**:
Il tentativo precedente ha fallito per questo motivo:
{previous_error}

Analizza l'errore e rigenera il codice completo correggendo il problema.
NON ripetere lo stesso errore.
"""

    return prompt

# =========================================================
# GENERAZIONE AUTONOMA
# =========================================================

def generate_java_test(description: str) -> str:
    """Generazione completamente autonoma - nessun template"""
    
    previous_error = ""
    attempts_data = []

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\n{'='*60}")
        print(f"🤖 Tentativo {attempt}/{MAX_RETRIES}")
        print(f"{'='*60}")

        try:
            # Costruisci prompt
            prompt = build_prompt(description, previous_error)
            
            # Invoca LLM
            print("⏳ Generazione in corso...")
            response = llm.invoke(prompt).content
            
            # Salva debug
            os.makedirs(OUTPUT_FOLDER, exist_ok=True)
            debug_path = os.path.join(OUTPUT_FOLDER, f"attempt_{attempt}_raw.txt")
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(f"PROMPT:\n{prompt}\n\n{'='*80}\n\nRESPONSE:\n{response}")
            logging.info(f"Risposta salvata in {debug_path}")
            
            # Estrai codice
            java_code = extract_java_code(response)
            
            if not java_code:
                previous_error = "Nessun codice Java valido estratto dalla risposta. Assicurati di generare codice Java completo."
                print(f"❌ {previous_error}")
                logging.warning(f"Attempt {attempt}: {previous_error}")
                attempts_data.append({"attempt": attempt, "error": previous_error, "code_length": 0})
                continue
            
            # Salva codice estratto
            extracted_path = os.path.join(OUTPUT_FOLDER, f"attempt_{attempt}_extracted.java")
            with open(extracted_path, "w", encoding="utf-8") as f:
                f.write(java_code)
            
            print(f"📄 Codice estratto: {len(java_code)} caratteri, {len(java_code.splitlines())} righe")
            
            # Valida
            valid, error = validate_code(java_code)

            if valid:
                print(f"✅ VALIDAZIONE SUPERATA")
                print(f"📊 Statistiche:")
                print(f"   - Caratteri: {len(java_code)}")
                print(f"   - Righe: {len(java_code.splitlines())}")
                print(f"   - Test methods: {java_code.count('@Test')}")
                logging.info(f"Codice valido generato al tentativo {attempt}")
                return java_code

            # Validazione fallita
            print(f"⚠️ VALIDAZIONE FALLITA")
            print(f"   Errore: {error}")
            logging.warning(f"Attempt {attempt} validation failed: {error}")
            
            attempts_data.append({
                "attempt": attempt,
                "error": error,
                "code_length": len(java_code)
            })
            
            previous_error = error

        except Exception as e:
            error_msg = f"Errore tecnico durante generazione: {str(e)}"
            print(f"❌ {error_msg}")
            logging.error(f"Attempt {attempt} exception: {e}", exc_info=True)
            previous_error = error_msg
            attempts_data.append({"attempt": attempt, "error": error_msg, "code_length": 0})

    # Tutti i tentativi falliti
    print(f"\n{'='*60}")
    print("❌ GENERAZIONE FALLITA dopo tutti i tentativi")
    print(f"{'='*60}")
    print("\n📊 Riepilogo tentativi:")
    for data in attempts_data:
        print(f"   Tentativo {data['attempt']}: {data['code_length']} char - {data['error'][:80]}")
    
    logging.error("Generazione fallita dopo tutti i tentativi")
    logging.error(f"Tentativi: {attempts_data}")
    
    return ""

# =========================================================
# SALVATAGGIO
# =========================================================

def save_result_file(code: str) -> str:
    """Salva il codice generato"""
    if not code:
        return None
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    path = os.path.join(OUTPUT_FOLDER, f"{CLASS_NAME}.java")
    
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    
    logging.info(f"File salvato: {path}")
    return path

# =========================================================
# MAIN
# =========================================================

def main():
    print("="*70)
    print(" "*15 + "🤖 AUTONOMOUS CODE GENERATOR")
    print("="*70)
    print(f"\n  Model: {LLM_MODEL}")
    print(f"  Mode: Fully Autonomous (no templates, no boilerplate)")
    print(f"  Output: {OUTPUT_FOLDER}/{CLASS_NAME}.java")
    print(f"  Max Retries: {MAX_RETRIES}")
    print("\n" + "="*70)
    print("\nIl modello deve generare TUTTO il codice in modo autonomo.")
    print("Nessun template o esempio fornito.\n")

    while True:
        print("\n" + "-"*70)
        desc = input("📝 Descrivi il test automation (o 'esci' per uscire):\n> ").strip()

        if desc.lower() in ["esci", "exit", "quit", "q"]:
            print("\n👋 Chiusura generatore. Arrivederci!")
            break
        
        if not desc:
            print("⚠️  Inserisci una descrizione valida")
            continue

        print(f"\n{'='*70}")
        print(f"🎯 Requisito: {desc}")
        print(f"{'='*70}")
        
        # Genera
        code = generate_java_test(desc)

        if code:
            # Salva
            path = save_result_file(code)
            
            print(f"\n{'='*70}")
            print("✅ GENERAZIONE COMPLETATA CON SUCCESSO")
            print(f"{'='*70}")
            print(f"\n📁 File: {path}")
            print(f"📊 Statistiche:")
            print(f"   - Righe: {len(code.splitlines())}")
            print(f"   - Caratteri: {len(code)}")
            print(f"   - Import statements: {code.count('import ')}")
            print(f"   - Test methods: {code.count('@Test')}")
            
            if "ScreenshotExtension" in code:
                print(f"   - Screenshot: ✓ ScreenshotExtension presente")
            
            print(f"\n💡 Usa: javac {path} per compilare")
            print(f"📋 Log completo: generator.log")
            print(f"🐛 Debug files: {OUTPUT_FOLDER}/attempt_*")
            
        else:
            print(f"\n{'='*70}")
            print("❌ GENERAZIONE FALLITA")
            print(f"{'='*70}")
            print("\n🔍 Diagnostica:")
            print(f"   1. Verifica Ollama: ollama list | grep {LLM_MODEL}")
            print(f"   2. Controlla log: tail -f generator.log")
            print(f"   3. Vedi tentativi raw: {OUTPUT_FOLDER}/attempt_*_raw.txt")
            print(f"   4. Vedi codice estratto: {OUTPUT_FOLDER}/attempt_*_extracted.java")
            print("\n💡 Suggerimenti:")
            print("   - Descrizione più specifica")
            print("   - Verifica memoria disponibile per il modello 33B")
            print("   - Prova con descrizione più semplice prima")

if __name__ == "__main__":
    main()
