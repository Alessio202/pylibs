import os
import re
import logging
from typing import Tuple
from langchain_ollama import ChatOllama

# =========================================================
# CONFIGURAZIONE PRINCIPALE
# =========================================================

LLM_MODEL = "deepseek-coder:6.7b"
TEMPERATURE = 0
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
        num_predict=2048
    )
except Exception as e:
    logging.critical(f"Errore connessione Ollama: {e}")
    raise SystemExit(f"Errore connessione Ollama: {e}")

# =========================================================
# ESTRAZIONE CODICE
# =========================================================

def extract_java_code(text: str) -> str:
    pattern = r"```(?:java)?(.*?)```"
    match = re.search(pattern, text, re.DOTALL)

    if match:
        return match.group(1).strip()

    if "public class" in text:
        return text.strip()

    return ""

# =========================================================
# VALIDAZIONE CODICE
# =========================================================

def validate_code(code: str) -> Tuple[bool, str]:

    if not code:
        return False, "Codice vuoto"

    if f"public class {CLASS_NAME}" not in code:
        return False, "Nome classe errato"

    if "@Test" not in code:
        return False, "Manca annotazione @Test"

    if "WebDriverWait" not in code:
        return False, "Manca WebDriverWait"

    if "ScreenshotExtension" not in code:
        return False, "Manca ScreenshotExtension"

    for forbidden in FORBIDDEN_PATTERNS:
        if forbidden in code:
            return False, f"Uso vietato rilevato: {forbidden}"

    return True, ""

# =========================================================
# COSTRUZIONE PROMPT
# =========================================================

def build_prompt(description: str) -> str:
    return f"""
Sei un Senior Java QA Automation Engineer esperto Selenium 4.20+.

REQUISITO TEST:
{description}

REGOLE OBBLIGATORIE (SE NON RISPETTATE LA RISPOSTA È INVALIDA):

1. Nome classe: public class {CLASS_NAME}
2. Usa JUnit 5 (@Test, @BeforeEach, @AfterEach)
3. È ASSOLUTAMENTE VIETATO usare:
   - WebDriverManager
   - System.setProperty
   - io.github.bonigarcia
4. Usa Selenium 4 moderno
5. Usa WebDriverWait con Duration.ofSeconds(10)
6. Crea classe ScreenshotExtension
7. Usa TestWatcher
8. Screenshot anche in caso di successo
9. Commenti in IRANIANO
10. Solo import necessari
11. NON aggiungere spiegazioni
12. Restituisci SOLO codice Java

Genera ora il codice.
"""

# =========================================================
# GENERAZIONE CON RETRY INTELLIGENTE
# =========================================================

def generate_java_test(description: str) -> str:

    prompt = build_prompt(description)

    for attempt in range(1, MAX_RETRIES + 1):

        print(f"\n[AI] Tentativo {attempt}/{MAX_RETRIES} con {LLM_MODEL}...")

        try:
            response = llm.invoke(prompt).content
            java_code = extract_java_code(response)

            valid, error = validate_code(java_code)

            if valid:
                logging.info("Codice generato correttamente.")
                return java_code

            logging.warning(f"Tentativo {attempt} fallito: {error}")

            if STRICT_MODE:
                prompt += f"""

ATTENZIONE:
Il codice precedente è stato rifiutato per il seguente motivo:
{error}

RISCRIVI COMPLETAMENTE il codice rispettando TUTTE le regole.
NON ripetere l'errore.
"""

        except Exception as e:
            logging.error(f"Errore LLM: {e}")
            return f"Errore durante invocazione LLM: {e}"

    logging.error("Tutti i tentativi falliti.")
    return ""

# =========================================================
# SALVATAGGIO FILE
# =========================================================

def save_result_file(code: str) -> str:

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    path = os.path.join(OUTPUT_FOLDER, f"{CLASS_NAME}.java")

    with open(path, "w", encoding="utf-8") as f:
        f.write(code)

    return path

# =========================================================
# MAIN LOOP
# =========================================================

def main():

    print("==============================================")
    print("  Java Selenium Generator PRO")
    print(f"  Model: {LLM_MODEL}")
    print("  Output fisso: Result.java")
    print("==============================================")

    while True:

        desc = input("\nDescrivi il test (o 'esci'): \n> ").strip()

        if desc.lower() in ["esci", "exit", "quit"]:
            print("Chiusura programma.")
            break

        if not desc:
            continue

        code = generate_java_test(desc)

        if code:
            path = save_result_file(code)
            print(f"\n✅ File creato con successo: {path}")
        else:
            print("\n❌ Generazione fallita dopo multipli tentativi.")
            print("Controlla generator.log per dettagli.")

if __name__ == "__main__":
    main()
