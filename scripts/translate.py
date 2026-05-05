from deep_translator import GoogleTranslator
from langdetect import detect

LANG_MAP = {
    "te": "telugu",
    "hi": "hindi",
    "en": "english",
    "ta": "tamil",
    "kn": "kannada",
}

def detect_language(text: str) -> str:
    try:
        lang = detect(text)
        return lang if lang in LANG_MAP else "en"
    except Exception:
        return "en"

def to_english(text: str) -> tuple:
    lang = detect_language(text)
    if lang == "en":
        return text, "en"
    try:
        translated = GoogleTranslator(source="auto", target="en").translate(text)
        return translated, lang
    except Exception:
        return text, "en"

def from_english(text: str, target_lang: str) -> str:
    if target_lang == "en":
        return text
    try:
        return GoogleTranslator(source="en", target=target_lang).translate(text)
    except Exception:
        return text

def main():
    print("Translation Test")
    
    tests = [
        "Can my landlord evict me without notice?",
        "నా యజమాని నోటీసు లేకుండా నన్ను తొలగించగలరా?",
        "क्या मेरा मकान मालिक बिना नोटिस के मुझे निकाल सकता है?",
    ]

    for text in tests:
        print(f"\nInput: {text}")
        en_text, lang = to_english(text)
        print(f"Detected: {lang}")
        print(f"English: {en_text}")
        if lang != "en":
            back = from_english(en_text, lang)
            print(f"Back: {back}")

if __name__ == "__main__":
    main()