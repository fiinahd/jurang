import re
import nltk
nltk.download('punkt')

def clean_text(text: str) -> tuple[str, list[str]]:
    text = text.lower()
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    tokens = nltk.word_tokenize(text)
    return " ".join(tokens), tokens