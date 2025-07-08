import pandas as pd
import stanza
from collections import Counter
from typing import List

def run_postagging(input_csv: str, top_n: int = 30) -> List[str]:
    df = pd.read_csv(input_csv)
    df = df.dropna(subset=['cleaned_review'])
    texts = df['cleaned_review'].astype(str).tolist()

    nlp = stanza.Pipeline(lang='id', processors='tokenize,pos,lemma', use_gpu=False, verbose=False)

    counter = Counter()
    for doc_text in texts:
        doc = nlp(doc_text)
        for sent in doc.sentences:
            for word in sent.words:
                lemma = word.lemma
                if word.upos == 'NOUN' and lemma is not None and len(lemma) > 2:
                    counter[lemma] += 1
    
    top_nouns = [term for term, _ in counter.most_common(top_n)]
    return top_nouns