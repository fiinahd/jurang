from collections import Counter

def extract_nouns(tagged_tokens: list[tuple[str,str]]) -> tuple[list[str], list[int]]:
    nouns = [w for w, tag in tagged_tokens if tag.startswith('NN')]
    freq = Counter(nouns)
    aspects, freqs = zip(*freq.most_common())
    return list(aspects), list(freqs)