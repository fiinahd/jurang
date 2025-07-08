import nltk
nltk.download('averaged_perceptron_tagger')

def pos_tags(tokens: list[str]) -> list[tuple[str,str]]:
    return nltk.pos_tag(tokens)