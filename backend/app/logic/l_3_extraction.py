import pandas as pd
from typing import List

def _extract_aspects_internal(text: str, domain_aspects: set) -> list:
    if not isinstance(text, str): return []
    tokens = text.split()
    return sorted(list(domain_aspects.intersection(tokens)))

def run_extraction(input_path: str, output_path: str, selected_aspects: List[str]):
    df = pd.read_csv(input_path)
    df['cleaned_review'] = df['cleaned_review'].fillna('')
    
    domain_aspects_set = set(selected_aspects)
    
    results = []
    for _, row in df.iterrows():
        review = row.cleaned_review
        aspects = _extract_aspects_internal(review, domain_aspects_set)
        
        if aspects:
            results.append({
                'product_name': row.get('product_name', ''),
                'cleaned_review': review,
                'detected_aspects': ";".join(aspects),
            })

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_path, index=False)
    print(f"Ekstraksi selesai. Hasil disimpan di {output_path}")