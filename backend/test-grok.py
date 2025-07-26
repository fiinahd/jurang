import pandas as pd
import stanza
import os
import json
import uuid
import io
from fastapi.testclient import TestClient
from app.main import app
from app.logic.l_1_preprocess import _preprocess_text_internal, run_preprocessing
from app.logic.l_2_postag import run_postagging
from app.logic.l_3_extraction import _extract_aspects_internal, run_extraction
from app.logic.l_4_training import run_training_pipeline
from app.routers.analysis import _generate_visualization_data
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Initialize Stanza and Sastrawi
stanza.download('id', verbose=False)
nlp = stanza.Pipeline('id', processors='tokenize,pos,lemma', use_gpu=False, verbose=False)
stemmer = StemmerFactory().create_stemmer()

# Initialize FastAPI test client
client = TestClient(app)

# Helper to create temporary Excel/CSV files
def create_temp_excel(data, filename):
    df = pd.DataFrame(data)
    df.to_excel(filename, index=False)
    return filename

def create_temp_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    return filename

def print_test_case(case_num, module, input_data, expected_output, actual_output, passed):
    print(f"\n--- Test Case {case_num} ---")
    print(f"Module: {module}")
    print(f"Input: {repr(input_data)}")
    print(f"Expected: {repr(expected_output)}")
    print(f"Actual: {repr(actual_output)}")
    print(f"Status: {'PASS' if passed else 'FAIL'}")

def run_tests():
    print("="*50)
    print("Starting Whitebox Testing for senna")
    print("="*50)

    # Ensure directories exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("models_trained", exist_ok=True)
    process_id = str(uuid.uuid4())

    # Test 1-2: Preprocessing (_preprocess_text_internal)
    test_cases = [
        (1, "_preprocess_text_internal", "Kualitas BAGUS!!, harganya oke.", "kualitas bagus harga oke", lambda x: x == "kualitas bagus harga oke"),
        (2, "_preprocess_text_internal", "😊👍", "emotikon terdeteksi", lambda x: x == "emotikon terdeteksi"),
    ]
    for case_num, module, input_data, expected, check in test_cases:
        try:
            actual = _preprocess_text_internal(input_data, stemmer)
            passed = check(actual)
        except Exception as e:
            actual = f"Error: {e}"
            passed = False
        print_test_case(case_num, module, input_data, expected, actual, passed)

    # Test 3: Preprocessing (run_preprocessing)
    test_cases = [
        (3, "run_preprocessing", {"input_path": create_temp_excel({"ulasan": ["Kualitas bagus", "Harga oke"]}, f"data/raw_{process_id}.xlsx"), "output_path": f"data/cleaned_{process_id}.csv", "review_column": "ulasan"}, ["kualitas bagus", "harga oke"], lambda x: list(pd.read_csv(x)["cleaned_review"]) == ["kualitas bagus", "harga oke"]),
    ]
    for case_num, module, input_data, expected, check in test_cases:
        try:
            run_preprocessing(process_id, **input_data)
            actual = input_data["output_path"]
            passed = check(actual) if os.path.exists(actual) else False
        except Exception as e:
            actual = f"Error: {e}"
            passed = False
        print_test_case(case_num, module, input_data, expected, actual, passed)

    # Test 4-5: POS Tagging (run_postagging)
    test_cases = [
        (4, "run_postagging", {"input_csv": create_temp_csv({"cleaned_review": ["desain modern bahan kuat"]}, f"data/cleaned_{process_id}_postag.csv"), "top_n": 30}, ["desain", "bahan"], lambda x: set(json.load(open(x))["aspects"]) == {"desain", "bahan"}),
        (5, "run_postagging", {"input_csv": create_temp_csv({"cleaned_review": ["123"]}, f"data/invalid_{process_id}.csv"), "top_n": 30}, ["angka"], lambda x: set(json.load(open(x))["aspects"]) == {"angka"}),
    ]
    for case_num, module, input_data, expected, check in test_cases:
        try:
            run_postagging(process_id, **input_data)
            actual = f"data/aspects_{process_id}.json"
            passed = check(actual) if os.path.exists(actual) else False
        except Exception as e:
            actual = f"Error: {e}"
            passed = False
        print_test_case(case_num, module, input_data, expected, actual, passed)

    # Test 6: Aspect Extraction (_extract_aspects_internal)
    test_cases = [
        (6, "_extract_aspects_internal", {"text": "kualitas bagus", "aspects": {"kualitas", "harga"}}, ["kualitas"], lambda x: x == ["kualitas"]),
    ]
    for case_num, module, input_data, expected, check in test_cases:
        try:
            actual = _extract_aspects_internal(input_data["text"], input_data["aspects"])
            passed = check(actual)
        except Exception as e:
            actual = f"Error: {e}"
            passed = False
        print_test_case(case_num, module, input_data, expected, actual, passed)

    # Test 7: Aspect Extraction (run_extraction)
    test_cases = [
        (7, "run_extraction", {"input_path": create_temp_csv({"cleaned_review": [""]}, f"data/cleaned_{process_id}_extract.csv"), "output_path": f"data/extracted_{process_id}.csv", "selected_aspects": ["kualitas"]}, "Empty CSV", lambda x: os.path.exists(x) and len(pd.read_csv(x)) == 0),
    ]
    for case_num, module, input_data, expected, check in test_cases:
        try:
            run_extraction(**input_data)
            actual = input_data["output_path"]
            passed = check(actual)
        except Exception as e:
            actual = f"Error: {e}"
            passed = False
        print_test_case(case_num, module, input_data, expected, actual, passed)

    # Test 8: Labeling (/api/process/{process_id}/extract)
    create_temp_csv({"cleaned_review": ["kualitas bagus"]*100}, f"data/cleaned_{process_id}.csv")
    test_cases = [
        (8, "Labeling API", {"process_id": process_id, "payload": {"aspects": ["kualitas"], "sampling_percentage": 50}}, "~50 rows", lambda x: 40 <= len(x.get("labeling_data", [])) <= 60),
    ]
    for case_num, module, input_data, expected, check in test_cases:
        try:
            response = client.post(f"/api/process/{input_data['process_id']}/extract", json=input_data["payload"])
            actual = response.json() if response.status_code == 200 else f"Error: {response.status_code}"
            passed = check(response.json()) if response.status_code == 200 else False
        except Exception as e:
            actual = f"Error: {e}"
            passed = False
        print_test_case(case_num, module, input_data, expected, actual, passed)

    # Test 9: Training and Classification (run_training_pipeline)
    test_cases = [
        (9, "run_training_pipeline", {"labeled_csv": create_temp_csv({"cleaned_review": ["kualitas bagus"], "detected_aspects": ["kualitas"], "sentiment": ["positif"]}, f"data/labeled_{process_id}.csv")}, "Model and predictions", lambda x: os.path.exists(x)),
    ]
    for case_num, module, input_data, expected, check in test_cases:
        try:
            run_training_pipeline(process_id, input_data["labeled_csv"])
            actual = f"models_trained/model_{process_id}.joblib"
            passed = check(actual)
        except Exception as e:
            actual = f"Error: {e}"
            passed = False
        print_test_case(case_num, module, input_data, expected, actual, passed)

    # Test 10: Visualization (_generate_visualization_data)
    test_cases = [
        (10, "_generate_visualization_data", pd.DataFrame(), {"net_sentiment_scores": [{"aspect": "kualitas", "score": 0}], "aspect_details": {}, "all_aspects": []}, lambda x: len(x["net_sentiment_scores"]) > 0),
    ]
    for case_num, module, input_data, expected, check in test_cases:
        try:
            actual = _generate_visualization_data(input_data)
            passed = check(actual)
        except Exception as e:
            actual = f"Error: {e}"
            passed = False
        print_test_case(case_num, module, input_data.to_dict(), expected, actual, passed)

    print("\n" + "="*50)
    print("TESTING COMPLETED")
    print("="*50)

if __name__ == "__main__":
    run_tests()