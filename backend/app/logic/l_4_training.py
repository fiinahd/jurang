import pandas as pd
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

def run_training_pipeline(process_id: str, labeled_data_path: str):
    try:
        # --- 5.1: Split Data ---
        df_labeled = pd.read_csv(labeled_data_path)
        df_labeled = df_labeled.dropna(subset=['detected_aspects', 'sentiment'])
        
        df_exploded = (
            df_labeled
            .assign(aspect=df_labeled['detected_aspects'].str.split(';'))
            .explode('aspect')
            .reset_index(drop=True)
        )
        df_exploded = df_exploded[df_exploded['aspect'] != '']
        
        try:
            strata_counts = df_exploded['sentiment'].value_counts()
            if strata_counts.min() < 2:
                print("Peringatan: Beberapa kelas sentimen memiliki kurang dari 2 sampel. Menggunakan split tanpa stratifikasi.")
                stratify_param = None
            else:
                stratify_param = df_exploded['sentiment']
                
            X = df_exploded
            y = df_exploded['sentiment']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=stratify_param
            )
        except Exception:
             X = df_exploded
             y = df_exploded['sentiment']
             X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=None
            )
        
        train_df = X_train.copy()
        test_df = X_test.copy()

        # --- 5.2: Train Model ---
        train_df['input'] = train_df['aspect'].astype(str) + " " + train_df['cleaned_review'].astype(str)
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1,2))),
            ("knn",   KNeighborsClassifier(n_neighbors=5, metric="cosine"))
        ])
        pipeline.fit(train_df['input'], train_df['sentiment'])
        model_path = os.path.join("models_trained", f"model_{process_id}.joblib")
        joblib.dump(pipeline, model_path)

        # --- 5.3: Evaluate Model ---
        test_df['input'] = test_df['aspect'].astype(str) + " " + test_df['cleaned_review'].astype(str)
        y_pred = pipeline.predict(test_df['input'])
        
        labels = sorted(test_df['sentiment'].unique())
        cm = confusion_matrix(test_df['sentiment'], y_pred, labels=labels)
        
        evaluation_results = {
            "report": classification_report(test_df['sentiment'], y_pred, digits=2, zero_division=0),
            "matrix": {"labels": labels, "values": cm.tolist()}
        }
        with open(os.path.join("models_trained", f"evaluation_{process_id}.json"), 'w') as f:
            json.dump(evaluation_results, f, indent=4)

        # --- 5.4: Predict on All Data ---
        all_data_path = os.path.join("data", f"extracted_{process_id}.csv")
        df_all = pd.read_csv(all_data_path)
        df_all_exploded = (
            df_all
            .assign(aspect=df_all['detected_aspects'].str.split(';'))
            .explode('aspect')
            .reset_index(drop=True)
        )
        df_all_exploded = df_all_exploded[df_all_exploded['aspect'] != '']
        df_all_exploded['input'] = df_all_exploded['aspect'].astype(str) + " " + df_all_exploded['cleaned_review'].astype(str)
        df_all_exploded['predicted_sentiment'] = pipeline.predict(df_all_exploded['input'])
        
        final_prediction_path = os.path.join("data", f"final_predictions_{process_id}.csv")
        df_all_exploded.to_csv(final_prediction_path, index=False)

        print(f"Training pipeline untuk process_id {process_id} selesai.")
    except Exception as e:
        print(f"ERROR DALAM PROSES TRAINING: {e}")