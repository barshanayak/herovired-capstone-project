import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.matching_engine import ResumeJDMatcher
from src.preprocessing import extract_jd_resume
from src.skill_extractor import calculate_skill_gap


def build_embedding_features(matcher: ResumeJDMatcher, jd_texts, resume_texts):
    jd_embeddings = matcher.embedding_model.encode(
        jd_texts,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    resume_embeddings = matcher.embedding_model.encode(
        resume_texts,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    similarity_scores = cosine_similarity(jd_embeddings, resume_embeddings).diagonal().reshape(-1, 1)

    return np.hstack([
        jd_embeddings,
        resume_embeddings,
        np.abs(jd_embeddings - resume_embeddings),
        similarity_scores,
    ])


def build_skill_features(jd_texts, resume_texts):
    skill_scores = []
    matched_counts = []
    missing_counts = []
    jd_skill_counts = []
    resume_skill_counts = []
    jd_lengths = []
    resume_lengths = []

    for jd_text, resume_text in zip(jd_texts, resume_texts):
        gap = calculate_skill_gap(jd_text, resume_text)
        skill_scores.append(gap["skill_score"])
        matched_counts.append(len(gap["matched_skills"]))
        missing_counts.append(len(gap["missing_skills"]))
        jd_skill_counts.append(len(gap["jd_skills"]))
        resume_skill_counts.append(len(gap["resume_skills"]))
        jd_lengths.append(len(str(jd_text).split()))
        resume_lengths.append(len(str(resume_text).split()))

    delta_lengths = list(np.array(resume_lengths) - np.array(jd_lengths))

    return np.vstack([
        skill_scores,
        matched_counts,
        missing_counts,
        jd_skill_counts,
        resume_skill_counts,
        jd_lengths,
        resume_lengths,
        delta_lengths,
    ]).T


def build_features(matcher: ResumeJDMatcher, jd_texts, resume_texts):
    embedding_features = build_embedding_features(matcher, jd_texts, resume_texts)
    skill_features = build_skill_features(jd_texts, resume_texts)
    return np.hstack([embedding_features, skill_features])


def tune_model(X_train, y_train, sample_weights):
    search_space = {
        "n_estimators": [100, 200, 300],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
    }

    base_model = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=search_space,
        n_iter=8,
        scoring="accuracy",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        verbose=2,
        random_state=42,
        n_jobs=1,
        refit=True,
    )

    search.fit(X_train, y_train, sample_weight=sample_weights)
    print("Best hyperparameters:", search.best_params_)
    print("Best CV accuracy:", round(search.best_score_, 4))
    return search.best_estimator_, search.best_params_


def load_and_prepare_dataset():
    dataset = load_dataset("facehuggerapoorv/resume-jd-match")
    train_df = pd.DataFrame(dataset["train"])
    test_df = pd.DataFrame(dataset["test"])

    train_df[["job_description", "resume"]] = train_df["text"].apply(
        lambda x: pd.Series(extract_jd_resume(x))
    )
    test_df[["job_description", "resume"]] = test_df["text"].apply(
        lambda x: pd.Series(extract_jd_resume(x))
    )

    train_df.dropna(subset=["job_description", "resume"], inplace=True)
    test_df.dropna(subset=["job_description", "resume"], inplace=True)

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, test_df


def create_artifact_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def main():
    artifact_dir = create_artifact_dir(Path(__file__).resolve().parents[1] / "models")
    matcher = ResumeJDMatcher()

    train_df, test_df = load_and_prepare_dataset()

    X_full = build_features(
        matcher,
        train_df["job_description"].tolist(),
        train_df["resume"].tolist(),
    )
    y_full = train_df["label"].tolist()

    X_train, X_val, y_train, y_val = train_test_split(
        X_full,
        y_full,
        test_size=0.2,
        random_state=42,
        stratify=y_full,
    )

    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_val_encoded = encoder.transform(y_val)
    y_full_encoded = encoder.transform(y_full)

    train_sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train_encoded,
    )

    best_model, best_params = tune_model(
        X_train,
        y_train_encoded,
        train_sample_weights,
    )

    y_val_pred = best_model.predict(X_val)
    print("Validation Accuracy:", accuracy_score(y_val_encoded, y_val_pred))
    print(classification_report(y_val_encoded, y_val_pred))

    full_sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_full_encoded,
    )

    print("Retraining final model on all training data with best hyperparameters...")
    final_model = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        **best_params,
    )
    final_model.fit(X_full, y_full_encoded, sample_weight=full_sample_weights)

    X_test = build_features(
        matcher,
        test_df["job_description"].tolist(),
        test_df["resume"].tolist(),
    )
    y_test_encoded = encoder.transform(test_df["label"].tolist())

    y_test_pred = final_model.predict(X_test)
    print("Test Accuracy:", accuracy_score(y_test_encoded, y_test_pred))
    print(classification_report(y_test_encoded, y_test_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test_encoded, y_test_pred))

    joblib.dump(final_model, artifact_dir / "xgb_classifier.joblib")
    joblib.dump(encoder, artifact_dir / "label_encoder.joblib")
    print(f"Saved artifacts to {artifact_dir}")


if __name__ == "__main__":
    main()
