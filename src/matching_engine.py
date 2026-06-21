import joblib
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.skill_extractor import calculate_skill_gap, DEFAULT_SKILLS


class ResumeJDMatcher:
    def __init__(self, artifact_dir=None, model_name="all-MiniLM-L6-v2", skills=None):
        self.artifact_dir = Path(artifact_dir) if artifact_dir else None
        self.embedding_model = SentenceTransformer(model_name)
        self.skills = skills or DEFAULT_SKILLS
        self.classifier = None
        self.label_encoder = None

        if self.artifact_dir:
            self._load_artifacts()

    def _load_artifacts(self):
        classifier_path = self.artifact_dir / "xgb_classifier.joblib"
        encoder_path = self.artifact_dir / "label_encoder.joblib"

        if classifier_path.exists() and encoder_path.exists():
            try:
                self.classifier = joblib.load(classifier_path)
                self.label_encoder = joblib.load(encoder_path)
            except Exception:
                self.classifier = None
                self.label_encoder = None

    def _embed_texts(self, jd_text: str, resume_text: str):
        embeddings = self.embedding_model.encode(
            [jd_text, resume_text],
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings[0], embeddings[1]

    def semantic_similarity(self, jd_text: str, resume_text: str) -> float:
        jd_embedding, resume_embedding = self._embed_texts(jd_text, resume_text)
        similarity = cosine_similarity([jd_embedding], [resume_embedding])[0][0]
        return round(float(similarity) * 100, 2)

    def build_features(self, jd_text: str, resume_text: str):
        jd_embedding, resume_embedding = self._embed_texts(jd_text, resume_text)
        return np.hstack([
            jd_embedding,
            resume_embedding,
            np.abs(jd_embedding - resume_embedding),
        ])

    @staticmethod
    def get_fit_category(final_score: float):
        if final_score >= 75:
            return "Good Fit", "Strongly Recommend"
        elif final_score >= 40:
            return "Potential Fit", "Recommend for Screening"
        else:
            return "No Fit", "Not Recommended"

    def predict_candidate(self, jd_text: str, resume_text: str):
        gap = calculate_skill_gap(jd_text, resume_text, self.skills)
        skill_score = gap["skill_score"]
        semantic_score = self.semantic_similarity(jd_text, resume_text)
        final_score = round((0.6 * skill_score) + (0.4 * semantic_score), 2)
        fit_category, recommendation = self.get_fit_category(final_score)

        classifier_label = None
        classifier_probability = None

        if self.classifier is not None and self.label_encoder is not None:
            features = self.build_features(jd_text, resume_text).reshape(1, -1)
            try:
                pred = self.classifier.predict(features)
                classifier_label = self.label_encoder.inverse_transform(pred)[0]
                if hasattr(self.classifier, "predict_proba"):
                    proba = self.classifier.predict_proba(features)[0]
                    classifier_probability = round(float(max(proba)) * 100, 2)
            except Exception:
                classifier_label = None
                classifier_probability = None

        return {
            "skill_score": skill_score,
            "semantic_score": semantic_score,
            "final_score": final_score,
            "fit_category": fit_category,
            "recommendation": recommendation,
            "classifier_label": classifier_label,
            "classifier_probability": classifier_probability,
            "jd_skills": gap["jd_skills"],
            "resume_skills": gap["resume_skills"],
            "matched_skills": gap["matched_skills"],
            "missing_skills": gap["missing_skills"],
        }
