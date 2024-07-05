import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer
import numpy as np
import logging
from fuzzywuzzy import process, fuzz
from typing import List, Tuple, Dict, Any


class EntityExtractor:
    def __init__(self, similarity_threshold: float = 0.5, fuzzy_threshold: int = 80):
        self.nlp = spacy.load("en_core_web_trf")
        self.tfidf_vectorizer = TfidfVectorizer(stop_words="english")
        self.sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")
        self.similarity_threshold = similarity_threshold
        self.fuzzy_threshold = fuzzy_threshold
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def extract_entities(self, text: str) -> List[Tuple[str, int, int, str]]:
        doc = self.nlp(text)

        ner_entities = [
            (ent.text, ent.start_char, ent.end_char, "NER") for ent in doc.ents
        ]
        noun_phrases = [
            (np.text, np.start_char, np.end_char, "NOUN_PHRASE")
            for np in doc.noun_chunks
            if np.text.lower() not in self.nlp.Defaults.stop_words
        ]
        tfidf_entities = [
            (kw, start, end, "KEYWORD")
            for kw, start, end in self.extract_keywords(text)
        ]

        all_entities = ner_entities + noun_phrases + tfidf_entities

        seen = set()
        unique_entities = []
        for entity in all_entities:
            if entity[0].lower() not in seen:
                seen.add(entity[0].lower())
                unique_entities.append(entity)

        return unique_entities

    def extract_keywords(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Extract keywords using TF-IDF.

        Args:
        text (str): The input text to extract keywords from.

        Returns:
        List[Tuple[str, int, int]]: A list of tuples containing the keyword text, start index, and end index.
        """
        # Fit and transform the text
        tfidf_matrix = self.tfidf_vectorizer.fit_transform([text])
        feature_names = self.tfidf_vectorizer.get_feature_names_out()

        # Get top keywords
        top_n = 10  # Adjust this value to extract more or fewer keywords
        top_indices = tfidf_matrix.indices[tfidf_matrix.data.argsort()[-top_n:][::-1]]
        top_keywords = [feature_names[i] for i in top_indices]

        # Find the positions of keywords in the original text
        keyword_positions = []
        for keyword in top_keywords:
            start = text.lower().find(keyword.lower())
            if start != -1:
                end = start + len(keyword)
                keyword_positions.append((keyword, start, end))

        return keyword_positions

    def match_entities(
        self,
        entities: List[Tuple[str, int, int, str]],
        ontology_entities: Dict[str, Dict[str, Any]],
    ) -> List[Tuple[str, int, int, str, str, float, str]]:
        matched_entities = []

        for entity, start, end, entity_type in entities:
            processed_entity = entity.lower()
            entity_embedding = self.sentence_transformer.encode(processed_entity)

            best_match = None
            best_similarity = -1
            best_match_type = None

            for iri, ont_entity in ontology_entities.items():
                label = ont_entity.get("rdfs:label", "").lower()

                # Semantic similarity matching
                if "embedding" in ont_entity:
                    similarity = self.cosine_similarity(
                        entity_embedding, ont_entity["embedding"]
                    )
                    if (
                        similarity > best_similarity
                        and similarity >= self.similarity_threshold
                    ):
                        best_match = iri
                        best_similarity = similarity
                        best_match_type = "semantic"

                # Fuzzy matching
                fuzzy_score = fuzz.token_set_ratio(processed_entity, label)
                if (
                    fuzzy_score > best_similarity * 100
                    and fuzzy_score >= self.fuzzy_threshold
                ):
                    best_match = iri
                    best_similarity = fuzzy_score / 100
                    best_match_type = "fuzzy"

            if best_match:
                matched_entities.append(
                    (
                        entity,
                        start,
                        end,
                        entity_type,
                        best_match,
                        best_similarity,
                        best_match_type,
                    )
                )
            else:
                self.logger.info(
                    f"Unmatched term: {entity} (best similarity: {best_similarity:.2f})"
                )
                matched_entities.append(
                    (
                        entity,
                        start,
                        end,
                        entity_type,
                        None,
                        best_similarity,
                        "unmatched",
                    )
                )

        return matched_entities

    @staticmethod
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


# Example usage
# if __name__ == "__main__":
#     extractor = EntityExtractor(similarity_threshold=0.5, fuzzy_threshold=80)

#     # Example text
#     text = "The intellectual property lawyer specializes in patent law and copyright infringement cases."

#     # Extract entities
#     entities = extractor.extract_entities(text)
#     print("Extracted entities:")
#     for entity, start, end, entity_type in entities:
#         print(f"- {entity} ({start}, {end}) [{entity_type}]")

#     # Example branch embeddings and labels (in a real scenario, these would come from the ontology parser)
#     branch_embeddings = {
#         "Intellectual Property": extractor.sentence_transformer.encode(
#             "Intellectual Property"
#         ).tolist(),
#         "Criminal Law": extractor.sentence_transformer.encode("Criminal Law").tolist(),
#         "Corporate Law": extractor.sentence_transformer.encode(
#             "Corporate Law"
#         ).tolist(),
#     }
#     branch_labels = [
#         "Intellectual Property",
#         "Criminal Law",
#         "Corporate Law",
#         "Patent Law",
#         "Copyright Law",
#     ]

#     # Classify entities
#     classified_entities = extractor.classify_entities(
#         entities, branch_embeddings, branch_labels
#     )
#     print("\nClassified entities:")
#     for (
#         entity,
#         start,
#         end,
#         entity_type,
#         branch,
#         similarity,
#         match_type,
#     ) in classified_entities:
#         print(
#             f"- {entity} ({start}, {end}) [{entity_type}]: {branch} (similarity: {similarity:.2f}, match type: {match_type})"
#         )
