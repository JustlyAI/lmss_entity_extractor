import json
import logging
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import fuzz
from rdflib import Graph, URIRef, RDFS, Literal
from rdflib.namespace import Namespace

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OntologyClassifier:
    def __init__(
        self, graph_path: str, index_path: str, similarity_threshold: float = 0.65
    ):
        self.graph = Graph()
        self.graph.parse(graph_path, format="turtle")
        self.ontology_entities = self._load_ontology_index(index_path)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.top_classes = self._get_top_classes()
        self.similarity_threshold = similarity_threshold
        self.LMSS = Namespace("http://lmss.sali.org/")
        logger.info(f"Loaded {len(self.ontology_entities)} ontology entities")
        logger.info(f"Identified {len(self.top_classes)} top classes")

    def _load_ontology_index(self, index_path: str) -> List[Dict[str, Any]]:
        with open(index_path, "r") as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} entities from index")
        return data

    def _get_top_classes(self) -> Dict[str, str]:
        top_classes = {}
        for entity in self.ontology_entities:
            if (
                "subClassOf" in entity
                and len(entity["subClassOf"]) == 1
                and entity["subClassOf"][0] == "http://www.w3.org/2002/07/owl#Thing"
            ):
                top_classes[entity["rdf_about"]] = entity["rdfs_label"]
        return top_classes

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def _get_entity_embedding(self, entity_iri: str) -> np.ndarray:
        embeddings = []
        for _, _, embedding_node in self.graph.triples(
            (URIRef(entity_iri), self.LMSS.hasEmbedding, None)
        ):
            embedding_value = self.graph.value(embedding_node, self.LMSS.embeddingValue)
            if embedding_value:
                embeddings.append(np.array(json.loads(embedding_value)))
        if embeddings:
            return np.mean(embeddings, axis=0)
        return None

    def _find_best_match(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        best_match = None
        best_score = 0
        entity_embedding = np.array(entity["vector"])
        entity_text = entity["text"].lower()

        # Multi-word matching
        entity_words = entity_text.split()
        entity_bigrams = [
            " ".join(entity_words[i : i + 2]) for i in range(len(entity_words) - 1)
        ]
        entity_trigrams = [
            " ".join(entity_words[i : i + 3]) for i in range(len(entity_words) - 2)
        ]

        for ont_entity in self.ontology_entities:
            ont_embedding = self._get_entity_embedding(ont_entity["rdf_about"])

            if ont_embedding is not None:
                semantic_score = self._cosine_similarity(
                    entity_embedding, ont_embedding
                )
            else:
                semantic_score = 0

            label_lower = ont_entity["rdfs_label"].lower()
            fuzzy_score = max(
                [
                    fuzz.token_set_ratio(entity_text, label_lower) / 100,
                    max(
                        [
                            fuzz.partial_ratio(bigram, label_lower) / 100
                            for bigram in entity_bigrams
                        ]
                        or [0]
                    ),
                    max(
                        [
                            fuzz.partial_ratio(trigram, label_lower) / 100
                            for trigram in entity_trigrams
                        ]
                        or [0]
                    ),
                ]
            )

            combined_score = (semantic_score + fuzzy_score) / 2

            if combined_score > best_score:
                best_score = combined_score
                best_match = {
                    "iri": ont_entity["rdf_about"],
                    "label": ont_entity["rdfs_label"],
                    "score": best_score,
                }

        if best_match and best_match["score"] >= self.similarity_threshold:
            logger.info(
                f"Best match for '{entity['text']}': {best_match['label']} (score: {best_match['score']:.2f})"
            )
            return best_match
        else:
            logger.warning(
                f"No match found for entity: {entity['text']} (best score: {best_score:.2f})"
            )
            return None

    def _post_process_match(
        self, entity: Dict[str, Any], match: Dict[str, Any]
    ) -> Dict[str, Any]:
        # Handle location entities
        if (
            entity["type"] == "NOUN_PHRASE"
            and self._get_branch(match["iri"]) == "Location"
        ):
            # Here you could implement a more sophisticated location matching logic
            # For simplicity, we'll just keep the match if it's already a location
            return match

        # Handle general verbs
        if entity["type"] == "VERB" and match["score"] < 0.7:
            return None

        return match

    def match_entities(
        self, extracted_entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        results = []
        for entity in extracted_entities:
            best_match = self._find_best_match(entity)
            if best_match:
                best_match = self._post_process_match(entity, best_match)
                if best_match:
                    branch = self._get_branch(best_match["iri"])
                    logger.info(f"Found branch for {best_match['label']}: {branch}")
                    result = {
                        "start": entity["start"],
                        "end": entity["end"],
                        "text": entity["text"],
                        "branch": branch,
                        "label": best_match["label"],
                        "score": best_match["score"],
                        "iri": best_match["iri"],
                    }
                    results.append(result)
                else:
                    logger.warning(
                        f"Match removed after post-processing: {entity['text']}"
                    )
            else:
                logger.warning(f"No match found for entity: {entity['text']}")

        logger.info(f"Matched {len(results)} out of {len(extracted_entities)} entities")
        logger.info(f"Results: {results}")  # Log the entire results list
        return results

    def _get_branch(self, entity_iri: str) -> str:
        logger.info(f"Getting branch for entity: {entity_iri}")
        for parent in self.graph.transitive_objects(
            URIRef(entity_iri), RDFS.subClassOf
        ):
            logger.info(f"Checking parent: {parent}")
            if str(parent) in self.top_classes:
                logger.info(f"Found top class: {self.top_classes[str(parent)]}")
                return self.top_classes[str(parent)]
        logger.warning(f"No branch found for entity: {entity_iri}")
        return "Unknown"

    def print_ontology_sample(self, n: int = 5):
        logger.info(f"Sample of {n} ontology entities:")
        for entity in self.ontology_entities[:n]:
            logger.info(f"Label: {entity['rdfs_label']}, IRI: {entity['rdf_about']}")
            embedding = self._get_entity_embedding(entity["rdf_about"])
            if embedding is not None:
                logger.info(f"Embedding: {embedding[:5]}...")
            else:
                logger.info("No embedding found")
            logger.info("---")
