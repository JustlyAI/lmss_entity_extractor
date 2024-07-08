from pydantic import BaseModel, Field
import numpy as np
from typing import List, Dict, Any, Tuple, Set
from fuzzywuzzy import fuzz
import logging
from rdflib import Graph, URIRef, RDFS, OWL
from collections import defaultdict
import json
import nltk
from nltk.corpus import stopwords


# Define Pydantic models for structured data
class Entity(BaseModel):
    rdf_about: str
    rdfs_label: str
    description: str = ""
    rdfs_seeAlso: List[str] = Field(default_factory=list)
    skos_altLabel: List[str] = Field(default_factory=list)
    skos_definition: str = ""
    skos_example: List[str] = Field(default_factory=list)
    skos_prefLabel: str = ""
    subClassOf: List[str] = Field(default_factory=list)
    embedding: List[float] = Field(default_factory=list)


class MatchResult(BaseModel):
    iri: str = ""
    label: str = ""
    similarity: float = 0.0
    match_type: str = "unmatched"


class OntologyMatcher:
    def __init__(
        self,
        graph_path: str,
        index_path: str,
        similarity_threshold: float = 0.5,
        fuzzy_threshold: int = 80,
        context_window: int = 2,
    ):
        self.graph = Graph()
        self.graph.parse(graph_path, format="turtle")
        self.ontology_entities = self._load_ontology_index(index_path)
        self.similarity_threshold = similarity_threshold
        self.fuzzy_threshold = fuzzy_threshold
        self.context_window = context_window
        self.logger = logging.getLogger(__name__)
        self.hierarchy_cache = self._build_hierarchy_cache()

        nltk.download("stopwords", quiet=True)
        self.stop_words = set(stopwords.words("english"))

    def _load_ontology_index(self, index_path: str) -> Dict[str, Entity]:
        with open(index_path, "r") as f:
            data = json.load(f)
        return {entity["rdf_about"]: Entity(**entity) for entity in data}

    def _build_hierarchy_cache(self) -> Dict[str, List[str]]:
        hierarchy_cache = defaultdict(list)
        for s, p, o in self.graph.triples((None, RDFS.subClassOf, None)):
            if isinstance(s, URIRef) and isinstance(o, URIRef):
                hierarchy_cache[str(s)].append(str(o))
        return hierarchy_cache

    def _filter_ontology(self, selected_classes: List[str]) -> Set[str]:
        filtered_entities = set()
        for class_iri in selected_classes:
            filtered_entities.add(class_iri)
            filtered_entities.update(self._get_subclasses(URIRef(class_iri)))
        return filtered_entities

    def _get_subclasses(self, class_iri: URIRef) -> Set[str]:
        subclasses = set()
        for s, p, o in self.graph.triples((None, RDFS.subClassOf, class_iri)):
            subclasses.add(str(s))
            subclasses.update(self._get_subclasses(s))
        return subclasses

    def match_entities(
        self, extracted_data: List[Dict[str, Any]], selected_classes: List[str] = None
    ) -> List[Dict[str, Any]]:
        matched_entities = []

        if selected_classes:
            filtered_entities = self._filter_ontology(selected_classes)
        else:
            filtered_entities = set(self.ontology_entities.keys())

        for i, entity in enumerate(extracted_data):
            context = self._get_context(extracted_data, i)
            match_result = self._find_best_match_with_context(
                entity, context, filtered_entities
            )
            entity["match"] = match_result.dict()
            if match_result.iri:
                entity["hierarchy"] = self._get_entity_hierarchy(match_result.iri)
                entity["classification"] = self.classify_hierarchically(entity)
            else:
                entity["hierarchy"] = []
                entity["classification"] = []
                self._apply_fallback_matching(entity, context, filtered_entities)
            matched_entities.append(entity)

        return matched_entities

    def _get_context(self, entities: List[Dict[str, Any]], index: int) -> str:
        start = max(0, index - self.context_window)
        end = min(len(entities), index + self.context_window + 1)
        context_entities = entities[start:end]
        return " ".join(entity["text"] for entity in context_entities)

    def _find_best_match_with_context(
        self, entity: Dict[str, Any], context: str, filtered_entities: Set[str]
    ) -> MatchResult:
        processed_entity = entity["text"].lower()
        processed_context = context.lower()
        entity_embedding = np.array(entity["vector"])

        best_match = ""
        best_similarity = -1
        best_match_type = "unmatched"
        best_match_label = ""

        for iri, ont_entity in self.ontology_entities.items():
            if iri not in filtered_entities:
                continue

            label = ont_entity.rdfs_label.lower()

            # Context-aware matching
            context_similarity = self._context_similarity(processed_context, label)

            # Semantic similarity
            if ont_entity.embedding:
                semantic_similarity = self._cosine_similarity(
                    entity_embedding, np.array(ont_entity.embedding)
                )
                combined_similarity = (semantic_similarity + context_similarity) / 2

                if (
                    combined_similarity > best_similarity
                    and combined_similarity >= self.similarity_threshold
                ):
                    best_match = iri
                    best_similarity = combined_similarity
                    best_match_type = "semantic+context"
                    best_match_label = ont_entity.rdfs_label

            # Fuzzy matching with context consideration
            fuzzy_score = fuzz.token_set_ratio(processed_entity, label)
            fuzzy_similarity = fuzzy_score / 100
            combined_fuzzy_similarity = (fuzzy_similarity + context_similarity) / 2

            if (
                combined_fuzzy_similarity > best_similarity
                and fuzzy_score >= self.fuzzy_threshold
            ):
                best_match = iri
                best_similarity = combined_fuzzy_similarity
                best_match_type = "fuzzy+context"
                best_match_label = ont_entity.rdfs_label

        if not best_match:
            self.logger.info(
                f"Unmatched term: {entity['text']} (best similarity: {best_similarity:.2f})"
            )

        return MatchResult(
            iri=best_match,
            label=best_match_label,
            similarity=best_similarity,
            match_type=best_match_type,
        )

    def _context_similarity(self, context: str, label: str) -> float:
        context_words = set(w for w in context.split() if w not in self.stop_words)
        label_words = set(w for w in label.split() if w not in self.stop_words)
        intersection = context_words.intersection(label_words)
        return len(intersection) / max(len(context_words), len(label_words))

    def _get_entity_hierarchy(self, entity_iri: str) -> List[str]:
        hierarchy = []
        current_entity = entity_iri

        while current_entity and current_entity != str(OWL.Thing):
            hierarchy.append(current_entity)
            parents = self.hierarchy_cache.get(current_entity, [])
            if not parents:
                break
            current_entity = parents[0]  # Assuming single inheritance

        if current_entity == str(OWL.Thing):
            hierarchy.append(str(OWL.Thing))

        return hierarchy[::-1]  # Reverse to get root-to-leaf order

    @staticmethod
    def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def classify_hierarchically(
        self, matched_entity: Dict[str, Any]
    ) -> List[Tuple[str, float]]:
        if not matched_entity["match"]["iri"]:
            return []

        hierarchy = self._get_entity_hierarchy(matched_entity["match"]["iri"])
        classified_hierarchy = []

        for level, class_iri in enumerate(hierarchy):
            class_label = self.graph.value(URIRef(class_iri), RDFS.label)
            confidence = max(
                0, 1 - (level * 0.1)
            )  # Decrease confidence as we go up the hierarchy
            classified_hierarchy.append((str(class_label), confidence))

        return classified_hierarchy

    def graph_based_matching(
        self, entity: Dict[str, Any], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        entity_embedding = np.array(entity["vector"])
        candidates = []

        for iri, ont_entity in self.ontology_entities.items():
            if ont_entity.embedding:
                similarity = self._cosine_similarity(
                    entity_embedding, np.array(ont_entity.embedding)
                )
                candidates.append(
                    {
                        "iri": iri,
                        "similarity": similarity,
                        "label": ont_entity.rdfs_label,
                    }
                )

        candidates.sort(key=lambda x: x["similarity"], reverse=True)
        top_candidates = candidates[:top_k]

        # Enhance matching using graph structure
        for candidate in top_candidates:
            candidate["parents"] = self.hierarchy_cache.get(candidate["iri"], [])
            candidate["children"] = [
                child
                for child, parents in self.hierarchy_cache.items()
                if candidate["iri"] in parents
            ]

        return top_candidates

    def _apply_fallback_matching(
        self, entity: Dict[str, Any], context: str, filtered_entities: Set[str]
    ):
        words = entity["text"].split()
        content_words = [w for w in words if w.lower() not in self.stop_words]

        if len(content_words) == 0:
            return  # No content words to match

        if len(content_words) > 1:
            for i in range(1, len(content_words)):
                left_part = " ".join(content_words[:i])
                right_part = " ".join(content_words[i:])
                left_match = self._find_best_match_with_context(
                    {"text": left_part, "vector": entity["vector"]},
                    context,
                    filtered_entities,
                )
                right_match = self._find_best_match_with_context(
                    {"text": right_part, "vector": entity["vector"]},
                    context,
                    filtered_entities,
                )

                if left_match.iri or right_match.iri:
                    best_match = (
                        left_match
                        if left_match.similarity > right_match.similarity
                        else right_match
                    )
                    entity["match"] = best_match.dict()
                    entity["hierarchy"] = self._get_entity_hierarchy(
                        entity["match"]["iri"]
                    )
                    entity["classification"] = self.classify_hierarchically(entity)
                    return

        # If still no match, try graph-based matching
        graph_matches = self.graph_based_matching(entity)
        if graph_matches:
            best_graph_match = graph_matches[0]
            entity["match"] = {
                "iri": best_graph_match["iri"],
                "label": best_graph_match["label"],
                "similarity": best_graph_match["similarity"],
                "match_type": "graph_based",
            }
            entity["hierarchy"] = self._get_entity_hierarchy(best_graph_match["iri"])
            entity["classification"] = self.classify_hierarchically(entity)
