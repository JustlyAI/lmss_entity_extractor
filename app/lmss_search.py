import json
from typing import List, Dict, Optional, Set
import numpy as np
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import fuzz
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDFS, SKOS, OWL
from pydantic import BaseModel, Field
import time
import re


class Entity(BaseModel):
    rdf_about: str
    rdfs_label: str
    skos_prefLabel: str = ""
    skos_altLabel: List[str] = Field(default_factory=list)
    description: str = ""
    rdfs_seeAlso: List[str] = Field(default_factory=list)
    skos_definition: str = ""
    skos_example: List[str] = Field(default_factory=list)
    subClassOf: List[str] = Field(default_factory=list)


class TopClass(BaseModel):
    iri: str
    label: str
    entities_count: int


class SearchResult(BaseModel):
    iri: str
    label: str
    score: float
    match_type: str
    parent_class: Optional[str] = None
    hierarchy: List[str] = Field(default_factory=list)
    branch: str


class LMSSSearch:
    def __init__(self, index_path: str, graph_path: str, top_classes_path: str):
        self.index = self._load_json(index_path, is_entity=True)
        self.graph = Graph()
        self.graph.parse(graph_path, format="turtle")
        self.top_classes = self._load_json(top_classes_path, is_entity=False)
        self.LMSS = Namespace("http://lmss.sali.org/")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.entity_embeddings = self._load_embeddings()
        self.entity_branches = self._precompute_entity_branches()

    def _load_json(self, path: str, is_entity: bool) -> List[Entity]:
        with open(path, "r") as f:
            data = json.load(f)
            if is_entity:
                return [Entity(**entity) for entity in data]
            else:
                return [TopClass(**top_class) for top_class in data]

    def _load_embeddings(self) -> Dict[str, Dict[str, np.ndarray]]:
        embeddings = {}
        for s, p, o in self.graph.triples((None, self.LMSS.hasEmbedding, None)):
            entity_iri = str(s)
            embedding_field = str(self.graph.value(o, self.LMSS.embeddingField))
            embedding_value = self.graph.value(o, self.LMSS.embeddingValue)
            if embedding_value:
                if entity_iri not in embeddings:
                    embeddings[entity_iri] = {}
                embeddings[entity_iri][embedding_field] = np.array(
                    json.loads(str(embedding_value))
                )
        return embeddings

    def _precompute_entity_branches(self) -> Dict[str, str]:
        entity_branches = {}
        for entity in self.index:
            entity_branches[entity.rdf_about] = self._get_branch(entity.rdf_about)
        return entity_branches

    def search(
        self, query: str, top_k: int = 10, selected_branches: Optional[List[str]] = None
    ) -> List[SearchResult]:
        start_time = time.time()
        query_embedding = self.model.encode(query)

        results = []
        for entity in self.index:
            if (
                selected_branches
                and self.entity_branches[entity.rdf_about] not in selected_branches
            ):
                continue

            score, match_type = self._compute_relevance_score(
                entity, query, query_embedding
            )
            if score > 0:
                parent_class = self._get_parent_class(entity.rdf_about)
                hierarchy = self._get_hierarchy(entity.rdf_about)
                branch = self.entity_branches[entity.rdf_about]
                results.append(
                    SearchResult(
                        iri=entity.rdf_about,
                        label=entity.rdfs_label,
                        score=score,
                        match_type=match_type,
                        parent_class=parent_class,
                        hierarchy=hierarchy,
                        branch=branch,
                    )
                )

        results = sorted(results, key=lambda x: x.score, reverse=True)[:top_k]

        end_time = time.time()
        print(f"Search completed in {end_time - start_time:.2f} seconds")

        return results

    def _compute_relevance_score(
        self,
        entity: Entity,
        query: str,
        query_embedding: np.ndarray,
    ) -> (float, str):
        exact_match_score = self._exact_match_score(entity, query)
        if exact_match_score > 0:
            return exact_match_score, "Exact Match"

        partial_match_score = self._partial_match_score(entity, query)
        if partial_match_score > 0:
            return partial_match_score, "Partial Match"

        fuzzy_match_score = self._fuzzy_match_score(entity, query)
        semantic_score = self._semantic_similarity(query_embedding, entity.rdf_about)
        graph_score = self._graph_based_score(entity, query)

        # Adjust weights here
        weights = [0.5, 0.3, 0.2]  # Fuzzy Match, Semantic, Graph
        combined_score = (
            weights[0] * fuzzy_match_score
            + weights[1] * semantic_score
            + weights[2] * graph_score
        )

        # Apply hierarchical boost
        hierarchical_boost = 1.0 + (0.1 * self._is_root_class(entity.rdf_about))

        return combined_score * hierarchical_boost, "Semantic Match"

    def _exact_match_score(self, entity: Entity, query: str) -> float:
        query_lower = query.lower()
        if query_lower == entity.rdfs_label.lower():
            return 1.0
        if query_lower == entity.skos_prefLabel.lower():
            return 0.95
        if any(query_lower == alt.lower() for alt in entity.skos_altLabel):
            return 0.9
        return 0.0

    def _partial_match_score(self, entity: Entity, query: str) -> float:
        query_lower = query.lower()
        query_words = set(query_lower.split())

        label_words = set(entity.rdfs_label.lower().split())
        if query_words.issubset(label_words):
            return 0.85

        pref_label_words = set(entity.skos_prefLabel.lower().split())
        if query_words.issubset(pref_label_words):
            return 0.8

        for alt_label in entity.skos_altLabel:
            alt_label_words = set(alt_label.lower().split())
            if query_words.issubset(alt_label_words):
                return 0.75

        return 0.0

    def _fuzzy_match_score(self, entity: Entity, query: str) -> float:
        query_lower = query.lower()

        # Check rdfs:label
        label_score = fuzz.partial_ratio(query_lower, entity.rdfs_label.lower()) / 100

        # Check skos:prefLabel
        pref_label_score = (
            fuzz.partial_ratio(query_lower, entity.skos_prefLabel.lower()) / 100
            if entity.skos_prefLabel
            else 0
        )

        # Check skos:altLabel
        alt_label_scores = [
            fuzz.partial_ratio(query_lower, alt.lower()) / 100
            for alt in entity.skos_altLabel
        ]
        alt_label_score = max(alt_label_scores) if alt_label_scores else 0

        # Prioritize label fields
        weights = [0.5, 0.3, 0.2]  # label, prefLabel, altLabel
        return (
            weights[0] * label_score
            + weights[1] * pref_label_score
            + weights[2] * alt_label_score
        )

    def _semantic_similarity(
        self, query_embedding: np.ndarray, entity_iri: str
    ) -> float:
        if entity_iri in self.entity_embeddings:
            entity_emb = self.entity_embeddings[entity_iri]
            label_sim = self._cosine_similarity(
                query_embedding,
                entity_emb.get("rdfs_label", np.zeros_like(query_embedding)),
            )
            pref_label_sim = self._cosine_similarity(
                query_embedding,
                entity_emb.get("skos_prefLabel", np.zeros_like(query_embedding)),
            )
            alt_label_sim = self._cosine_similarity(
                query_embedding,
                entity_emb.get("skos_altLabel", np.zeros_like(query_embedding)),
            )

            # Prioritize label fields
            weights = [0.5, 0.3, 0.2]  # label, prefLabel, altLabel
            return (
                weights[0] * label_sim
                + weights[1] * pref_label_sim
                + weights[2] * alt_label_sim
            )
        return 0.0

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        if v1.size == 0 or v2.size == 0:
            return 0.0
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm == 0:
            return 0.0
        return np.dot(v1, v2) / norm

    def _graph_based_score(self, entity: Entity, query: str) -> float:
        score = 0.0
        entity_node = URIRef(entity.rdf_about)

        # Check direct subclass relationship
        for parent in self.graph.objects(entity_node, RDFS.subClassOf):
            if query.lower() in str(parent).lower():
                score += 0.5
                break

        # Check for siblings with matching query
        for sibling in self.graph.subjects(RDFS.subClassOf, entity_node):
            if query.lower() in str(sibling).lower():
                score += 0.3
                break

        return score

    def _is_root_class(self, entity_iri: str) -> bool:
        return any(tc.iri == entity_iri for tc in self.top_classes)

    def _get_parent_class(self, entity_iri: str) -> Optional[str]:
        entity_node = URIRef(entity_iri)
        for parent in self.graph.objects(entity_node, RDFS.subClassOf):
            return str(parent)
        return None

    def _get_hierarchy(self, entity_iri: str) -> List[str]:
        hierarchy = []
        current = URIRef(entity_iri)
        while current:
            hierarchy.append(str(current))
            parents = list(self.graph.objects(current, RDFS.subClassOf))
            current = parents[0] if parents else None
        return list(reversed(hierarchy))

    def _get_branch(self, entity_iri: str) -> str:
        for parent in self.graph.transitive_objects(
            URIRef(entity_iri), RDFS.subClassOf
        ):
            if str(parent) in [tc.iri for tc in self.top_classes]:
                return next(
                    tc.label for tc in self.top_classes if tc.iri == str(parent)
                )
        return "Unknown"

    def get_top_classes(self) -> List[TopClass]:
        return self.top_classes
