import json
from typing import List, Dict, Optional, Set
import numpy as np
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import fuzz
from rdflib import Graph, URIRef, Namespace
from rdflib.namespace import RDFS
from pydantic import BaseModel, Field


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
    embedding: Optional[List[float]] = None


class TopClass(BaseModel):
    iri: str
    label: str
    entities_count: int


class LMSSSearch:
    def __init__(self, index_path: str, graph_path: str, top_classes_path: str):
        self.index = self._load_json(index_path, is_entity=True)
        self.graph = Graph()
        self.graph.parse(graph_path, format="turtle")
        self.top_classes = self._load_json(top_classes_path, is_entity=False)
        self.LMSS = Namespace("http://lmss.sali.org/")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def _load_json(self, path: str, is_entity: bool) -> List[Entity]:
        with open(path, "r") as f:
            data = json.load(f)
            if is_entity:
                return [Entity(**entity) for entity in data]
            else:
                return [TopClass(**top_class) for top_class in data]

    def _filter_entities(self, selected_branches: List[str]) -> Set[str]:
        filtered_entities = set()
        for branch in selected_branches:
            filtered_entities.add(branch)
            filtered_entities.update(self._get_subclasses(URIRef(branch)))
        return filtered_entities

    def _get_subclasses(self, class_iri: URIRef) -> Set[str]:
        subclasses = set()
        for s, p, o in self.graph.triples((None, RDFS.subClassOf, class_iri)):
            subclasses.add(str(s))
            subclasses.update(self._get_subclasses(s))
        return subclasses

    def search(
        self, query: str, selected_branches: Optional[List[str]] = None
    ) -> List[Dict]:
        results = []
        query_embedding = np.array(self._get_embedding(query))

        if selected_branches:
            filtered_entities = self._filter_entities(selected_branches)
        else:
            filtered_entities = set(entity.rdf_about for entity in self.index)

        for entity in self.index:
            if entity.rdf_about not in filtered_entities:
                continue

            label = entity.rdfs_label
            score = self._compute_score(
                query, label, query_embedding, np.array(entity.embedding or [])
            )

            if score > 0:
                results.append(
                    {"iri": entity.rdf_about, "label": label, "score": score}
                )

        return sorted(results, key=lambda x: x["score"], reverse=True)[:10]

    def _compute_score(
        self,
        query: str,
        label: str,
        query_embedding: np.ndarray,
        label_embedding: np.ndarray,
    ) -> float:
        regex_score = fuzz.token_set_ratio(query.lower(), label.lower()) / 100
        fuzzy_score = fuzz.partial_ratio(query.lower(), label.lower()) / 100
        vector_score = self._cosine_similarity(query_embedding, label_embedding)

        weights = [0.3, 0.3, 0.4]  # Regex, Fuzzy, Vector
        return (
            weights[0] * regex_score
            + weights[1] * fuzzy_score
            + weights[2] * vector_score
        )

    @staticmethod
    def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        if v1.size == 0 or v2.size == 0:
            return 0
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def _get_embedding(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()

    def get_top_classes(self) -> List[TopClass]:
        return self.top_classes
