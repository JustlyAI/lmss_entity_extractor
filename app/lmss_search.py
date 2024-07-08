import json
from typing import List, Dict, Optional, Set
from fuzzywuzzy import fuzz
import numpy as np
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDFS


class LMSSSearch:
    def __init__(self, index_path: str, graph_path: str, top_classes_path: str):
        self.index = self._load_json(index_path)
        self.graph = Graph()
        self.graph.parse(graph_path, format="turtle")
        self.top_classes = self._load_json(top_classes_path)
        self.LMSS = Namespace("http://lmss.sali.org/")

    def _load_json(self, path: str) -> Dict:
        with open(path, "r") as f:
            return json.load(f)

    def _filter_entities(self, selected_branches: List[str]) -> Set[str]:
        """
        Filter entities based on selected branches.
        """
        filtered_entities = set()
        for branch in selected_branches:
            filtered_entities.add(branch)
            filtered_entities.update(self._get_subclasses(URIRef(branch)))
        return filtered_entities

    def _get_subclasses(self, class_iri: URIRef) -> Set[str]:
        """
        Recursively get all subclasses of a given class.
        """
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
            filtered_entities = set(entity["rdf_about"] for entity in self.index)

        for entity in self.index:
            if entity["rdf_about"] not in filtered_entities:
                continue

            label = entity["rdfs_label"]
            score = self._compute_score(
                query, label, query_embedding, np.array(entity.get("embedding", []))
            )

            if score > 0:
                results.append(
                    {"iri": entity["rdf_about"], "label": label, "score": score}
                )

        return sorted(results, key=lambda x: x["score"], reverse=True)[:10]

    def _is_in_selected_branches(self, iri: str, selected_branches: List[str]) -> bool:
        for branch in selected_branches:
            if iri.startswith(branch):
                return True
        return False

    def _compute_score(
        self,
        query: str,
        label: str,
        query_embedding: np.ndarray,
        label_embedding: np.ndarray,
    ) -> float:
        # Regex match (using fuzzywuzzy's token_set_ratio for flexibility)
        regex_score = fuzz.token_set_ratio(query.lower(), label.lower()) / 100

        # Fuzzy match
        fuzzy_score = fuzz.partial_ratio(query.lower(), label.lower()) / 100

        # Vector similarity
        vector_score = (
            self._cosine_similarity(query_embedding, label_embedding)
            if len(label_embedding) > 0
            else 0
        )

        # Combine scores (you can adjust the weights)
        return max(regex_score, fuzzy_score, vector_score)

    @staticmethod
    def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        if v1.size == 0 or v2.size == 0:
            return 0
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def _get_embedding(self, text: str) -> List[float]:
        # This method would typically use a pre-trained model to get embeddings
        # For now, we'll return a random vector as a placeholder
        return np.random.rand(768).tolist()  # 768 is a common embedding size

    def get_top_classes(self) -> List[Dict]:
        return self.top_classes
