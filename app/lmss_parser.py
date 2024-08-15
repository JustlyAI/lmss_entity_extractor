import os
import json
import hashlib
import requests
from rdflib import Graph, URIRef, Literal, Namespace, BNode
from rdflib.namespace import RDF, RDFS, OWL, SKOS, DC
from sentence_transformers import SentenceTransformer
from typing import Dict, List
import logging


class OntologyParser:
    def __init__(self, ontology_file: str, model: SentenceTransformer = None):
        self.ontology_file = ontology_file
        self.graph = Graph()
        self.LMSS = Namespace("http://lmss.sali.org/")
        self.graph.bind("lmss", self.LMSS)
        self.entities = {}
        self.top_classes = []
        self.model = model or SentenceTransformer("all-MiniLM-L6-v2")
        self.excluded_prefixes = ["ZZZ - SANDBOX: UNDER CONSTRUCTION"]
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def download_ontology(url: str, save_path: str) -> bool:
        raw_url = url.replace("github.com", "raw.githubusercontent.com").replace(
            "/blob/", "/"
        )
        response = requests.get(raw_url)
        if response.status_code == 200:
            with open(save_path, "wb") as file:
                file.write(response.content)
            return True
        return False

    @staticmethod
    def calculate_file_hash(file_path: str) -> str:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def parse_ontology(self):
        self.logger.info("Parsing ontology...")
        self.graph.parse(self.ontology_file, format="xml")
        for s, p, o in self.graph:
            if p == RDF.type and o == OWL.Class:
                iri = str(s)
                label = self.get_literal(s, RDFS.label)
                if not any(prefix in label for prefix in self.excluded_prefixes):
                    self.entities[iri] = {
                        "rdf_about": iri,
                        "rdfs_label": label,
                        "description": self.get_literal(s, DC.description),
                        "rdfs_seeAlso": self.get_literals(s, RDFS.seeAlso),
                        "skos_altLabel": self.get_literals(s, SKOS.altLabel),
                        "skos_definition": self.get_literal(s, SKOS.definition),
                        "skos_example": self.get_literals(s, SKOS.example),
                        "skos_prefLabel": self.get_literal(s, SKOS.prefLabel),
                        "subClassOf": self.get_literals(s, RDFS.subClassOf),
                    }
        self.logger.info(f"Parsed {len(self.entities)} entities.")

    def get_literal(self, s: URIRef, p: URIRef) -> str:
        return str(self.graph.value(s, p) or "")

    def get_literals(self, s: URIRef, p: URIRef) -> List[str]:
        return [str(o) for o in self.graph.objects(s, p)]

    def save_index(self, file_path: str):
        self.logger.info(f"Saving index to {file_path}...")
        with open(file_path, "w") as f:
            json.dump(list(self.entities.values()), f, indent=2)
        self.logger.info("Index saved.")

    def identify_top_classes(self):
        self.logger.info("Identifying top classes...")
        owl_thing = OWL.Thing
        self.top_classes = [
            cls
            for cls in self.graph.subjects(RDFS.subClassOf, owl_thing)
            if not any(
                prefix in self.get_literal(cls, RDFS.label)
                for prefix in self.excluded_prefixes
            )
        ]
        self.logger.info(f"Identified {len(self.top_classes)} top classes.")

    def save_top_classes(self, file_path: str):
        self.logger.info(f"Saving top classes to {file_path}...")
        top_classes_data = [
            {
                "iri": str(cls),
                "label": self.get_literal(cls, RDFS.label),
                "entities_count": len(self.get_entities_under_class(cls)),
            }
            for cls in self.top_classes
        ]
        # Sort top_classes_data by label in alphabetical order
        top_classes_data = sorted(top_classes_data, key=lambda x: x["label"])
        with open(file_path, "w") as f:
            json.dump(top_classes_data, f, indent=2)
        self.logger.info("Top classes saved.")

    def get_entities_under_class(self, class_uri: URIRef) -> Dict[str, Dict]:
        entities = {}
        for s, p, o in self.graph.triples((None, RDFS.subClassOf, class_uri)):
            if str(s) in self.entities:
                entities[str(s)] = self.entities[str(s)]
                entities.update(self.get_entities_under_class(s))
        return entities

    def generate_embeddings(self):
        self.logger.info("Generating embeddings and updating graph...")
        print(f"Number of entities: {len(self.entities)}")
        for iri, entity in self.entities.items():
            entity_uri = URIRef(iri)
            for field in [
                "rdfs_label",
                "skos_definition",
                "skos_prefLabel",
                "skos_altLabel",
            ]:
                text = entity.get(field, "")
                if isinstance(text, list):
                    text = " ".join(text)
                if text:
                    print(f"Generating embedding for {field}: {text}")
                    embedding = self.model.encode(text)

                    # Convert to list if it's a numpy array
                    if hasattr(embedding, "tolist"):
                        embedding = embedding.tolist()

                    print(
                        f"Generated embedding for {field}: {embedding[:5]}..."
                    )  # Print first 5 elements

                    # Add embedding to the graph
                    embedding_node = BNode()
                    self.graph.add((entity_uri, self.LMSS.hasEmbedding, embedding_node))
                    self.graph.add(
                        (embedding_node, self.LMSS.embeddingField, Literal(field))
                    )
                    self.graph.add(
                        (
                            embedding_node,
                            self.LMSS.embeddingValue,
                            Literal(json.dumps(embedding)),
                        )
                    )

        self.logger.info("Embeddings generated and added to the graph.")
        print(f"Total triples in graph after adding embeddings: {len(self.graph)}")

    def get_statistics(self):
        return {
            "branches": len(self.top_classes),
            "classes": len(self.entities),
            "attributes_with_embeddings": sum(
                1 for _ in self.graph.triples((None, self.LMSS.hasEmbedding, None))
            ),
        }

    def save_graph(self, file_path: str):
        self.logger.info(f"Saving graph to {file_path}...")
        self.graph.serialize(destination=file_path, format="turtle")
        self.logger.info("Graph saved.")

    def process_ontology(
        self, index_path: str, graph_path: str, top_classes_path: str, stats_path: str
    ):
        # Step 1: Parse ontology and create lmss_index
        self.parse_ontology()
        self.save_index(index_path)

        # Step 2: Identify and save top_classes
        self.identify_top_classes()
        self.save_top_classes(top_classes_path)

        # Step 3: Generate embeddings and update graph
        self.generate_embeddings()

        # Step 4: Save the updated graph
        self.save_graph(graph_path)

        # Step 5: Generate and save statistics
        stats = self.get_statistics()
        self.logger.info(f"Saving statistics to {stats_path}...")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        self.logger.info("Statistics saved.")

        return stats
