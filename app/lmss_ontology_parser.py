import asyncio
import os
import json
import hashlib
import requests
from rdflib import Graph, URIRef, Literal, BNode, Namespace
from rdflib.namespace import RDF, RDFS, OWL, SKOS
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Tuple, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedLMSSOntologyParser:
    def __init__(self, ontology_file: str):
        self.ontology_file = ontology_file
        self.graph = Graph()
        self.entities: Dict[str, Dict[str, Any]] = {}
        self.top_classes: List[URIRef] = []
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.LMSS = Namespace("http://lmss.sali.org/")
        self.graph.bind("lmss", self.LMSS)

    @staticmethod
    def download_ontology(url: str, save_path: str):
        """Downloads the OWL ontology from the given URL and saves it to the specified path."""
        raw_url = url.replace("github.com", "raw.githubusercontent.com").replace(
            "/blob/", "/"
        )
        response = requests.get(raw_url)
        if response.status_code == 200:
            with open(save_path, "wb") as file:
                file.write(response.content)
            if os.path.exists(save_path):
                logger.info("We have updated the LMSS ontology with the latest version")
            else:
                logger.info(
                    f"Successfully downloaded ontology and saved to {save_path}"
                )
        else:
            logger.error(f"Failed to download ontology. Status code: {response.status}")

    @staticmethod
    def calculate_file_hash(file_path: str) -> str:
        """Calculates the SHA-256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    async def parse_ontology(self):
        """Parses the ontology to extract classes, hierarchical relationships, attributes, and inter-class relationships."""
        logger.info("Parsing ontology...")
        self.graph.parse(self.ontology_file, format="xml")

        for s, p, o in self.graph:
            if p == RDF.type and o == OWL.Class:
                iri = str(s)
                self.entities[iri] = {
                    "rdf:about": iri,
                    "rdfs:label": self.get_literal(s, RDFS.label),
                    "description": self.get_literal(
                        s, URIRef("http://purl.org/dc/elements/1.1/description")
                    ),
                    "rdfs:seeAlso": self.get_literals(s, RDFS.seeAlso),
                    "skos:altLabel": self.get_literals(s, SKOS.altLabel),
                    "skos:definition": self.get_literal(s, SKOS.definition),
                    "skos:example": self.get_literals(s, SKOS.example),
                    "skos:prefLabel": self.get_literal(s, SKOS.prefLabel),
                }

        self.identify_top_classes()
        logger.info(f"Parsed {len(self.entities)} entities from the ontology.")

    def get_literal(self, s: URIRef, p: URIRef) -> str:
        return str(self.graph.value(s, p, default=""))

    def get_literals(self, s: URIRef, p: URIRef) -> List[str]:
        return [str(o) for o in self.graph.objects(s, p)]

    def identify_top_classes(self):
        """Identifies direct children of OWL:Thing, excluding 'ZZZ - SANDBOX: UNDER CONSTRUCTION'."""
        self.top_classes = [
            s
            for s, p, o in self.graph.triples((None, RDFS.subClassOf, OWL.Thing))
            if "ZZZ - SANDBOX: UNDER CONSTRUCTION" not in str(s)
        ]
        logger.info(f"Identified {len(self.top_classes)} high-level parent classes.")

    async def generate_embeddings(self):
        """Generates embeddings for all entities."""
        logger.info("Generating embeddings...")
        embedding_tasks = [
            self.generate_entity_embedding(iri, entity)
            for iri, entity in self.entities.items()
        ]
        await asyncio.gather(*embedding_tasks)
        logger.info("Finished generating embeddings.")

    async def generate_embeddings_for_class(self, top_class: URIRef):
        """Generates embeddings for all entities under a specific top-class."""
        entities_under_class = self.get_entities_under_class(top_class)
        embedding_tasks = [
            self.generate_entity_embedding(iri, entity)
            for iri, entity in entities_under_class.items()
        ]
        await asyncio.gather(*embedding_tasks)

    def get_entities_under_class(self, class_uri: URIRef) -> Dict[str, Dict[str, Any]]:
        """Returns all entities that are subclasses of the given class."""
        entities = {}
        for s, p, o in self.graph.triples((None, RDFS.subClassOf, class_uri)):
            if str(s) in self.entities:
                entities[str(s)] = self.entities[str(s)]
            entities.update(self.get_entities_under_class(s))
        return entities

    async def generate_entity_embedding(self, iri: str, entity: Dict[str, Any]):
        """Generates embeddings for a single entity and adds them to the graph."""
        fields_to_embed = [
            "rdfs:label",
            "description",
            "skos:definition",
            "skos:prefLabel",
        ]
        list_fields_to_embed = ["rdfs:seeAlso", "skos:altLabel", "skos:example"]

        embeddings = {}
        entity_has_data = False

        for field in fields_to_embed:
            text = entity.get(field, "").strip()
            if text:
                embeddings[field] = self.model.encode(text).tolist()
                entity_has_data = True

        for field in list_fields_to_embed:
            text = " ".join(entity.get(field, [])).strip()
            if text:
                embeddings[field] = self.model.encode(text).tolist()
                entity_has_data = True

        if entity_has_data:
            # Store the embeddings in the graph
            entity_uri = URIRef(iri)
            for field, embedding in embeddings.items():
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

            logger.info(f"Added embeddings for entity: {iri}")
        else:
            logger.warning(f"No data to embed for entity: {iri}")

    def save_to_json(self, file_path: str):
        """Saves the parsed ontology data to a JSON file."""
        with open(file_path, "w") as f:
            json.dump(self.entities, f, indent=2)
        logger.info(f"Saved ontology data to {file_path}")

    def save_graph(self, file_path: str):
        """Saves the RDF graph to a file."""
        self.graph.serialize(destination=file_path, format="turtle")
        logger.info(f"Saved RDF graph to {file_path}")

    async def process_ontology(self, index_path: str, graph_path: str):
        """Processes the ontology, creating the index and generating embeddings."""
        await self.parse_ontology()
        self.save_to_json(index_path)
        await self.generate_embeddings()
        self.save_graph(graph_path)

    def search(
        self, query: str, field: str = "rdfs:label"
    ) -> List[Tuple[str, str, float]]:
        """Searches the index for matching entities using semantic similarity."""
        query_embedding = self.model.encode(query)
        results = []
        for iri, entity in self.entities.items():
            entity_uri = URIRef(iri)
            for embedding_node in self.graph.objects(
                entity_uri, self.LMSS.hasEmbedding
            ):
                embedding_field = self.graph.value(
                    embedding_node, self.LMSS.embeddingField
                )
                if str(embedding_field) == field:
                    embedding_value = self.graph.value(
                        embedding_node, self.LMSS.embeddingValue
                    )
                    if embedding_value:
                        entity_embedding = json.loads(str(embedding_value))
                        similarity = self.cosine_similarity(
                            query_embedding, entity_embedding
                        )
                        if similarity > 0.5:  # Adjust threshold as needed
                            results.append((entity["rdfs:label"], iri, similarity))
                            break  # We found the matching field, no need to continue searching
        return sorted(results, key=lambda x: x[2], reverse=True)

    @staticmethod
    def cosine_similarity(v1: List[float], v2: List[float]) -> float:
        """Computes cosine similarity between two vectors."""
        dot_product = sum(x * y for x, y in zip(v1, v2))
        magnitude1 = sum(x * x for x in v1) ** 0.5
        magnitude2 = sum(y * y for y in v2) ** 0.5
        return dot_product / (magnitude1 * magnitude2)


# Example usage
# async def main():
#     ontology_url = "https://github.com/sali-legal/LMSS/blob/main/LMSS.owl"
#     ontology_path = "LMSS.owl"
#     index_path = "lmss_index.json"
#     graph_path = "lmss_graph.ttl"

#     if not os.path.exists(ontology_path):
#         await EnhancedLMSSOntologyParser.download_ontology(ontology_url, ontology_path)

#     parser = EnhancedLMSSOntologyParser(ontology_path)
#     await parser.process_ontology(index_path, graph_path)

#     # Example search
#     search_results = parser.search("intellectual property")
#     print("Search results for 'intellectual property':")
#     for label, iri, similarity in search_results[:5]:
#         print(f"- {label} (IRI: {iri}, Similarity: {similarity:.2f})")


# if __name__ == "__main__":
#     asyncio.run(main())
