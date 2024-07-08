import re
import asyncio
import os
import json
import hashlib
import requests
from pydantic import BaseModel, Field
from rdflib import Graph, URIRef, Literal, BNode, Namespace
from rdflib.namespace import RDF, RDFS, OWL, SKOS
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import fuzz
from typing import Dict, List, Tuple, Any
import logging
from tqdm.asyncio import tqdm

# from tqdm import tqdm


# Define the InconsistencyError class
class InconsistencyError(Exception):
    pass


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


class LMSSOntologyParser:
    def __init__(self, ontology_file: str):
        self.ontology_file = ontology_file
        self.graph = Graph()
        self.LMSS = Namespace("http://lmss.sali.org/")
        self.graph.bind("lmss", self.LMSS)
        self.entities: Dict[str, Entity] = {}
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.excluded_prefixes = ["ZZZ - SANDBOX: UNDER CONSTRUCTION"]
        self.top_classes = []
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        self.logger.info("Initialized parser")

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
                logging.getLogger(__name__).info(
                    "We have updated the LMSS ontology with the latest version"
                )
            else:
                logging.getLogger(__name__).info(
                    f"Successfully downloaded ontology and saved to {save_path}"
                )
        else:
            logging.getLogger(__name__).error(
                f"Failed to download ontology. Status code: {response.status_code}"
            )

    @staticmethod
    def calculate_file_hash(file_path: str) -> str:
        """Calculates the SHA-256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def should_exclude(self, label: str) -> bool:
        """Check if the entity should be excluded based on its label."""
        return any(prefix in label for prefix in self.excluded_prefixes)

    async def parse_ontology(self):
        """Parses the ontology to extract classes, hierarchical relationships, attributes, and inter-class relationships."""
        self.logger.info("Parsing ontology...")
        self.graph.parse(self.ontology_file, format="xml")

        for s, p, o in self.graph:
            if p == RDF.type and o == OWL.Class:
                iri = str(s)
                label = self.get_literal(s, RDFS.label)
                if not self.should_exclude(label):
                    self.entities[iri] = Entity(
                        rdf_about=iri,
                        rdfs_label=label,
                        description=self.get_literal(
                            s, URIRef("http://purl.org/dc/elements/1.1/description")
                        ),
                        rdfs_seeAlso=self.get_literals(s, RDFS.seeAlso),
                        skos_altLabel=self.get_literals(s, SKOS.altLabel),
                        skos_definition=self.get_literal(s, SKOS.definition),
                        skos_example=self.get_literals(s, SKOS.example),
                        skos_prefLabel=self.get_literal(s, SKOS.prefLabel),
                        subClassOf=self.get_literals(
                            s, RDFS.subClassOf
                        ),  # Capture all subClassOf relationships
                    )
                    self.logger.info(f"Parsed entity: {iri} with label: {label}")

        self.identify_top_classes()
        self.logger.info(f"Parsed {len(self.entities)} entities from the ontology.")

        excluded_count = sum(
            1
            for s, p, o in self.graph.triples((None, RDF.type, OWL.Class))
            if self.should_exclude(self.get_literal(s, RDFS.label))
        )
        self.logger.info(f"Excluded {excluded_count} entities based on prefix rules.")

    def get_literal(self, s: URIRef, p: URIRef) -> str:
        value = self.graph.value(s, p)
        return str(value) if value is not None else ""

    def get_literals(self, s: URIRef, p: URIRef) -> List[str]:
        return [str(o) for o in self.graph.objects(s, p)]

    def identify_top_classes(self):
        """Identifies direct children of OWL:Thing, excluding specified prefixes."""
        self.logger.info("Identifying top classes...")

        owl_thing = OWL.Thing
        direct_subclasses = set(self.graph.subjects(RDFS.subClassOf, owl_thing))

        self.logger.debug(f"Direct subclasses of OWL:Thing: {direct_subclasses}")

        self.top_classes = []
        for cls in direct_subclasses:
            label = self.get_literal(cls, RDFS.label)
            self.logger.debug(f"Class: {cls}, Label: {label}")
            if not self.should_exclude(label):
                self.top_classes.append(cls)
                self.logger.info(f"Top class identified: {cls} with label: {label}")
            else:
                self.logger.info(f"Excluded top class: {cls} with label: {label}")

        self.logger.info(
            f"Identified {len(self.top_classes)} high-level parent classes."
        )
        self.logger.debug(f"Identified top classes: {self.top_classes}")

    async def generate_embeddings(self):
        """Generates embeddings for all entities."""
        self.logger.info("Generating embeddings...")

        total_entities = len(self.entities)
        progress_bar = tqdm(
            total=total_entities, desc="Generating embeddings", unit="entity"
        )

        async def process_entity(iri, entity):
            await self.generate_entity_embedding(iri, entity)
            progress_bar.update(1)

        embedding_tasks = [
            process_entity(iri, entity)
            for iri, entity in self.entities.items()
            if not self.should_exclude(entity.rdfs_label)
        ]

        await asyncio.gather(*embedding_tasks)
        progress_bar.close()
        self.logger.info("Finished generating embeddings.")

    async def generate_embeddings_for_class(self, top_class: URIRef):
        """Generates embeddings for all entities under a specific top-class."""
        entities_under_class = self.get_entities_under_class(top_class)
        embedding_tasks = [
            self.generate_entity_embedding(iri, entity)
            for iri, entity in entities_under_class.items()
        ]
        await asyncio.gather(*embedding_tasks)

    def get_entities_under_class(self, class_uri: URIRef) -> Dict[str, Entity]:
        """Returns all entities that are subclasses of the given class."""
        entities = {}
        for s, p, o in self.graph.triples((None, RDFS.subClassOf, class_uri)):
            if str(s) in self.entities:
                entities[str(s)] = self.entities[str(s)]
                entities.update(self.get_entities_under_class(s))
        return entities

    async def generate_entity_embedding(self, iri: str, entity: Entity):
        """Generates embeddings for a single entity and adds them to the graph."""
        fields_to_embed = [
            "rdfs_label",
            "description",
            "skos_definition",
            "skos_prefLabel",
        ]
        list_fields_to_embed = ["rdfs_seeAlso", "skos_altLabel", "skos_example"]

        embeddings = {}
        entity_has_data = False

        for field in fields_to_embed:
            text = getattr(entity, field, "").strip()
            if text:
                embeddings[field] = self.model.encode(text).tolist()  # Convert to list
                entity_has_data = True

        for field in list_fields_to_embed:
            text = " ".join(getattr(entity, field, [])).strip()
            if text:
                embeddings[field] = self.model.encode(text).tolist()  # Convert to list
                entity_has_data = True

        if entity_has_data:
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

            self.logger.info(f"Added embeddings for entity: {iri}")
        else:
            self.logger.warning(f"No data to embed for entity: {iri}")

    def save_to_json(self, file_path: str):
        """Saves the parsed ontology data to a JSON file."""
        with open(file_path, "w") as f:
            json.dump(
                [entity.model_dump() for entity in self.entities.values()], f, indent=2
            )
        self.logger.info(f"Saved ontology data to {file_path}")

    def save_graph(self, file_path: str):
        """Saves the RDF graph to a file."""
        self.graph.serialize(destination=file_path, format="turtle")
        self.logger.info(f"Saved RDF graph to {file_path}")

    async def _process_ontology_internal(self, index_path: str, graph_path: str):
        """Internal method to process the ontology, generate embeddings, and save results."""
        self.logger.info("Starting to parse ontology...")
        await self.parse_ontology()
        self.logger.info("Ontology parsed successfully.")

        self.logger.info("Saving parsed ontology data to JSON...")
        self.save_to_json(index_path)
        self.logger.info("Ontology data saved to JSON.")

        self.logger.info("Starting to generate embeddings...")
        await self.generate_embeddings()
        self.logger.info("Embeddings generated successfully.")

        self.logger.info("Saving RDF graph...")
        self.save_graph(graph_path)
        self.logger.info("RDF graph saved successfully.")

    async def process_ontology(
        self, index_path: str, graph_path: str, max_attempts: int = 3
    ):
        stages = [
            "Parsing ontology",
            "Generating embeddings",
            "Saving data",
            "Ensuring consistency",
        ]

        for attempt in range(max_attempts):
            try:
                self.logger.info(f"Processing ontology, attempt {attempt + 1}...")

                with tqdm(
                    total=len(stages), desc="Processing stages", unit="stage"
                ) as pbar:
                    self.logger.info("Starting to parse ontology...")
                    await self.parse_ontology()
                    pbar.update(1)
                    self.logger.info("Ontology parsed successfully.")

                    self.logger.info("Starting to generate embeddings...")
                    await self.generate_embeddings()
                    pbar.update(1)
                    self.logger.info("Embeddings generated successfully.")

                    self.logger.info("Saving parsed ontology data...")
                    self.save_to_json(index_path)
                    self.save_graph(graph_path)
                    pbar.update(1)
                    self.logger.info("Data saved successfully.")

                    self.logger.info("Ensuring consistency...")
                    inconsistencies = self.ensure_consistency()
                    pbar.update(1)

                    if inconsistencies:
                        self.logger.warning(f"Inconsistencies found: {inconsistencies}")
                        self.logger.info(
                            "Reprocessing ontology after resolving inconsistencies..."
                        )
                        continue

                    self.logger.info("Ontology processed successfully.")
                    return
            except Exception as e:
                self.logger.error(f"Error during processing: {str(e)}", exc_info=True)
                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt + 1 < max_attempts:
                    self.logger.info("Retrying...")
                    # Clear existing data before retrying
                    self.entities.clear()
                    self.graph = Graph()
                else:
                    self.logger.error("Max attempts reached. Aborting.")
                    raise

    def ensure_consistency(self) -> List[str]:
        inconsistencies = []

        # Ensure all entities are in the graph
        for iri, entity in self.entities.items():
            uri = URIRef(iri)
            if (uri, RDF.type, OWL.Class) not in self.graph:
                self.graph.add((uri, RDF.type, OWL.Class))
                inconsistencies.append(f"Added missing class to graph: {iri}")

            label = entity.rdfs_label
            if label and (uri, RDFS.label, None) not in self.graph:
                self.graph.add((uri, RDFS.label, Literal(label)))
                inconsistencies.append(f"Added missing label to graph: {iri}")

        if inconsistencies:
            self.logger.warning(
                f"Inconsistencies found and resolved:\n{inconsistencies}"
            )

        self.logger.info(f"Consistency ensured. Total entities: {len(self.entities)}")
        return inconsistencies

    def get_entity_hierarchy(self, entity_iri: str) -> List[str]:
        """
        Retrieves the full hierarchy (parent classes) for a given entity.

        Args:
        entity_iri (str): The IRI of the entity.

        Returns:
        List[str]: A list of IRIs representing the hierarchy, from the entity to the root.
        """
        hierarchy = []
        current_entity = URIRef(entity_iri)

        while current_entity != OWL.Thing:
            if isinstance(current_entity, URIRef):
                hierarchy.append(str(current_entity))
            parent = self.graph.value(current_entity, RDFS.subClassOf)
            if parent is None or not isinstance(parent, (URIRef, BNode)):
                break
            current_entity = parent

        if current_entity == OWL.Thing:
            hierarchy.append(str(OWL.Thing))

        return hierarchy

    def load_entities(self, index_path: str, max_attempts: int = 3):
        for attempt in range(max_attempts):
            try:
                self._load_entities_internal(index_path)
                self.ensure_consistency()
                self.logger.info("Entities loaded successfully.")
                return
            except InconsistencyError as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt + 1 < max_attempts:
                    self.logger.info("Retrying...")
                    # Clear existing data before retrying
                    self.entities.clear()
                    self.graph = Graph()
                else:
                    self.logger.error("Max attempts reached. Aborting.")
                    raise

    def _load_entities_internal(self, index_path: str):
        with open(index_path, "r") as f:
            all_entities = json.load(f)

        self.entities = {
            entity["rdf_about"]: Entity(**entity)
            for entity in all_entities
            if not self.should_exclude(entity.get("rdfs_label", ""))
        }
        self.logger.info(
            f"Loaded {len(self.entities)} entities from the index (excluding sandbox)"
        )
