from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF, RDFS, SKOS, OWL
import requests
import os
import json


class LMSSOntologyParser:
    def __init__(self, ontology_file):
        self.g = Graph()
        self.g.parse(ontology_file, format="xml")
        self.entities = {}
        self.parse_ontology()

    @staticmethod
    def download_ontology(url, save_path):
        """
        Downloads the OWL ontology from the given URL and saves it to the specified path.

        Args:
            url (str): The URL to download the OWL ontology from.
            save_path (str): The path to save the downloaded OWL file.
        """
        # Modify the URL to point to the raw content
        raw_url = url.replace("github.com", "raw.githubusercontent.com").replace(
            "/blob/", "/"
        )

        response = requests.get(raw_url)
        if response.status_code == 200:
            with open(save_path, "wb") as file:
                file.write(response.content)
            print(f"Successfully downloaded ontology and saved to {save_path}")
        else:
            print(f"Failed to download ontology. Status code: {response.status_code}")

    def parse_ontology(self):
        # Iterate over all triples to find classes and attributes
        for s, p, o in self.g:
            if p == RDF.type and (o == RDFS.Class or o == OWL.Class):
                label = self.g.value(s, RDFS.label)
                if label:
                    self.entities[str(label)] = {
                        "iri": str(s),
                        "definition": str(self.g.value(s, SKOS.definition)),
                        "examples": [str(ex) for ex in self.g.objects(s, SKOS.example)],
                        "subClassOf": [
                            str(sub) for sub in self.g.objects(s, RDFS.subClassOf)
                        ],
                        "isDefinedBy": str(self.g.value(s, RDFS.isDefinedBy)),
                    }
                else:
                    print(f"Class without label: {s}")
            elif p in {
                RDFS.subClassOf,
                RDFS.isDefinedBy,
                RDFS.label,
                SKOS.definition,
                SKOS.example,
            }:
                print(f"Found attribute: {p}")

    def get_entities(self):
        return self.entities

    def get_entity_iri(self, entity_name):
        return self.entities.get(entity_name, {}).get("iri")

    def get_entity_info(self, entity_name):
        return self.entities.get(entity_name, {})


# Example usage
if __name__ == "__main__":
    ontology_url = "https://github.com/sali-legal/LMSS/blob/main/LMSS.owl"  # Replace with the actual URL
    save_path = "app/LMSS.owl"

    # Check if the file already exists, if not then download
    if not os.path.exists(save_path):
        LMSSOntologyParser.download_ontology(ontology_url, save_path)
    else:
        print(f"Ontology file already exists at {save_path}")

    parser = LMSSOntologyParser(save_path)
    entities = parser.get_entities()

    # Print entities to console
    print("Entities:", entities)

    # Write entities to a JSON file for inspection
    with open("entities.json", "w") as f:
        json.dump(entities, f, indent=4)
