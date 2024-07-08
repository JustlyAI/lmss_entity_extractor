import requests
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF, RDFS, SKOS


def download_ontology(url, save_path):
    """
    Downloads the OWL ontology from the given URL and saves it to the specified path.

    Args:
        url (str): The URL to download the OWL ontology from.
        save_path (str): The path to save the downloaded OWL file.
    """
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, "wb") as file:
            file.write(response.content)
        print(f"Successfully downloaded ontology and saved to {save_path}")
    else:
        print(f"Failed to download ontology. Status code: {response.status_code}")


def extract_ontology_attributes(ontology_file):
    """
    Extracts the necessary attributes from the downloaded ontology file.

    Args:
        ontology_file (str): The path to the downloaded OWL file.

    Returns:
        dict: A dictionary containing the extracted entities with their attributes.
    """
    g = Graph()
    g.parse(ontology_file, format="xml")
    entities = {}

    for s, p, o in g.triples((None, RDF.type, RDFS.Class)):
        label = g.value(s, RDFS.label)
        if label:
            entities[str(label)] = {
                "iri": str(s),
                "definition": str(g.value(s, SKOS.definition)),
                "examples": [str(ex) for ex in g.objects(s, SKOS.example)],
            }

    return entities


if __name__ == "__main__":
    # URL of the OWL ontology
    ontology_url = "https://github.com/sali-legal/LMSS/blob/main/LMSS.owl?raw=true"
    # Path to save the downloaded OWL file
    save_path = "lmss/LMSS.owl"

    # Download the ontology
    download_ontology(ontology_url, save_path)
