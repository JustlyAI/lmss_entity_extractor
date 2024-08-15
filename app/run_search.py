import json
import logging
import argparse
from pathlib import Path
from app.lmss_search import LMSSSearch
from rdflib import Graph
from typing import List

# Hard-coded paths for LMSS files
LMSS_DIR = Path(__file__).parent / "lmss"
ONTOLOGY_FILE = LMSS_DIR / "LMSS.owl"
INDEX_FILE = LMSS_DIR / "lmss_index.json"
GRAPH_FILE = LMSS_DIR / "lmss_graph.ttl"
TOP_CLASSES_FILE = LMSS_DIR / "top_classes.json"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(asctime)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_top_classes(file_path: Path) -> List[str]:
    with open(file_path, "r") as f:
        top_classes_data = json.load(f)
    return [cls["iri"] for cls in top_classes_data]


def main():
    parser = argparse.ArgumentParser(description="Search LMSS ontology")
    args = parser.parse_args()

    # Load the ontology
    ontology_graph = Graph()
    ontology_graph.parse(ONTOLOGY_FILE, format="xml")
    logger.info(f"Loaded ontology from {ONTOLOGY_FILE}")

    # Load top-level classes from JSON file
    top_level_classes = load_top_classes(TOP_CLASSES_FILE)
    logger.info(
        f"Loaded {len(top_level_classes)} top-level classes from {TOP_CLASSES_FILE}"
    )

    # Prompt user for keyword search
    keyword = input("Enter the keyword for search: ")

    # Prompt user for top classes
    top_classes_input = input(
        "Enter top classes separated by commas (or press Enter for all classes): "
    )
    if top_classes_input:
        selected_classes = [cls.strip() for cls in top_classes_input.split(",")]
    else:
        selected_classes = None

    # Perform search with selected classes
    searcher = LMSSSearch(INDEX_FILE, GRAPH_FILE, TOP_CLASSES_FILE)
    search_results = searcher.search(keyword, selected_branches=selected_classes)

    # Print search results
    print("\nSearch Results:")
    for result in search_results:
        print(
            f"- {result['label']} (IRI: {result['iri']}, Score: {result['score']:.2f})"
        )

    logger.info(f"Search results: {search_results}")


if __name__ == "__main__":
    main()
