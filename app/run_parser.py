import os
import sys
import logging
from app.lmss_parser import OntologyParser

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    ontology_url = "https://github.com/sali-legal/LMSS/blob/main/LMSS.owl"
    ontology_path = "app/lmss/LMSS.owl"
    index_path = "app/lmss/lmss_index.json"
    graph_path = "app/lmss/lmss_graph.ttl"
    hash_path = "app/lmss/lmss_hash.txt"
    top_classes_path = "app/lmss/top_classes.json"
    stats_path = "app/lmss/lmss_stats.json"

    # Ensure the app/lmss directory exists
    os.makedirs(os.path.dirname(ontology_path), exist_ok=True)

    # Step 1: Download ontology and check hash
    should_update = False
    if not os.path.exists(ontology_path):
        logger.info("LMSS ontology not found. Downloading...")
        should_update = True
    elif input("Update LMSS? (y/n): ").lower() == "y":
        should_update = True

    if should_update:
        if OntologyParser.download_ontology(ontology_url, ontology_path):
            logger.info("LMSS ontology downloaded/updated successfully.")
        else:
            logger.error("Failed to download/update LMSS ontology.")
            sys.exit(1)

    current_hash = OntologyParser.calculate_file_hash(ontology_path)
    if os.path.exists(hash_path):
        with open(hash_path, "r") as f:
            stored_hash = f.read().strip()
        if current_hash == stored_hash and not should_update:
            logger.info("LMSS ontology is up to date. No further processing needed.")
            return

    # Steps 2-5: Process ontology
    parser = OntologyParser(ontology_path)
    logger.info("Processing LMSS ontology...")
    stats = parser.process_ontology(
        index_path, graph_path, top_classes_path, stats_path
    )

    # Save new hash
    with open(hash_path, "w") as f:
        f.write(current_hash)

    logger.info("LMSS ontology processed successfully.")
    logger.info(f"Statistics: {stats}")


if __name__ == "__main__":
    main()
