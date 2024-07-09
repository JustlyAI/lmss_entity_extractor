import json
import logging
from app.lmss_classification import OntologyClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_classifier():
    # Load extracted data
    logger.info("Loading extracted data from app/lmss/extraction_results.json")
    try:
        with open("app/lmss/extraction_results.json", "r") as f:
            extracted_data = json.load(f)
        logger.info(f"Loaded {len(extracted_data)} extracted entities")
    except FileNotFoundError:
        logger.error(
            "extraction_results.json not found. Please run the extraction process first."
        )
        return
    except json.JSONDecodeError:
        logger.error(
            "Error decoding extraction_results.json. The file may be corrupted."
        )
        return

    if not extracted_data:
        logger.warning("No extracted entities found in extraction_results.json")
        return

    # Initialize the matcher with a lower similarity threshold
    try:
        matcher = OntologyClassifier(
            graph_path="app/lmss/lmss_graph.ttl",
            index_path="app/lmss/lmss_index.json",
            similarity_threshold=0.3,  # Lower threshold for more matches
        )
    except Exception as e:
        logger.error(f"Error initializing OntologyClassifier: {str(e)}")
        return

    # Print a sample of ontology entities
    matcher.print_ontology_sample(5)

    # Match and classify entities
    logger.info("Matching and classifying entities")
    results = matcher.match_entities(extracted_data)

    # Save results
    logger.info("Saving results to app/lmss/matching_results.json")
    with open("app/lmss/matching_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("Matching and Classification Summary:")
    print(f"Total entities processed: {len(extracted_data)}")
    print(f"Total entities matched: {len(results)}")

    print("\nSample Matched Entities:")
    for entity in results[:10]:  # Print first 10 entities as a sample
        print(
            f"- {entity['text']} -> {entity['label']} (Branch: {entity['branch']}, Score: {entity['score']:.2f}, IRI: {entity['iri']})"
        )

    logger.info("Full results saved to app/lmss/matching_results.json")


if __name__ == "__main__":
    run_classifier()
