import json
import logging
from app.lmss_classification import EnhancedOntologyMatcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_classifier():
    # Load extracted data
    logger.info("Loading extracted data from app/lmss/extraction_results.json")
    with open("app/lmss/extraction_results.json", "r") as f:
        extracted_data = json.load(f)

    # Initialize the matcher
    matcher = EnhancedOntologyMatcher(
        graph_path="app/lmss/lmss_graph.ttl", index_path="app/lmss/lmss_index.json"
    )

    # Match and classify entities
    logger.info("Matching and classifying entities")
    results = matcher.match_entities(extracted_data)

    # Save results
    logger.info("Saving results to app/lmss/matching_results.json")
    with open("app/lmss/matching_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("Matching and Classification Summary:")
    total_entities = len(results)
    print(f"Total entities processed: {total_entities}")

    match_types = {}
    for entity in results:
        match_type = entity["match"]["match_type"]
        match_types[match_type] = match_types.get(match_type, 0) + 1

    for match_type, count in match_types.items():
        print(f"{match_type.capitalize()} entities: {count}")

    print("\nSample Matched Entities:")
    for entity in results[:10]:  # Print first 10 entities as a sample
        match = entity["match"]
        print(
            f"- {entity['text']} -> {match['iri']} (Label: {match['label']}) ({match['match_type']}, similarity: {match['similarity']:.2f})"
        )
        if "classification" in entity:
            hierarchy = " -> ".join(
                [f"{cls[0]} ({cls[1]:.2f})" for cls in entity["classification"]]
            )
            print(f"  Hierarchy: {hierarchy}")
        else:
            print("  No classification available")

    logger.info("Full results saved to app/lmss/matching_results.json")


if __name__ == "__main__":
    run_classifier()
