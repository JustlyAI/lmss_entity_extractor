import json
import logging
from app.entity_extractor import EntityExtractor as Extractor
from app.lmss_ontology_parser import LMSSOntologyParser as Parser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Paths to pre-processed data
    index_path = "app/lmss_index.json"
    ontology_path = "app/LMSS.owl"

    # Load ontology entities
    with open(index_path, "r") as f:
        ontology_entities = json.load(f)

    logger.info(f"Loaded {len(ontology_entities)} ontology entities")

    # Initialize the UpdatedEntityExtractor
    extractor = Extractor(similarity_threshold=0.5, fuzzy_threshold=80)

    # Initialize the ontology parser (for additional context, not for processing)
    parser = Parser(ontology_path)

    # Example text (in a real scenario, this could be loaded from a file or user input)
    text = """
    The intellectual property lawyer specializes in patent law and copyright infringement cases.
    She also handles trademark disputes and trade secret litigation. Recently, she's been working
    on a high-profile case involving software licensing and open source compliance.
    """

    # Extract entities
    entities = extractor.extract_entities(text)
    logger.info(f"Extracted {len(entities)} entities")

    # Match entities
    matched_entities = extractor.match_entities(entities, ontology_entities)

    # Separate matched and unmatched entities
    matched = []
    unmatched = []
    for entity in matched_entities:
        if entity[4] is not None:  # entity[4] is the match (IRI)
            matched.append(entity)
        else:
            unmatched.append(entity)

    # Print results for matched entities
    print("\nMatched entities:")
    for entity, start, end, entity_type, match, similarity, match_type in matched:
        print(
            f"- {entity} ({start}, {end}) [{entity_type}]: {match} (similarity: {similarity:.2f}, match type: {match_type})"
        )

    # Print additional context for matched entities
    print("\nAdditional context for matched entities:")
    for entity, _, _, _, match, _, _ in matched:
        ont_entity = ontology_entities[match]
        print(f"- {entity}:")
        print(f"  IRI: {match}")
        print(f"  Label: {ont_entity.get('rdfs:label', 'N/A')}")
        print(f"  Description: {ont_entity.get('description', 'N/A')}")

    # Print information about unmatched entities
    print("\nUnmatched entities:")
    for entity, start, end, entity_type, _, similarity, _ in unmatched:
        print(
            f"- {entity} ({start}, {end}) [{entity_type}] (best similarity: {similarity:.2f})"
        )

    # Print statistics
    print(f"\nTotal entities extracted: {len(entities)}")
    print(f"Matched entities: {len(matched)}")
    print(f"Unmatched entities: {len(unmatched)}")


if __name__ == "__main__":
    main()
