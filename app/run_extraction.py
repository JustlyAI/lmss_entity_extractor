import json
import logging
import argparse
from pathlib import Path
from app.entity_extraction import EntityExtractor, ExtractedEntity
from typing import List

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def save_results(data: List[ExtractedEntity], file_path: str):
    logger.info(f"Saving results to {file_path}")
    with open(file_path, "w") as f:
        json.dump([entity.dict() for entity in data], f, indent=2)


def process_text(text: str, extractor: EntityExtractor) -> List[ExtractedEntity]:
    logger.info("Extracting entities")
    extracted_entities = extractor.extract_entities(text)
    return extracted_entities


def print_summary(entities: List[ExtractedEntity]):
    print("\nExtraction Summary:")
    print(f"Total entities extracted: {len(entities)}")

    print("\nSample Extracted Entities:")
    for entity in entities[:5]:  # Print first 5 entities
        print(f"- {entity.text} ({entity.type})")


def calculate_statistics(entities: List[ExtractedEntity]) -> dict:
    stats = {
        "total_entities": len(entities),
        "entity_types": {},
    }

    for entity in entities:
        if entity.type not in stats["entity_types"]:
            stats["entity_types"][entity.type] = 0
        stats["entity_types"][entity.type] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="Extract entities from text")
    parser.add_argument("--input", type=str, help="Path to input text file")
    parser.add_argument(
        "--output",
        type=str,
        default="app/lmss/extraction_results.json",
        help="Path to output JSON file",
    )
    parser.add_argument(
        "--stats",
        type=str,
        default="app/lmss/extraction_stats.json",
        help="Path to output statistics JSON file",
    )
    args = parser.parse_args()

    extractor = EntityExtractor()

    if args.input:
        with open(args.input, "r") as f:
            text = f.read()
    else:
        # Example text if no input file is provided
        text = """
        The intellectual property lawyer specializes in patent law and copyright infringement cases.
        She also handles trademark disputes and trade secret litigation. Recently, she's been working
        on a high-profile case involving software licensing and open source compliance in Paris, Texas.
        """

    results = process_text(text, extractor)
    save_results(results, args.output)
    print_summary(results)

    # Calculate and save statistics
    stats = calculate_statistics(results)
    with open(args.stats, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Full results saved to {args.output}")
    logger.info(f"Statistics saved to {args.stats}")


if __name__ == "__main__":
    main()
