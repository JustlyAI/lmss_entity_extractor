import json
import logging
import argparse
from pathlib import Path
from app.extract_entities import EntityExtractor, Entity
from typing import List

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def save_results(data: List[Entity], file_path: str):
    logger.info(f"Saving results to {file_path}")
    with open(file_path, "w") as f:
        json.dump([entity.dict() for entity in data], f, indent=2)


def process_text(text: str, extractor: EntityExtractor) -> List[Entity]:
    logger.info("Extracting entities")
    extracted_entities = extractor.extract_entities(text)
    return extracted_entities


def print_summary(entities: List[Entity]):
    print("\nExtraction Summary:")
    print(f"Total entities extracted: {len(entities)}")

    print("\nSample Extracted Entities:")
    for entity in entities[:5]:  # Print first 5 entities
        print(f"- {entity.text} ({entity.type})")


def main():
    parser = argparse.ArgumentParser(description="Extract entities from text")
    parser.add_argument("--input", type=str, help="Path to input text file")
    parser.add_argument(
        "--output",
        type=str,
        default="app/lmss/extraction_results.json",
        help="Path to output JSON file",
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
        on a high-profile case involving software licensing and open source compliance.
        """

    results = process_text(text, extractor)
    save_results(results, args.output)
    print_summary(results)

    logger.info(f"Full results saved to {args.output}")


if __name__ == "__main__":
    main()
