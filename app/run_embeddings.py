import os
import asyncio
import time
from app.lmss_ontology_parser import LMSSOntologyParser as Parser
from rdflib.namespace import RDFS


async def main():
    ontology_url = "https://github.com/sali-legal/LMSS/blob/main/LMSS.owl"
    ontology_path = "app/LMSS.owl"
    index_path = "app/lmss_index.json"
    graph_path = "app/lmss_graph.ttl"
    hash_path = "app/lmss_hash.txt"

    def get_stored_hash() -> str:
        if os.path.exists(hash_path):
            with open(hash_path, "r") as f:
                return f.read().strip()
        return ""

    def store_hash(hash_value: str):
        with open(hash_path, "w") as f:
            f.write(hash_value)

    start_time = time.time()

    if os.path.exists(ontology_path):
        overwrite = (
            input("LMSS.owl already exists. Do you want to overwrite it? (Y/N): ")
            .strip()
            .lower()
        )
        if overwrite == "y":
            Parser.download_ontology(ontology_url, ontology_path)
            print(
                "Ontology file has been overwritten. You will need to recalculate embeddings."
            )
        else:
            print("Using existing ontology file.")
    else:
        Parser.download_ontology(ontology_url, ontology_path)

    parser = Parser(ontology_path)

    current_hash = Parser.calculate_file_hash(ontology_path)
    stored_hash = get_stored_hash()

    if (
        os.path.exists(index_path)
        and os.path.exists(graph_path)
        and current_hash == stored_hash
    ):
        print(
            "Embeddings already present and ontology has not changed. Skipping generation."
        )
    else:
        await parser.process_ontology(index_path, graph_path)
        print(f"Embeddings have been generated and saved.")
        store_hash(current_hash)

    end_time = time.time()
    duration = end_time - start_time
    print(f"Total operation time: {duration:.2f} seconds")

    print("\nDebug: Checking for embeddings in the graph")
    embedding_count = sum(
        1 for _ in parser.graph.triples((None, parser.LMSS.hasEmbedding, None))
    )
    print(f"Number of entities with embeddings: {embedding_count}")

    print("\nTop-class statistics:")
    sorted_top_classes = sorted(
        parser.top_classes, key=lambda cls: parser.get_literal(cls, RDFS.label)
    )
    for top_class in sorted_top_classes:
        entities_count = len(parser.get_entities_under_class(top_class))
        print(
            f"- {parser.get_literal(top_class, RDFS.label)}: {entities_count} entities"
        )


if __name__ == "__main__":
    asyncio.run(main())
