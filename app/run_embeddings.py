import os
import asyncio
import time
from app.lmss_ontology_parser import EnhancedLMSSOntologyParser
from rdflib.namespace import RDFS


async def main():
    ontology_url = "https://github.com/sali-legal/LMSS/blob/main/LMSS.owl"
    ontology_path = "app/LMSS.owl"
    index_path = "app/lmss_index.json"
    graph_path = "app/lmss_graph.ttl"
    hash_path = "app/lmss_hash.txt"

    def get_stored_hash() -> str:
        """Reads the stored hash from the file."""
        if os.path.exists(hash_path):
            with open(hash_path, "r") as f:
                return f.read().strip()
        return ""

    def store_hash(hash_value: str):
        """Stores the hash to a file."""
        with open(hash_path, "w") as f:
            f.write(hash_value)

    # Start the timer
    start_time = time.time()

    # Check if the ontology file already exists
    if os.path.exists(ontology_path):
        overwrite = (
            input("LMSS.owl already exists. Do you want to overwrite it? (Y/N): ")
            .strip()
            .lower()
        )
        if overwrite == "y":
            EnhancedLMSSOntologyParser.download_ontology(ontology_url, ontology_path)
            print(
                "Ontology file has been overwritten. You will need to recalculate embeddings."
            )
        else:
            print("Using existing ontology file.")
    else:
        EnhancedLMSSOntologyParser.download_ontology(ontology_url, ontology_path)

    parser = EnhancedLMSSOntologyParser(ontology_path)

    # Calculate the hash of the current ontology file
    current_hash = EnhancedLMSSOntologyParser.calculate_file_hash(ontology_path)
    stored_hash = get_stored_hash()

    # Check if embeddings are already present and if the ontology has changed
    if (
        os.path.exists(index_path)
        and os.path.exists(graph_path)
        and current_hash == stored_hash
    ):
        print(
            "Embeddings already present and ontology has not changed. Skipping embedding generation."
        )
    else:
        await parser.process_ontology(index_path, graph_path)
        print(
            f"Embeddings have been generated and saved to {index_path} and {graph_path}."
        )
        # Store the new hash
        store_hash(current_hash)

    # End the timer and print the duration
    end_time = time.time()
    duration = end_time - start_time
    print(f"Total operation time: {duration:.2f} seconds")

    # After processing the ontology
    print("\nDebug: Checking for embeddings in the graph")
    embedding_count = sum(
        1 for _ in parser.graph.triples((None, parser.LMSS.hasEmbedding, None))
    )
    print(f"Number of entities with embeddings: {embedding_count}")

    # Example search
    # search_query = "intellectual property"
    # search_field = "rdfs:label"
    # search_results = parser.search(search_query, search_field)
    # print(f"\nSearch results for '{search_query}' in field '{search_field}':")
    # for label, iri, similarity in search_results[:5]:
    #     print(f"- {label} (IRI: {iri}, Similarity: {similarity:.2f})")

    # Print some statistics about top-classes
    print("\nTop-class statistics:")
    for top_class in parser.top_classes:
        entities_count = len(parser.get_entities_under_class(top_class))
        print(
            f"- {parser.get_literal(top_class, RDFS.label)}: {entities_count} entities"
        )


if __name__ == "__main__":
    asyncio.run(main())
