import json
import logging
import argparse
from pathlib import Path
from app.lmss_search import LMSSSearch, SearchResult
from typing import List, Dict, Any
import time

# Hard-coded paths for LMSS files
LMSS_DIR = Path(__file__).parent / "lmss"
INDEX_FILE = LMSS_DIR / "lmss_index.json"
GRAPH_FILE = LMSS_DIR / "lmss_graph.ttl"
TOP_CLASSES_FILE = LMSS_DIR / "top_classes.json"
SEARCH_RESULTS_FILE = LMSS_DIR / "search_results.json"
SEARCH_STATS_FILE = LMSS_DIR / "search_stats.json"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def calculate_stats(
    search_results: List[SearchResult], search_duration: float
) -> Dict[str, Any]:
    if not search_results:
        return {
            "total_results": 0,
            "top_result_score": 0,
            "average_score": 0,
            "search_duration": search_duration,
        }

    total_results = len(search_results)
    top_result_score = search_results[0].score
    average_score = sum(result.score for result in search_results) / total_results

    match_type_counts = {}
    branch_counts = {}
    for result in search_results:
        match_type_counts[result.match_type] = (
            match_type_counts.get(result.match_type, 0) + 1
        )
        branch_counts[result.branch] = branch_counts.get(result.branch, 0) + 1

    return {
        "total_results": total_results,
        "top_result_score": top_result_score,
        "average_score": average_score,
        "search_duration": search_duration,
        "match_type_counts": match_type_counts,
        "branch_counts": branch_counts,
    }


def print_search_results(results: List[SearchResult]):
    print("\nSearch Results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.label} (IRI: {result.iri})")
        print(f"   Score: {result.score:.4f}")
        print(f"   Match Type: {result.match_type}")
        print(f"   Parent Class: {result.parent_class or 'N/A'}")
        print(f"   Branch: {result.branch}")
        print(f"   Hierarchy: {' > '.join(result.hierarchy)}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Search LMSS ontology")
    parser.add_argument("--query", type=str, help="Search query")
    parser.add_argument(
        "--branches", type=str, help="Comma-separated list of branches to search within"
    )
    args = parser.parse_args()

    # Initialize LMSSSearch
    searcher = LMSSSearch(INDEX_FILE, GRAPH_FILE, TOP_CLASSES_FILE)
    logger.info("LMSSSearch initialized")

    # Get search query
    if args.query:
        query = args.query
    else:
        query = input("Enter the keyword for search: ")

    # Get branches to search within
    if args.branches:
        selected_branches = [branch.strip() for branch in args.branches.split(",")]
    else:
        print("\nAvailable branches:")
        for i, tc in enumerate(searcher.get_top_classes(), 1):
            print(f"{i}. {tc.label}")
        branch_input = input(
            "\nEnter the numbers of branches to search (comma-separated), or press Enter for all: "
        )
        if branch_input.strip():
            selected_indices = [int(idx.strip()) - 1 for idx in branch_input.split(",")]
            selected_branches = [
                searcher.get_top_classes()[idx].label for idx in selected_indices
            ]
        else:
            selected_branches = None

    # Perform search
    start_time = time.time()
    search_results = searcher.search(query, selected_branches=selected_branches)
    search_duration = time.time() - start_time

    # Print search results
    print_search_results(search_results)

    # Calculate and print search stats
    search_stats = calculate_stats(search_results, search_duration)
    print("\nSearch Statistics:")
    for key, value in search_stats.items():
        print(f"{key}: {value}")

    # Export search results to JSON file
    with open(SEARCH_RESULTS_FILE, "w") as f:
        json.dump([result.dict() for result in search_results], f, indent=4)
    logger.info(f"Search results exported to {SEARCH_RESULTS_FILE}")

    # Export search stats to JSON file
    with open(SEARCH_STATS_FILE, "w") as f:
        json.dump(search_stats, f, indent=4)
    logger.info(f"Search stats exported to {SEARCH_STATS_FILE}")


if __name__ == "__main__":
    main()
