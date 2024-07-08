# SALI-E: Legal Matter Specification Standard Entity Recognizer - Updated Project Plan

## Project Overview

SALI-E is an open-source software project that applies the Legal Matter Standards Specification ontology (LMSS) in OWL format to text documents. The project uses Python, FastAPI, HTML, and JavaScript, leveraging open-source libraries and models. The goal is to extract relevant features from documents using Natural Language Processing (NLP) and Named-Entity Recognition (NER) based on the LMSS ontology.

## Current Status

### Completed Tasks

1. Ontology Analysis and Preprocessing

   - Implemented `LMSSOntologyParser` class
   - Created searchable index with vector embeddings
   - Identified high-level parent classes
   - Implemented efficient storage and retrieval
   - Added retry mechanism for ontology processing
   - Ensured consistency between JSON index and RDF graph

2. Entity Extraction and Classification Process

   - Implemented `EntityExtractor` class using spaCy
   - Integrated NER, noun phrase extraction, and keyword extraction (TF-IDF)
   - Implemented entity merging and deduplication
   - Generated vector embeddings for entities

3. Ontology Matching

   - Implemented `EnhancedOntologyMatcher` class
   - Developed context-aware matching algorithm
   - Implemented multiple matching techniques (semantic, fuzzy, context)
   - Added graph-based matching
   - Implemented fallback mechanisms for unmatched terms

4. Hierarchical Classification

   - Implemented ontology hierarchy traversal
   - Generated confidence scores for hierarchical classifications
   - Provided full hierarchical path for matched entities

5. Search Functionality

   - Implemented `LMSSSearch` class
   - Integrated regex, fuzzy, and vector search capabilities

6. Scripts and Utilities

   - Created `run_parser.py` for ontology processing
   - Developed `run_extraction.py` for entity extraction
   - Implemented `run_classifier.py` for matching and classification
   - Added `run_search.py` for interactive ontology exploration

7. Testing
   - Developed comprehensive test suite for ontology parsing and consistency checking

### Partially Implemented

1. User-Driven Class Selection

   - Basic filtering mechanism implemented in `LMSSSearch`

2. Confidence Scoring
   - Initial implementation in `EnhancedOntologyMatcher`

## Upcoming Tasks

### 1. API Development (FastAPI)

1.1. Implement FastAPI application structure
1.2. Create endpoint for document upload and processing
1.3. Develop endpoint for ontology class selection
1.4. Implement endpoint for retrieving extracted and matched entities
1.5. Create endpoint for detailed entity/class information
1.6. Integrate search functionality into API
1.7. Implement error handling and input validation
1.8. Add API documentation using Swagger UI

### 2. Front-End Development (HTML/JavaScript)

2.1. Design and implement main interface layout
2.2. Create text input field for document content
2.3. Implement document uploader (with PDF-to-text conversion)
2.4. Develop interface for selecting ontology branches
2.5. Add process initiation button
2.6. Design and implement results table displaying:

- Document Title
- Span Start
- Span End
- Text
- Parent Class
- Matching Class
- IRI
- Match Type
- Confidence Score
  2.7. Add JSON download button for full classification results
  2.8. Implement search interface for ontology exploration

### 7. Visualization (To be implemented - Later - Deferred)

7.1. Create a visual representation of extracted entities:

- Show their place in the ontology hierarchy
- Display relationships to other entities using a force-directed graph layout
- Implement zooming and panning features for large ontologies

### 3. Integration and Testing

3.1. Integrate FastAPI backend with front-end
3.2. Implement end-to-end testing of the entire pipeline
3.3. Perform stress testing and optimize performance
3.4. Conduct user acceptance testing

### 4. Documentation and Deployment

4.1. Write comprehensive API documentation
4.2. Create user guide for the web interface
4.3. Update README.md with project overview, installation, and usage instructions
4.4. Prepare Dockerfile for containerized deployment
4.5. Set up continuous integration and deployment (CI/CD) pipeline

### 5. Final Polishing and Open-Source Release

5.1. Refactor code for clarity and maintainability
5.2. Ensure code comments and docstrings are comprehensive
5.3. Review and update all documentation
5.4. Prepare the project for open-source release on GitHub
5.5. Choose and apply an appropriate open-source license (e.g., MIT License)

## Next Steps

1. Begin implementation of the FastAPI application structure
2. Start designing the front-end interface
3. Continue refining the matching and classification algorithms based on real-world testing
4. Plan for scalability and performance optimization

## Project Structure

```
sali-e/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── lmss_parser.py
│   ├── lmss_matcher_enhanced.py
│   ├── lmss_search.py
│   ├── entities.py
│   ├── run_parser.py
│   ├── run_extraction.py
│   ├── run_classifier.py
│   ├── lmss/
│   │   ├── LMSS.owl
│   │   ├── lmss_index.json
│   │   ├── lmss_graph.ttl
│   │   ├── lmss_hash.txt
│   │   ├── extraction_results.json
│   │   ├── matching_results.json
│   │   └── top_classes.json
|   ├── static/
│       └── js/
│           └── main.js
|   ├── templates/
│      └── index.html
├── tests/
│   ├── test_parser.py
│   ├── test_entities.py
│   ├── test_matcher.py
│   └── test_classifier.py
├── requirements.txt
├── README.md
└── Dockerfile
```
