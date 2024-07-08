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

1.1. Set up FastAPI application structure

- Create main.py with FastAPI app initialization
- Implement CORS middleware for frontend integration
- Set up dependency injection for shared resources

  1.2. Implement LMSS Ontology Management Endpoints
  1.2.1. GET /api/lmss/status - Return current LMSS status (ready, not ready, updating) - Serve: Left Nav status indicators (2.1.2)
  1.2.2. POST /api/lmss/update - Trigger LMSS ontology download and update process - Serve: "Download / Update" button (2.1.3)
  1.2.3. GET /api/lmss/statistics - Return LMSS statistics (branches, classes, attributes with embeddings) - Serve: LMSS Statistics section (2.1.5)
  1.2.4. GET /api/lmss/download/{file_type} - Download LMSS index or graph file - Serve: "Download Index" and "Download Graph" buttons (2.1.6)

  1.3. Implement Document Processing Endpoints
  1.3.1. POST /api/document/upload - Handle document upload and text extraction - Serve: "Single Document Uploader" button (2.2.2)
  1.3.2. POST /api/document/process - Process document text, extract entities, and classify - Serve: Text input area and classification process (2.2.1, 2.2.3, 2.2.4)

  1.4. Implement LMSS Class Selection Endpoints
  1.4.1. GET /api/lmss/classes - Retrieve top-level LMSS classes with entity counts - Serve: LMSS Class selection interface (2.2.3)
  1.4.2. POST /api/lmss/filter - Apply selected class filters to classification process - Serve: LMSS Class selection interface (2.2.3)

  1.5. Implement Search and Exploration Endpoints
  1.5.1. GET /api/search - Perform keyword search on LMSS ontology - Serve: Explorer "Key Word Search" (2.3.1, 2.3.3)
  1.5.2. GET /api/search/filter - Apply class filter to search results - Serve: Explorer class filter dropdown (2.3.2)

  1.6. Implement Error Handling and Validation
  1.6.1. Create custom exception handlers for common errors
  1.6.2. Implement request body validation using Pydantic models
  1.6.3. Add proper error responses with meaningful messages

### 2. Front-End Development (HTML/JavaScript)

2.1. Retractable Left Nav
2.1.1. Implement "Legal Matter Specification Standard" header
2.1.2. Add "LMSS Ready!" and "Get LMSS" status indicators
2.1.3. Create "Download / Update" button for LMSS ontology
2.1.4. Display "Preparing LMSS" progress indicator
2.1.5. Implement LMSS Statistics section

- Show number of branches (top classes)
- Display number of classes
- Show number of attributes with embeddings
  2.1.6. Add "Download Index" and "Download Graph" buttons
  2.1.7. Implement retractable functionality for left nav

  2.2. Classifier (Main Section - Always Open)
  2.2.1. Create large text input area for document content
  2.2.2. Implement "Single Document Uploader" button
  2.2.3. Develop LMSS Class selection interface

- Display top-level LMSS classes with entity counts
- Add checkboxes for each class
- Implement "Select All" and "Clear All" buttons
- Add helper text: "Select the classes you hope to classify by"
  2.2.4. Create results table with columns:
- Start
- End
- Text
- Branch
- Label
- Score + Explain for top-k
- IRI
  2.2.5. Add "Download JSON" button for classification results

  2.3. Explorer (Right Section - Make retractable)
  2.3.1. Implement "Key Word Search" input field
  2.3.2. Add class filter dropdown
  2.3.3. Create search output area with columns:

- Branch
- Label
- Score + Explain all hits
- IRI
  2.3.4. Add placeholder for future Q&A Bot integration

  2.4. General Layout and Functionality
  2.4.1. Implement responsive design for different screen sizes
  2.4.2. Ensure proper styling and consistent look across all sections
  2.4.3. Implement smooth transitions for retractable sections
  2.4.4. Add loading indicators for asynchronous operations
  2.4.5. Implement error handling and user feedback mechanisms

#### Next Steps (deferred)

##### API Next Steps

1.7. Implement API Documentation
1.7.1. Add docstrings to all API endpoints
1.7.2. Configure Swagger UI for interactive API documentation
1.7.3. Include example requests and responses in the documentation

1.8. Optimize API Performance
1.8.1. Implement caching for frequently accessed data
1.8.2. Use background tasks for long-running processes
1.8.3. Implement pagination for large result sets

1.9. Implement WebSocket for Real-time Updates
1.9.1. Set up WebSocket connection for LMSS update progress
1.9.2. Implement real-time updates for classification process

1.10. Security Measures
1.10.1. Implement rate limiting to prevent abuse
1.10.2. Add authentication for sensitive endpoints (if required)
1.10.3. Implement proper input sanitization to prevent injection attacks

##### Front-End Next Steps

2.6. Performance Optimization
2.6.1. Implement lazy loading for large result sets
2.6.2. Optimize rendering of large tables
2.6.3. Implement caching mechanisms for frequently accessed data
2.6.4. Minimize API calls by batching requests where possible

2.7. Accessibility and Usability
2.7.1. Ensure proper keyboard navigation
2.7.2. Implement ARIA attributes for screen reader compatibility
2.7.3. Add tooltips and help text for complex features
2.7.4. Ensure color contrast meets accessibility standards

2.8. Testing and Quality Assurance
2.8.1. Develop unit tests for front-end components
2.8.2. Implement end-to-end tests for user workflows
2.8.3. Perform cross-browser testing
2.8.4. Conduct usability testing with potential users

## Project Structure

```
sali-e/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── lmss_parser.py
│   ├── lmss_classification.py
│   ├── lmss_search.py
│   ├── entity_extraction.py
│   ├── run_parser.py
│   ├── run_extraction.py
│   ├── run_classifier.py
│   ├── run_search.py
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
│   └── test_embeddings.py
├── requirements.txt
├── README.md
└── Dockerfile
```

##### Previously...

2.5. Integration with Backend
2.5.1. Connect LMSS download/update functionality to LMSSOntologyParser
2.5.2. Integrate text input and document upload with EntityExtractor
2.5.3. Link LMSS class selection with OntologyMatcher filtering
2.5.4. Connect classification results display with LMSSClassification output
2.5.5. Integrate Explorer search functionality with LMSSSearch
2.5.6. Implement real-time updates for LMSS statistics

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
