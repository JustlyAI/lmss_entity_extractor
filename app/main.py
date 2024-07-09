import os
import json
import asyncio
import logging
from fastapi import FastAPI, Request, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import PyPDF2
import docx
import io
from sentence_transformers import SentenceTransformer

from app.lmss_parser import OntologyParser
from app.entity_extraction import EntityExtractor, ExtractedEntity
from app.lmss_classification import OntologyClassifier
from app.lmss_search import LMSSSearch

app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
ONTOLOGY_URL = "https://github.com/sali-legal/LMSS/blob/main/LMSS.owl"
ONTOLOGY_PATH = "app/lmss/LMSS.owl"
INDEX_PATH = "app/lmss/lmss_index.json"
GRAPH_PATH = "app/lmss/lmss_graph.ttl"
HASH_PATH = "app/lmss/lmss_hash.txt"
TOP_CLASSES_PATH = "app/lmss/top_classes.json"
STATS_PATH = "app/lmss/lmss_stats.json"
EXTRACTION_STATS_PATH = "app/lmss/extraction_stats.json"

lmss_status = "not_ready"
lmss_parser = None
extractor = None
classifier = None
searcher = None


class LMSSClass(BaseModel):
    iri: str
    label: str
    entities_count: int


class ProcessRequest(BaseModel):
    text: str
    selected_classes: Optional[List[str]] = None


class ExtractionStats(BaseModel):
    total_entities: int
    entity_types: Dict[str, int]


class ProcessResponse(BaseModel):
    results: List[Dict[str, Any]]
    extraction_stats: ExtractionStats


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


def check_lmss_status():
    global lmss_status, lmss_parser, extractor, classifier, searcher
    if all(
        os.path.exists(path)
        for path in [
            ONTOLOGY_PATH,
            INDEX_PATH,
            GRAPH_PATH,
            HASH_PATH,
            TOP_CLASSES_PATH,
            STATS_PATH,
        ]
    ):
        current_hash = OntologyParser.calculate_file_hash(ONTOLOGY_PATH)
        with open(HASH_PATH, "r") as f:
            stored_hash = f.read().strip()
        if current_hash == stored_hash:
            lmss_status = "ready"
            sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")
            lmss_parser = OntologyParser(ONTOLOGY_PATH, model=sentence_transformer)
            extractor = EntityExtractor()
            classifier = OntologyClassifier(
                GRAPH_PATH, INDEX_PATH, TOP_CLASSES_PATH, similarity_threshold=0.7
            )
            searcher = LMSSSearch(
                INDEX_PATH, GRAPH_PATH, TOP_CLASSES_PATH
            )  # Use GRAPH_PATH instead of lmss_parser.graph
        else:
            lmss_status = "outdated"
    else:
        lmss_status = "not_ready"


async def process_lmss():
    global lmss_status, lmss_parser, extractor, classifier, searcher
    try:
        # Step 1: Download ontology and check hash
        if OntologyParser.download_ontology(ONTOLOGY_URL, ONTOLOGY_PATH):
            logger.info("LMSS ontology downloaded successfully.")
        else:
            logger.error("Failed to download LMSS ontology.")
            lmss_status = "error"
            return

        # Steps 2-5: Process ontology
        sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")
        lmss_parser = OntologyParser(ONTOLOGY_PATH, model=sentence_transformer)
        stats = lmss_parser.process_ontology(
            INDEX_PATH, GRAPH_PATH, TOP_CLASSES_PATH, STATS_PATH
        )

        # Save new hash
        current_hash = OntologyParser.calculate_file_hash(ONTOLOGY_PATH)
        with open(HASH_PATH, "w") as f:
            f.write(current_hash)

        # Initialize other components
        extractor = EntityExtractor()
        classifier = OntologyClassifier(
            GRAPH_PATH, INDEX_PATH, similarity_threshold=0.7
        )
        searcher = LMSSSearch(
            INDEX_PATH, GRAPH_PATH, TOP_CLASSES_PATH
        )  # Use GRAPH_PATH instead of lmss_parser.graph

        lmss_status = "ready"
        logger.info("LMSS ontology processed successfully.")
        logger.info(f"Statistics: {stats}")

    except Exception as e:
        lmss_status = "error"
        logger.error(f"Error processing ontology: {str(e)}", exc_info=True)


@app.on_event("startup")
async def startup_event():
    check_lmss_status()


@app.get("/api/lmss/status")
async def get_lmss_status():
    return {"status": lmss_status}


@app.post("/api/lmss/update")
async def update_lmss(background_tasks: BackgroundTasks):
    global lmss_status
    lmss_status = "processing"
    background_tasks.add_task(process_lmss)
    return {"message": "LMSS update started"}


@app.get("/api/lmss/statistics")
async def get_lmss_statistics():
    if lmss_status != "ready":
        raise HTTPException(status_code=400, detail="LMSS is not ready")
    with open(STATS_PATH, "r") as f:
        stats = json.load(f)
    # Add more detailed embedding statistics if available
    if lmss_parser:
        stats["embedding_fields"] = [
            "rdfs_label",
            "skos_definition",
            "skos_prefLabel",
            "skos_altLabel",
        ]
        stats["total_embeddings"] = sum(
            1
            for _ in lmss_parser.graph.triples(
                (None, lmss_parser.LMSS.hasEmbedding, None)
            )
        )
    return stats


@app.get("/api/lmss/download/{file_type}")
async def download_lmss(file_type: str):
    if file_type not in ["index", "graph"]:
        raise HTTPException(status_code=400, detail="Invalid file type")
    file_path = INDEX_PATH if file_type == "index" else GRAPH_PATH
    return FileResponse(file_path, filename=os.path.basename(file_path))


@app.post("/api/document/upload")
async def upload_document(file: UploadFile = File(...)):
    content = await file.read()
    if file.filename.endswith(".txt"):
        text = content.decode()
    elif file.filename.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    elif file.filename.endswith(".docx"):
        doc = docx.Document(io.BytesIO(content))
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    elif file.filename.endswith(".doc"):
        raise HTTPException(status_code=400, detail="Unsupported file type: .doc")
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    return {"text": text}


@app.post("/api/document/process", response_model=ProcessResponse)
async def process_document(request: ProcessRequest):
    if lmss_status != "ready":
        raise HTTPException(status_code=400, detail="LMSS is not ready")
    try:
        extracted_entities = extractor.extract_entities(request.text)

        # Calculate extraction stats
        extraction_stats = ExtractionStats(
            total_entities=len(extracted_entities),
            entity_types={entity.type: 0 for entity in extracted_entities},
        )
        for entity in extracted_entities:
            extraction_stats.entity_types[entity.type] += 1

        # Save extraction stats
        with open(EXTRACTION_STATS_PATH, "w") as f:
            json.dump(extraction_stats.dict(), f, indent=2)

        # Use the classifier to match entities
        classified_entities = classifier.match_entities(
            [entity.dict() for entity in extracted_entities]
        )

        # Check for missing 'branch' property and log any issues
        for entity in classified_entities:
            if "branch" not in entity:
                logger.error(f"Missing 'branch' property for entity: {entity}")
                entity["branch"] = "Unknown"  # Set a default value

        return ProcessResponse(
            results=classified_entities, extraction_stats=extraction_stats
        )
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error processing document: {str(e)}"
        )


@app.get("/api/lmss/classes")
async def get_lmss_classes():
    if lmss_status != "ready":
        raise HTTPException(status_code=400, detail="LMSS is not ready")
    with open(TOP_CLASSES_PATH, "r") as f:
        top_classes = json.load(f)
    return [LMSSClass(**cls) for cls in top_classes]


@app.get("/api/search")
async def search_lmss(query: str, class_filter: Optional[str] = None):
    if lmss_status != "ready":
        raise HTTPException(status_code=400, detail="LMSS is not ready")
    selected_branches = [class_filter] if class_filter else None
    results = searcher.search(query, selected_branches)
    return {"results": results}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
