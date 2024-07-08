import os
import asyncio
import time
import json
from fastapi import FastAPI, Request, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional
import PyPDF2
import io
from app.lmss_parser import LMSSOntologyParser
from app.entity_recognition import EntityExtractor
from app.lmss_classification import OntologyMatcher
from app.lmss_search import LMSSSearch
from rdflib.namespace import RDFS


app = FastAPI()

# Mount the static files directory
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Set up the templates directory
templates = Jinja2Templates(directory="app/templates")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
lmss_status = "not_ready"
lmss_statistics = {}
extractor = None
matcher = None
searcher = None


class LMSSClass(BaseModel):
    iri: str
    label: str
    entities_count: int


class ProcessRequest(BaseModel):
    text: str
    selected_classes: Optional[List[str]] = None


def check_lmss_status():
    global lmss_status, lmss_statistics
    ontology_path = "app/lmss/LMSS.owl"
    index_path = "app/lmss/lmss_index.json"
    graph_path = "app/lmss/lmss_graph.ttl"
    hash_path = "app/lmss/lmss_hash.txt"

    if (
        os.path.exists(ontology_path)
        and os.path.exists(index_path)
        and os.path.exists(graph_path)
    ):
        parser = LMSSOntologyParser(ontology_path)
        current_hash = LMSSOntologyParser.calculate_file_hash(ontology_path)
        stored_hash = ""
        if os.path.exists(hash_path):
            with open(hash_path, "r") as f:
                stored_hash = f.read().strip()

        if current_hash == stored_hash:
            lmss_status = "ready"
            lmss_statistics = {
                "branches": len(parser.top_classes),
                "classes": len(parser.entities),
                "attributes_with_embeddings": sum(
                    1
                    for _ in parser.graph.triples(
                        (None, parser.LMSS.hasEmbedding, None)
                    )
                ),
            }
        else:
            lmss_status = "not_ready"
    else:
        lmss_status = "not_ready"


async def process_lmss():
    global lmss_status, lmss_statistics, extractor, matcher, searcher
    ontology_url = "https://github.com/sali-legal/LMSS/blob/main/LMSS.owl"
    ontology_path = "app/lmss/LMSS.owl"
    index_path = "app/lmss/lmss_index.json"
    graph_path = "app/lmss/lmss_graph.ttl"
    hash_path = "app/lmss/lmss_hash.txt"
    top_classes_path = "app/lmss/top_classes.json"

    def get_stored_hash() -> str:
        if os.path.exists(hash_path):
            with open(hash_path, "r") as f:
                return f.read().strip()
        return ""

    def store_hash(hash_value: str):
        with open(hash_path, "w") as f:
            f.write(hash_value)

    start_time = time.time()

    try:
        if os.path.exists(ontology_path):
            current_hash = LMSSOntologyParser.calculate_file_hash(ontology_path)
            stored_hash = get_stored_hash()

            if (
                os.path.exists(index_path)
                and os.path.exists(graph_path)
                and current_hash == stored_hash
            ):
                parser = LMSSOntologyParser(ontology_path)
                parser.load_entities(index_path)
                lmss_status = "ready"
                lmss_statistics = {
                    "branches": len(parser.top_classes),
                    "classes": len(parser.entities),
                    "attributes_with_embeddings": sum(
                        1
                        for _ in parser.graph.triples(
                            (None, parser.LMSS.hasEmbedding, None)
                        )
                    ),
                }
            else:
                LMSSOntologyParser.download_ontology(ontology_url, ontology_path)
                parser = LMSSOntologyParser(ontology_path)
                await parser.process_ontology(index_path, graph_path)
                current_hash = LMSSOntologyParser.calculate_file_hash(ontology_path)
                store_hash(current_hash)
                lmss_status = "ready"
                lmss_statistics = {
                    "branches": len(parser.top_classes),
                    "classes": len(parser.entities),
                    "attributes_with_embeddings": sum(
                        1
                        for _ in parser.graph.triples(
                            (None, parser.LMSS.hasEmbedding, None)
                        )
                    ),
                }
        else:
            LMSSOntologyParser.download_ontology(ontology_url, ontology_path)
            parser = LMSSOntologyParser(ontology_path)
            await parser.process_ontology(index_path, graph_path)
            current_hash = LMSSOntologyParser.calculate_file_hash(ontology_path)
            store_hash(current_hash)
            lmss_status = "ready"
            lmss_statistics = {
                "branches": len(parser.top_classes),
                "classes": len(parser.entities),
                "attributes_with_embeddings": sum(
                    1
                    for _ in parser.graph.triples(
                        (None, parser.LMSS.hasEmbedding, None)
                    )
                ),
            }

        # Save top classes to JSON
        sorted_top_classes = sorted(
            parser.top_classes, key=lambda cls: parser.get_literal(cls, RDFS.label)
        )
        top_classes_data = [
            {
                "iri": str(top_class),
                "label": parser.get_literal(top_class, RDFS.label),
                "entities_count": len(parser.get_entities_under_class(top_class)),
            }
            for top_class in sorted_top_classes
        ]
        with open(top_classes_path, "w") as f:
            json.dump(top_classes_data, f, indent=2)

        # Initialize components after processing LMSS
        extractor = EntityExtractor()
        matcher = OntologyMatcher(graph_path, index_path)
        searcher = LMSSSearch(index_path, graph_path, top_classes_path)

        end_time = time.time()
        duration = end_time - start_time
        print(f"Total operation time: {duration:.2f} seconds")

    except Exception as e:
        lmss_status = "error"
        print(f"Error processing ontology: {str(e)}")


@app.on_event("startup")
async def startup_event():
    check_lmss_status()


@app.get("/api/lmss/status")
async def get_lmss_status():
    global lmss_status
    return {"status": lmss_status}


@app.post("/api/lmss/update")
async def update_lmss(background_tasks: BackgroundTasks):
    global lmss_status
    lmss_status = "processing"
    background_tasks.add_task(process_lmss)
    return {"message": "LMSS update started"}


@app.get("/api/lmss/statistics")
async def get_lmss_statistics():
    return lmss_statistics


@app.get("/api/lmss/download/{file_type}")
async def download_lmss(file_type: str):
    if file_type not in ["index", "graph"]:
        raise HTTPException(status_code=400, detail="Invalid file type")
    file_path = f"app/lmss/lmss_{file_type}.{'json' if file_type == 'index' else 'ttl'}"
    return FileResponse(
        file_path,
        filename=f"lmss_{file_type}.{'json' if file_type == 'index' else 'ttl'}",
    )


# Document Processing Endpoints
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
        raise HTTPException(status_code=400, detail="Unsupported file type: .docx")
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    return {"text": text}


@app.post("/api/document/process")
async def process_document(request: ProcessRequest):
    if not extractor or not matcher:
        raise HTTPException(status_code=500, detail="LMSS not processed yet")
    extracted_entities = extractor.extract_entities(request.text)
    classified_entities = matcher.match_entities(
        extracted_entities, request.selected_classes
    )
    return {"results": classified_entities}


# LMSS Class Selection Endpoints
@app.get("/api/lmss/classes")
async def get_lmss_classes():
    with open("app/lmss/top_classes.json", "r") as f:
        top_classes = json.load(f)
    return [LMSSClass(**cls) for cls in top_classes]


# Search and Exploration Endpoints
@app.get("/api/search")
async def search_lmss(query: str, class_filter: Optional[str] = None):
    if not searcher:
        raise HTTPException(status_code=500, detail="LMSS not processed yet")
    selected_branches = [class_filter] if class_filter else None
    results = searcher.search(query, selected_branches)
    return {"results": results}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
