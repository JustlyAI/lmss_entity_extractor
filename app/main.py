from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional
import json
import os

from app.lmss_parser import LMSSOntologyParser
from app.extract_entities import EntityExtractor
from app.lmss_classification import OntologyMatcher
from app.lmss_search import LMSSSearch

app = FastAPI(title="SALI-E: Legal Matter Specification Standard Class Recognizer")

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Initialize components
ontology_parser = LMSSOntologyParser("app/lmss/LMSS.owl")
entity_extractor = EntityExtractor()
ontology_matcher = OntologyMatcher(
    graph_path="app/lmss/lmss_graph.ttl", index_path="app/lmss/lmss_index.json"
)
lmss_search = LMSSSearch(
    index_path="app/lmss/lmss_index.json",
    graph_path="app/lmss/lmss_graph.ttl",
    top_classes_path="app/lmss/top_classes.json",
)


class ProcessTextRequest(BaseModel):
    text: str
    selected_classes: Optional[List[str]] = None


class SearchRequest(BaseModel):
    query: str
    selected_branches: Optional[List[str]] = None


@app.post("/process")
async def process_text(request: ProcessTextRequest):
    try:
        # Extract entities
        entities = entity_extractor.extract_entities(request.text)

        # Match entities to ontology
        matched_entities = ontology_matcher.match_entities(
            entities, request.selected_classes
        )

        return JSONResponse(content={"entities": matched_entities})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_file(file: UploadFile = File(...), selected_classes: str = Form(None)):
    try:
        contents = await file.read()
        text = contents.decode("utf-8")
        selected_classes_list = (
            json.loads(selected_classes) if selected_classes else None
        )

        # Extract entities
        entities = entity_extractor.extract_entities(text)

        # Match entities to ontology
        matched_entities = ontology_matcher.match_entities(
            entities, selected_classes_list
        )

        return JSONResponse(content={"entities": matched_entities})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
async def search_ontology(request: SearchRequest):
    try:
        search_results = lmss_search.search(request.query, request.selected_branches)
        return JSONResponse(content={"results": search_results})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/top_classes")
async def get_top_classes():
    try:
        top_classes = lmss_search.get_top_classes()
        return JSONResponse(content={"top_classes": top_classes})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
