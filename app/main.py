import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List
from app.lmss_parser import LMSSOntologyParser
from app.entities import EntityRecognitionEngine

app = FastAPI(title="LMSS Entity Recognizer")

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")
templates.env.cache = {}

# Check for LMSS in app/LMSS.owl or download from Github
ontology_url = "https://github.com/sali-legal/LMSS/blob/main/LMSS.owl"
save_path = "app/LMSS.owl"

if not os.path.exists(save_path):
    LMSSOntologyParser.download_ontology(ontology_url, save_path)
else:
    print(f"Ontology file already exists at {save_path}")

# Initialize the ontology parser
ontology_parser = LMSSOntologyParser("app/LMSS.owl")
entities = ontology_parser.get_entities()

# Initialize the entity recognition engine
recognition_engine = EntityRecognitionEngine(entities)


class RecognizedEntity(BaseModel):
    start: int
    end: int
    text: str
    entity: str
    iri: str


class RecognitionRequest(BaseModel):
    text: str


class RecognitionResponse(BaseModel):
    entities: List[RecognizedEntity]


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/recognize", response_model=RecognitionResponse)
async def recognize(request: RecognitionRequest):
    try:
        recognized_entities = recognition_engine.recognize_entities(request.text)
        return RecognitionResponse(entities=recognized_entities)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/entities")
async def get_entities():
    return {"entities": list(entities.keys())}


@app.get("/entity/{entity_name}")
async def get_entity_info(entity_name: str):
    entity_info = ontology_parser.get_entity_info(entity_name)
    if entity_info:
        return entity_info
    else:
        raise HTTPException(status_code=404, detail="Entity not found")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
