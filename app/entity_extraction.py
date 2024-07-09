import re
import spacy
from spacy.language import Language
from spacy.tokens import Doc
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
from pydantic import BaseModel
import logging

# Load spaCy's English stop words
nlp = spacy.load("en_core_web_sm")
STOP_WORDS = nlp.Defaults.stop_words


class ExtractedEntity(BaseModel):
    text: str
    start: int
    end: int
    type: str
    label: str = None
    vector: List[float] = None
    confidence: float = None  # Confidence score of the extraction
    source: str = None  # Source of the entity (e.g., NER, NOUN_PHRASE, KEYWORD)
    context: str = None  # Surrounding text for context


def remove_leading_stop_words(text: str) -> Tuple[str, int]:
    words = text.split()
    start_offset = 0
    for i, word in enumerate(words):
        if word.lower() not in STOP_WORDS:
            return " ".join(words[i:]), start_offset
        start_offset += len(word) + 1  # +1 for the space
    return text, 0  # Return original if all words are stop words


def merge_entities(
    entities: List[Tuple[str, int, int, str]]
) -> List[Tuple[str, int, int, str]]:
    logger = logging.getLogger(__name__)
    sorted_entities = sorted(entities, key=lambda x: (x[1], -x[2]))
    merged = []
    i = 0
    while i < len(sorted_entities):
        current = sorted_entities[i]
        j = i + 1
        while j < len(sorted_entities):
            next_entity = sorted_entities[j]

            # Debug: Log current and next entity
            logger.debug(f"Current entity: {current}")
            logger.debug(f"Next entity: {next_entity}")

            # Check for overlap or adjacency
            if next_entity[1] <= current[2] or (
                next_entity[1] - current[2] <= 1
                and current[3] == next_entity[3] == "NOUN_PHRASE"
            ):
                if current[3].startswith("NER_"):
                    # Always keep NER entities as is
                    break
                elif next_entity[3].startswith("NER_"):
                    # Prefer NER entities
                    current = next_entity
                elif current[3] == "NOUN_PHRASE" and next_entity[3] == "NOUN_PHRASE":
                    # Merge overlapping or adjacent noun phrases
                    current = (
                        f"{current[0]} {next_entity[0]}".strip(),
                        min(current[1], next_entity[1]),
                        max(current[2], next_entity[2]),
                        "NOUN_PHRASE",
                    )
                elif current[3] == "KEYWORD" and next_entity[3] in [
                    "NOUN_PHRASE",
                    "KEYWORD",
                ]:
                    # Prefer noun phrases over keywords, or longer keywords
                    if next_entity[3] == "NOUN_PHRASE" or len(next_entity[0]) > len(
                        current[0]
                    ):
                        current = next_entity
                else:
                    break
                j += 1
            else:
                break

        # Debug: Log merged entity
        logger.debug(f"Merged entity: {current}")

        # Remove leading stop words
        cleaned_text, offset = remove_leading_stop_words(current[0])
        current = (cleaned_text, current[1] + offset, current[2], current[3])

        # Check if the current entity is a subset of any existing merged entity
        if not any(m[1] <= current[1] and m[2] >= current[2] for m in merged):
            merged.append(current)
        i = j

    # Remove duplicates while preserving order
    seen = set()
    deduped = []
    for entity in merged:
        if entity[0].lower() not in seen:
            deduped.append(entity)
            seen.add(entity[0].lower())

    return deduped


@Language.component("combined_extractor")
def combined_extractor(doc):
    # Collect NER entities
    ner_entities = [
        (ent.text, ent.start_char, ent.end_char, f"NER_{ent.label_}")
        for ent in doc.ents
    ]

    # Collect noun phrases
    noun_phrases = [
        (chunk.text, chunk.start_char, chunk.end_char, "NOUN_PHRASE")
        for chunk in doc.noun_chunks
        if not chunk.root.is_stop
    ]

    # Perform keyword extraction
    tfidf = TfidfVectorizer(stop_words="english")
    try:
        tfidf.fit([doc.text])
        feature_names = tfidf.get_feature_names_out()
        tfidf_scores = tfidf.transform([doc.text]).data
        keywords = []
        for i in tfidf_scores.argsort()[-10:][::-1]:  # Top 10 keywords
            keyword = feature_names[i]
            for match in re.finditer(
                r"\b" + re.escape(keyword) + r"\b", doc.text.lower()
            ):
                start, end = match.span()
                keywords.append((keyword, start, end, "KEYWORD"))
    except ValueError:
        # Handle empty input
        keywords = []

    # Store all entities in doc.user_data
    doc.user_data["all_entities"] = ner_entities + noun_phrases + keywords
    return doc


class EntityExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.add_pipe("combined_extractor", last=True)
        self.sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")
        self.logger = logging.getLogger(__name__)

    def extract_entities(self, text: str) -> List[ExtractedEntity]:
        self.logger.info("Extracting entities from text")

        if not text.strip():
            self.logger.info("Empty input, returning empty list")
            return []

        doc = self.nlp(text)

        all_entities = doc.user_data["all_entities"]
        self.logger.debug(f"All entities before merging: {all_entities}")
        merged_entities = merge_entities(all_entities)
        self.logger.debug(f"Merged entities: {merged_entities}")

        self.logger.info(f"Extracted and merged {len(merged_entities)} entities")

        # Generate embeddings only for merged entities
        entities = []
        for text, start, end, ent_type in merged_entities:
            vector = self.sentence_transformer.encode(text).tolist()
            # Assuming a default confidence score and source for now
            confidence = 1.0  # Placeholder for actual confidence score
            source = ent_type.split("_")[0]  # Extract source from type

            # Extract full sentences for context
            sent_start = max(
                0, start - 100
            )  # Take up to 100 characters before the entity
            sent_end = min(
                len(doc.text), end + 100
            )  # Take up to 100 characters after the entity
            context = doc.text[sent_start:sent_end]

            entities.append(
                ExtractedEntity(
                    text=text,
                    start=start,
                    end=end,
                    type=ent_type,
                    vector=vector,
                    confidence=confidence,
                    source=source,
                    context=context,
                )
            )

        self.logger.info("Generated embeddings for all entities")
        return entities
