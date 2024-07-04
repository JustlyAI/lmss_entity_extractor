import re


class EntityRecognitionEngine:
    def __init__(self, entities):
        self.entities = entities
        self.entity_patterns = self.compile_patterns()

    def compile_patterns(self):
        return {
            re.compile(r"\b" + re.escape(entity) + r"\b", re.IGNORECASE): entity
            for entity in self.entities
        }

    def recognize_entities(self, text):
        recognized = []
        for pattern, entity in self.entity_patterns.items():
            for match in pattern.finditer(text):
                recognized.append(
                    {
                        "start": match.start(),
                        "end": match.end(),
                        "text": match.group(),
                        "entity": entity,
                        "iri": self.entities[entity]["iri"],
                    }
                )
        return sorted(recognized, key=lambda x: x["start"])
