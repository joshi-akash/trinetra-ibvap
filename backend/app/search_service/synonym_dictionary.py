import re
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple

class SynonymDictionary:
    """
    Deterministic Layer-1 multilingual synonym dictionary (FR-SRCH-01).
    Maps English, Hindi (Devanagari), and Hinglish (Latin script) terms to canonical filters.
    Requires ZERO external internet or GPU inference.
    """

    COLOR_MAP = {
        "red": ["red", "laal", "lal", "लाल", "crimson", "maroon"],
        "blue": ["blue", "neela", "nila", "नीला", "neeli", "nili", "नीली", "navy"],
        "black": ["black", "kaala", "kala", "काला", "kaali", "kali", "काली", "dark"],
        "white": ["white", "safed", "safaid", "सफेद", "light"],
        "green": ["green", "hara", "हरा", "hari", "हरी"],
        "yellow": ["yellow", "peela", "pila", "पीला", "peeli", "pili", "पीली"],
        "grey": ["grey", "gray", "dhusar", "धूसर", "slate"]
    }


    ENTITY_MAP = {
        "human": ["human", "person", "man", "woman", "guy", "boy", "girl", "aadmi", "admi", "आदमी", "insan", "इंसान", "banda", "बंदा", "vyakti", "व्यक्ति"],
        "vehicle": ["vehicle", "car", "truck", "bike", "motorcycle", "gaadi", "gadi", "गाड़ी", "vahan", "वाहन", "jeep", "suv"],
        "animal": ["animal", "dog", "cow", "horse", "jaanwar", "janwar", "जानवर", "pashu", "पशु", "kutta", "गाय"]
    }

    POSTURE_MAP = {
        "crouching": ["crouching", "crouch", "jhuka", "jhuk", "झुका", "squatting"],
        "sprinting": ["sprinting", "running", "bhaag", "bhagta", "भागता", "daud", "दौड़"],
        "prone": ["prone", "crawling", "rengna", "रेंगना", "lying down"]
    }

    LOW_LIGHT_MAP = {
        True: ["andhera", "andhere", "अंधेरा", "अंधेरे", "dark", "darkness", "low_light", "lowlight", "kam_roshni", "roshni_kam", "dim"]
    }

    PROP_MAP = {
        "weapon": ["weapon", "hathiyar", "hathyar", "हथियार", "bandook", "gun", "rifle", "arms"],
        "covered_face": ["covered_face", "naqab", "nakab", "नकाब", "mask", "mukhota"],
        "large_backpack": ["large_backpack", "backpack", "bag", "jhola", "thaila", "बैग"]
    }

    def __init__(self):
        # Build inverted lookup indices
        self.inverted_colors = {}
        for canonical, syns in self.COLOR_MAP.items():
            for s in syns:
                self.inverted_colors[s.lower()] = canonical

        self.inverted_entities = {}
        for canonical, syns in self.ENTITY_MAP.items():
            for s in syns:
                self.inverted_entities[s.lower()] = canonical

        self.inverted_postures = {}
        for canonical, syns in self.POSTURE_MAP.items():
            for s in syns:
                self.inverted_postures[s.lower()] = canonical

        self.inverted_low_light = {}
        for canonical, syns in self.LOW_LIGHT_MAP.items():
            for s in syns:
                self.inverted_low_light[s.lower()] = canonical

        self.inverted_props = {}
        for canonical, syns in self.PROP_MAP.items():
            for s in syns:
                self.inverted_props[s.lower()] = canonical

    def parse_query(self, query_text: Optional[str]) -> Dict[str, Any]:
        """
        Extracts structured filters from raw query text.
        Returns:
            resolved_terms: dict of canonical attributes
            unresolved_terms: list of unrecognized words
        """
        if not query_text:
            return {"resolved": {}, "unresolved": []}

        # Normalize tokens
        tokens = re.findall(r'[\w]+', query_text.lower())
        resolved = {}
        unresolved = []

        now = datetime.now(timezone.utc)
        i = 0
        while i < len(tokens):
            token = tokens[i]

            # 1. Colors
            if token in self.inverted_colors:
                resolved["color"] = self.inverted_colors[token]
                i += 1
                continue

            # 2. Entities
            if token in self.inverted_entities:
                resolved["entity_type"] = self.inverted_entities[token]
                i += 1
                continue

            # 3. Postures
            if token in self.inverted_postures:
                resolved["posture"] = self.inverted_postures[token]
                i += 1
                continue

            # 4. Low-Light / Darkness
            if token in self.inverted_low_light:
                resolved["is_low_light"] = True
                i += 1
                continue

            # 5. Threat Props
            if token in self.inverted_props:
                resolved["prop"] = self.inverted_props[token]
                i += 1
                continue

            # 6. License Plate pattern (e.g. DL01AB1234 or HR26DQ5555)
            if re.match(r'^[a-z]{2}[0-9]{1,2}[a-z]{0,3}[0-9]{4}$', token):
                resolved["plate_text"] = token.upper()
                i += 1
                continue

            # 7. Relative Time Indicators (kal, aaj, subah, shaam, raat)
            if token in ["kal", "कल", "yesterday"]:
                yesterday = now - timedelta(days=1)
                resolved["time_start"] = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
                resolved["time_end"] = yesterday.replace(hour=23, minute=59, second=59, microsecond=0)
                i += 1
                continue

            if token in ["aaj", "आज", "today"]:
                resolved["time_start"] = now.replace(hour=0, minute=0, second=0, microsecond=0)
                resolved["time_end"] = now
                i += 1
                continue

            if token in ["subah", "सुबह", "morning"]:
                # Morning: 05:00 to 11:59
                base_day = resolved.get("time_start", now)
                resolved["time_start"] = base_day.replace(hour=5, minute=0, second=0, microsecond=0)
                resolved["time_end"] = base_day.replace(hour=11, minute=59, second=59, microsecond=0)
                i += 1
                continue

            if token in ["shaam", "शाम", "evening"]:
                # Evening: 17:00 to 20:59
                base_day = resolved.get("time_start", now)
                resolved["time_start"] = base_day.replace(hour=17, minute=0, second=0, microsecond=0)
                resolved["time_end"] = base_day.replace(hour=20, minute=59, second=59, microsecond=0)
                i += 1
                continue

            if token in ["raat", "रात", "night"]:
                # Night: 21:00 to 04:59 (and implies low-light condition)
                base_day = resolved.get("time_start", now)
                resolved["time_start"] = base_day.replace(hour=21, minute=0, second=0, microsecond=0)
                resolved["time_end"] = base_day + timedelta(days=1)
                resolved["time_end"] = resolved["time_end"].replace(hour=4, minute=59, second=59, microsecond=0)
                resolved["is_low_light"] = True
                i += 1
                continue

            # Common stop words
            if token in ["wala", "wali", "wale", "tha", "thi", "koi", "in", "the", "a", "is", "of", "and", "mein"]:
                i += 1
                continue

            unresolved.append(token)
            i += 1

        return {
            "resolved": resolved,
            "unresolved": unresolved
        }

synonym_dictionary = SynonymDictionary()

