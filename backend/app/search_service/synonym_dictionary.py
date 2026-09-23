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
        "grey": ["grey", "gray", "dhusar", "धूसर", "slate"],
        "brown": ["brown", "bhura", "bhoora", "भूरा"],
        "orange": ["orange", "narangi", "santra", "नारंगी"],
        "pink": ["pink", "gulabi", "गुलाबी"],
        "purple": ["purple", "baingani", "बैंगनी", "violet"]
    }

    ENTITY_MAP = {
        "human": ["human", "person", "man", "woman", "guy", "boy", "girl", "aadmi", "admi", "आदमी", "insan", "इंसान", "banda", "बंदा", "vyakti", "व्यक्ति", "aurat", "औरत", "pedestrian", "individual", "ladka", "ladki", "babu", "suspect"],
        "vehicle": ["vehicle", "car", "truck", "bike", "motorcycle", "gaadi", "gadi", "गाड़ी", "vahan", "वाहन", "jeep", "suv", "bus", "van", "auto", "rickshaw", "tempo", "tractor", "cycle", "scooter", "thar", "bullet"],
        "animal": ["animal", "dog", "cow", "horse", "jaanwar", "janwar", "जानवर", "pashu", "पशु", "kutta", "कुत्ता", "गाय", "bull", "bhalu", "hathi", "bhed", "bakri"]
    }

    POSTURE_MAP = {
        "standing": ["standing", "stand", "khada", "khadi", "खड़ा", "खड़ी", "upright"],
        "crouching": ["crouching", "crouch", "jhuka", "jhuk", "झुका", "squatting", "squat", "sitting", "baitha", "बैठा"],
        "sprinting": ["sprinting", "sprint", "running", "run", "bhaag", "bhagta", "भागता", "daud", "दौड़", "fast", "tez", "तेज़", "rapid"],
        "prone": ["prone", "crawling", "crawl", "rengna", "रेंगना", "lying", "soya", "creeping", "stealth"]
    }

    DIRECTION_MAP = {
        "West": ["west", "left", "bayen", "baayein", "बायें", "बाएं", "paschim", "पश्चिम"],
        "East": ["east", "right", "dayen", "daayein", "दायें", "दाएं", "purva", "पूर्व"],
        "North": ["north", "up", "retreating", "away", "peeche", "पीछे", "uttar", "उत्तर"],
        "South": ["south", "down", "advancing", "coming", "forward", "aage", "आगे", "dakshin", "दक्षिण"],
        "North-West": ["northwest", "north-west"],
        "North-East": ["northeast", "north-east"],
        "South-West": ["southwest", "south-west"],
        "South-East": ["southeast", "south-east"],
        "Stationary": ["stationary", "loitering", "idle", "still", "ruka", "रुका", "khada_hua"]
    }

    LOW_LIGHT_MAP = {
        True: ["andhera", "andhere", "अंधेरा", "अंधेरे", "dark", "darkness", "low_light", "lowlight", "kam_roshni", "roshni_kam", "dim", "night", "raat", "रात"]
    }

    PROP_MAP = {
        "weapon": [
            "weapon", "hathiyar", "hathiyar", "हथियार", "bandook", "बंदूक", "gun", "rifle",
            "arms", "knife", "chaku", "चाकू", "chhuri", "dagger", "sword", "pistol",
            "tamancha", "katta", "desi_katta", "revolver", "firearm", "ammunition",
            "lathi", "danda", "rod", "stick", "explosive", "bomb", "barood"
        ],
        "covered_face": ["covered_face", "naqab", "nakab", "नकाब", "mask", "mukhota", "मुखौटा"],
        "large_backpack": [
            "large_backpack", "backpack", "bag", "बैग", "jhola", "झोला", "thaila", "थैला",
            "suitcase", "briefcase", "luggage", "duffel", "petticase", "sack", "bori"
        ]
    }

    ALERT_MAP = {
        True: ["alert", "breach", "threat", "danger", "khatra", "खतरा", "ghuspeth", "घुसपैठ", "suspect", "infiltrator", "violator", "chori", "hamla", "attack"]
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

        self.inverted_directions = {}
        for canonical, syns in self.DIRECTION_MAP.items():
            for s in syns:
                self.inverted_directions[s.lower()] = canonical

        self.inverted_low_light = {}
        for canonical, syns in self.LOW_LIGHT_MAP.items():
            for s in syns:
                self.inverted_low_light[s.lower()] = canonical

        self.inverted_props = {}
        for canonical, syns in self.PROP_MAP.items():
            for s in syns:
                self.inverted_props[s.lower()] = canonical

        self.inverted_alerts = {}
        for canonical, syns in self.ALERT_MAP.items():
            for s in syns:
                self.inverted_alerts[s.lower()] = canonical

    def parse_query(self, query_text: Optional[str]) -> Dict[str, Any]:
        """
        Extracts structured filters from raw query text.
        Returns:
            resolved_terms: dict of canonical attributes
            unresolved_terms: list of unrecognized words for broad-spectrum matching
        """
        if not query_text:
            return {"resolved": {}, "unresolved": []}

        # Multi-word normalizations
        normalized = query_text.lower()
        normalized = re.sub(r'\bmoving\s+left\b', 'left', normalized)
        normalized = re.sub(r'\bmoving\s+right\b', 'right', normalized)
        normalized = re.sub(r'\bheading\s+west\b', 'west', normalized)
        normalized = re.sub(r'\bheading\s+east\b', 'east', normalized)
        normalized = re.sub(r'\bheading\s+north\b', 'north', normalized)
        normalized = re.sub(r'\bheading\s+south\b', 'south', normalized)
        normalized = re.sub(r'\blow\s+light\b', 'low_light', normalized)
        normalized = re.sub(r'\bdesi\s+katta\b', 'tamancha', normalized)
        normalized = re.sub(r'\blying\s+down\b', 'prone', normalized)

        # Normalize tokens
        tokens = re.findall(r'[\w\-]+', normalized)
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

            # 4. Directions (West, East, North, South, etc.)
            if token in self.inverted_directions:
                resolved["direction"] = self.inverted_directions[token]
                i += 1
                continue

            # 5. Low-Light / Darkness
            if token in self.inverted_low_light:
                resolved["is_low_light"] = True
                i += 1
                continue

            # 6. Threat Props (weapon, knife, gun, backpack, etc.)
            if token in self.inverted_props:
                resolved["prop"] = self.inverted_props[token]
                i += 1
                continue

            # 7. Alert / Breach indicators
            if token in self.inverted_alerts:
                resolved["alert_only"] = True
                i += 1
                continue

            # 8. License Plate pattern (e.g. DL01AB1234 or HR26DQ5555)
            if re.match(r'^[a-z]{2}[0-9]{1,2}[a-z]{0,3}[0-9]{4}$', token):
                resolved["plate_text"] = token.upper()
                i += 1
                continue

            # 9. Relative Time Indicators (kal, aaj, subah, shaam, raat)
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
                base_day = resolved.get("time_start", now)
                resolved["time_start"] = base_day.replace(hour=5, minute=0, second=0, microsecond=0)
                resolved["time_end"] = base_day.replace(hour=11, minute=59, second=59, microsecond=0)
                i += 1
                continue

            if token in ["shaam", "शाम", "evening"]:
                base_day = resolved.get("time_start", now)
                resolved["time_start"] = base_day.replace(hour=17, minute=0, second=0, microsecond=0)
                resolved["time_end"] = base_day.replace(hour=20, minute=59, second=59, microsecond=0)
                i += 1
                continue

            if token in ["raat", "रात", "night"]:
                base_day = resolved.get("time_start", now)
                resolved["time_start"] = base_day.replace(hour=21, minute=0, second=0, microsecond=0)
                resolved["time_end"] = base_day + timedelta(days=1)
                resolved["time_end"] = resolved["time_end"].replace(hour=4, minute=59, second=59, microsecond=0)
                resolved["is_low_light"] = True
                i += 1
                continue

            # Common stop words & clothing carrier nouns
            if token in [
                "wala", "wali", "wale", "tha", "thi", "koi", "in", "the", "a", "is", "of", "and", "mein",
                "with", "near", "towards", "at", "by", "to", "shirt", "tshirt", "t-shirt", "pant", "pants",
                "jeans", "kurta", "pajama", "kapda", "kapde", "clothes", "clothing", "dress", "wearing",
                "pehne", "pehna"
            ]:
                i += 1
                continue

            unresolved.append(token)
            i += 1

        return {
            "resolved": resolved,
            "unresolved": unresolved
        }

synonym_dictionary = SynonymDictionary()

