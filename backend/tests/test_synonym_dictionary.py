import pytest
from backend.app.search_service.synonym_dictionary import SynonymDictionary

@pytest.fixture
def syn_dict():
    return SynonymDictionary()

def test_english_query_parsing(syn_dict):
    result = syn_dict.parse_query("red shirt man")
    resolved = result["resolved"]
    assert resolved.get("color") == "red"
    assert resolved.get("entity_type") == "human"

def test_hinglish_vernacular_parsing(syn_dict):
    result = syn_dict.parse_query("laal shirt aadmi")
    resolved = result["resolved"]
    assert resolved.get("color") == "red"
    assert resolved.get("entity_type") == "human"

def test_vehicle_and_color_parsing(syn_dict):
    result = syn_dict.parse_query("kaala gaadi")
    resolved = result["resolved"]
    assert resolved.get("color") == "black"
    assert resolved.get("entity_type") == "vehicle"

def test_posture_parsing(syn_dict):
    result = syn_dict.parse_query("crouching banda")
    resolved = result["resolved"]
    assert resolved.get("entity_type") == "human"
    assert resolved.get("posture") == "crouching"

def test_temporal_parsing(syn_dict):
    result = syn_dict.parse_query("kal subah")
    resolved = result["resolved"]
    assert "time_start" in resolved
    assert "time_end" in resolved

