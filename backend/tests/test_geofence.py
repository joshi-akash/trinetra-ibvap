import pytest
from backend.app.rule_engine.geofence_check import GeoFenceEvaluator

def test_point_inside_geofence():
    # Polygon around lat: 29.945-29.947, lon: 78.164-78.166
    # Note: Coordinates are (lon, lat) pairs
    geofence = [
        [78.1640, 29.9450],
        [78.1660, 29.9450],
        [78.1660, 29.9470],
        [78.1640, 29.9470],
        [78.1640, 29.9450]
    ]

    # Point clearly inside
    inside = GeoFenceEvaluator.is_inside_geofence(
        entity_lat=29.9460,
        entity_lon=78.1650,
        geofence_coords=geofence
    )
    assert inside is True

    # Point clearly outside
    outside = GeoFenceEvaluator.is_inside_geofence(
        entity_lat=29.9500,
        entity_lon=78.1700,
        geofence_coords=geofence
    )
    assert outside is False

def test_invalid_or_empty_geofence():
    assert GeoFenceEvaluator.is_inside_geofence(29.9460, 78.1650, None) is False
    assert GeoFenceEvaluator.is_inside_geofence(29.9460, 78.1650, []) is False
    assert GeoFenceEvaluator.is_inside_geofence(29.9460, 78.1650, [[78.0, 29.0], [78.1, 29.1]]) is False
