from typing import List, Optional, Tuple
from shapely.geometry import Point, Polygon

class GeoFenceEvaluator:
    """Evaluates spatial intersection between detected entities and camera geo-fences using Shapely."""

    @staticmethod
    def is_inside_geofence(
        entity_lat: float,
        entity_lon: float,
        geofence_coords: Optional[List[List[float]]]
    ) -> bool:
        """
        Check if an entity's (lon, lat) coordinate intersects or is inside the polygon.
        Coordinates are expected as [ [lon1, lat1], [lon2, lat2], ... ]
        """
        if not geofence_coords or len(geofence_coords) < 3:
            return False

        try:
            # Note: GeoJSON / PostGIS standard is (lon, lat) = (x, y)
            poly = Polygon(geofence_coords)
            if not poly.is_valid:
                poly = poly.buffer(0)  # Fix self-intersecting or slight geometry issues

            point = Point(entity_lon, entity_lat)
            return poly.contains(point) or poly.touches(point) or poly.intersects(point)
        except Exception:
            return False

    @staticmethod
    def bbox_intersects_geofence(
        bbox: List[float],
        calibration_matrix: Optional[dict],
        geofence_coords: Optional[List[List[float]]]
    ) -> bool:
        """
        Optional image-space bounding box check if pixel-space geofence is defined.
        """
        if not geofence_coords or len(geofence_coords) < 3:
            return False
        try:
            x1, y1, x2, y2 = bbox
            footprint_point = Point((x1 + x2) / 2.0, y2)  # Base/feet of entity
            poly = Polygon(geofence_coords)
            return poly.contains(footprint_point) or poly.intersects(footprint_point)
        except Exception:
            return False

geofence_evaluator = GeoFenceEvaluator()
