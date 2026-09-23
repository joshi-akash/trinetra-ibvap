from typing import Dict, Any, List, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_, desc

from backend.app.models import EntityLog
from backend.app.schemas import SearchRequest, EntityLogOut
from backend.app.search_service.synonym_dictionary import synonym_dictionary

class QueryTranslator:
    """Translates resolved search filters into parameterized database queries."""

    @staticmethod
    def execute_search(db: Session, request: SearchRequest) -> Tuple[List[EntityLog], int, Dict[str, Any]]:
        # 1. Parse free-text query with Layer-1 multilingual synonym dictionary
        parsed = synonym_dictionary.parse_query(request.query)
        resolved = parsed.get("resolved", {})

        # 2. Build SQLAlchemy Filter Conditions
        conditions = [EntityLog.deleted_manually == False]

        # Entity type (explicit takes precedence, otherwise parsed)
        entity_type = request.entity_type or resolved.get("entity_type")
        if entity_type:
            conditions.append(EntityLog.entity_type == entity_type)

        # Color (matches upper_color or lower_color)
        color = request.upper_color or resolved.get("color")
        if color:
            conditions.append(
                or_(
                    EntityLog.upper_color.ilike(f"%{color}%"),
                    EntityLog.lower_color.ilike(f"%{color}%")
                )
            )

        # Posture filter
        posture = request.posture or resolved.get("posture")
        if posture:
            conditions.append(EntityLog.posture.ilike(f"%{posture}%"))

        # Low-light / Darkness filter
        is_low_light = request.is_low_light if request.is_low_light is not None else resolved.get("is_low_light")
        if is_low_light is not None:
            conditions.append(EntityLog.is_low_light == is_low_light)

        # License plate text filter
        plate_text = resolved.get("plate_text")
        if plate_text:
            conditions.append(EntityLog.plate_text.ilike(f"%{plate_text}%"))

        # Threat prop filter (e.g. weapon / gun)
        prop = resolved.get("prop")
        if prop:
            conditions.append(EntityLog.rule_fired.ilike(f"%{prop}%"))

        # Direction filter (e.g. West, East, North, South, Moving Left, etc.)
        direction = request.direction or resolved.get("direction")
        if direction:
            conditions.append(EntityLog.direction.ilike(f"%{direction}%"))

        # Camera ID
        if request.camera_id:
            conditions.append(EntityLog.camera_id == request.camera_id)

        # Alert only filter
        alert_only = request.alert_only if request.alert_only is not None else resolved.get("alert_only")
        if alert_only is not None:
            conditions.append(EntityLog.is_alert == alert_only)

        # Time range
        start_time = request.start_time or resolved.get("time_start")
        if start_time:
            conditions.append(EntityLog.timestamp >= start_time)

        end_time = request.end_time or resolved.get("time_end")
        if end_time:
            conditions.append(EntityLog.timestamp <= end_time)

        # 3. Broad fallback filter for unrecognized / free-text keywords
        unresolved = parsed.get("unresolved", [])
        if unresolved:
            has_primary_filters = bool(
                request.entity_type or resolved.get("entity_type") or
                request.upper_color or resolved.get("color") or
                request.posture or resolved.get("posture") or
                request.direction or resolved.get("direction") or
                resolved.get("prop") or resolved.get("plate_text")
            )
            unresolved_filters = []
            for u_tok in unresolved:
                term = f"%{u_tok}%"
                unresolved_filters.append(
                    or_(
                        EntityLog.camera_id.ilike(term),
                        EntityLog.plate_text.ilike(term),
                        EntityLog.face_name.ilike(term),
                        EntityLog.rule_fired.ilike(term),
                        EntityLog.vehicle_type.ilike(term),
                        EntityLog.posture.ilike(term),
                        EntityLog.direction.ilike(term),
                        EntityLog.upper_color.ilike(term),
                        EntityLog.lower_color.ilike(term),
                        EntityLog.entity_type.ilike(term),
                        EntityLog.id.ilike(term),
                    )
                )
            if unresolved_filters:
                if not has_primary_filters:
                    conditions.append(and_(*unresolved_filters))
                else:
                    conditions.append(or_(*unresolved_filters))

        # 4. Query Execution
        query = db.query(EntityLog).filter(*conditions).order_by(desc(EntityLog.timestamp))
        total_count = query.count()
        fetch_limit = min(max(request.limit, 1), 500)
        results = query.offset(request.offset).limit(fetch_limit).all()

        return results, total_count, parsed


query_translator = QueryTranslator()
