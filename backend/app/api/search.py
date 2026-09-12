from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models import User
from backend.app.schemas import SearchRequest, SearchResponse
from backend.app.auth.jwt_auth import get_current_user
from backend.app.search_service.query_translator import query_translator
from backend.app.api.entities import _to_entity_out

router = APIRouter(prefix="/api/search", tags=["Search"])

@router.post("", response_model=SearchResponse)
def forensic_search(
    req: SearchRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    results, total_count, parsed_filters = query_translator.execute_search(db=db, request=req)
    
    return SearchResponse(
        parsed_filters=parsed_filters,
        total_count=total_count,
        results=[_to_entity_out(r) for r in results]
    )
