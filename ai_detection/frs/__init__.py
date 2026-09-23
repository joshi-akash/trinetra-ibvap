"""FRS (Facial Recognition System) module for TRINETRA."""
from .face_matcher import FaceMatcher
from .known_suspects import KnownSuspectStore

__all__ = ["FaceMatcher", "KnownSuspectStore"]
