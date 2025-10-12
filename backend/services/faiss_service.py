"""Service for FAISS index operations."""

from core.config import settings

from pathlib import Path
from loguru import logger
from typing import List, Tuple, Optional
import numpy as np
from models.schemas import ContourFAISSIndex


class FAISSService:
    """Singleton service for managing FAISS index."""

    _instance: Optional["FAISSService"] = None
    _index: Optional[ContourFAISSIndex] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FAISSService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize FAISS service (loads index on first call)."""
        if self._index is None:
            self.load_index()

    def load_index(self, index_path: Optional[str] = None) -> None:
        """Load FAISS index from disk.

        Args:
            index_path: Path to index file (without extension). If None, uses default path.
        """
        if index_path is None:
            # Default to backend/data/contour_hu_index
            backend_dir = Path(__file__).parent.parent
            index_path = str(backend_dir / "data" / "contour_hu_index")

        self._index = ContourFAISSIndex(use_weighted_distance=True)
        self._index.load_index(index_path)
        print(
            f"FAISS index loaded with {len(self._index.contour_metadata_s3)} contours"
        )

    def search_similar_contours(
        self, sketch_contour: np.ndarray, k: int = settings.faiss_top_k
    ) -> List[Tuple[float, str, int]]:
        """Search for similar contours using FAISS index.

        Args:
            sketch_contour: Sketch contour points (N, 2)
            k: Number of nearest neighbors to return

        Returns:
            List of (distance, image_path, contour_idx) tuples
        """
        if self._index is None:
            logger.error("FAISS index not loaded!")
            raise RuntimeError("FAISS index not loaded")

        logger.debug(f"Searching FAISS index for top {k} matches...")
        results = self._index.search_similar_contours(sketch_contour, k)
        logger.debug(f"FAISS search returned {len(results)} results")

        return results

    def get_metadata(self) -> List[Tuple[str, int]]:
        """Get contour metadata (image_path, contour_idx) for all indexed contours."""
        if self._index is None:
            raise RuntimeError("FAISS index not loaded")
        return self._index.contour_metadata_s3
