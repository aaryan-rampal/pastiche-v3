"""
Test helper: send a points .npy file to the running backend POST /api/sketch/match-points.

Set FILE_PATH below to the .npy you want to send (relative to repo root).
Requires the backend server to be running (uvicorn main:app --reload --port 8000).

This script sends an HTTP request to the backend, not a mocked in-process call.
"""

import json
import numpy as np
import requests
from loguru import logger

# === USER CONFIG: set this to the .npy file you want to send ===
FILE_PATH = "logs/sketch_debug/sketch_20251011_175633.npy"

# Backend URL (change port/host if your dev server runs elsewhere)
BACKEND_URL = "http://localhost:8000"
ENDPOINT = f"{BACKEND_URL}/api/sketch/match-points"


def load_points(path: str) -> np.ndarray:
    if not path or not path.endswith(".npy"):
        raise ValueError("Please set FILE_PATH to a .npy file")
    logger.info(f"Loading points from {path}")
    pts = np.load(path)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Loaded points must be shape (N,2), got {pts.shape}")
    pts[:, 1] = -pts[:, 1]
    return pts


def build_payload(points: np.ndarray, top_k_faiss: int = 5, top_k_final: int = 3):
    # MatchRequestBody expects { "points": { "points": [[x,y], ...] }, "top_k_faiss": int, "top_k_final": int }
    return {
        "points": {"points": points.tolist()},
        "top_k_faiss": top_k_faiss,
        "top_k_final": top_k_final,
    }


def send_request(payload: dict):
    headers = {"Content-Type": "application/json"}
    logger.info(f"Posting to {ENDPOINT}")
    resp = requests.post(
        ENDPOINT, data=json.dumps(payload), headers=headers, timeout=60
    )
    try:
        data = resp.json()
    except Exception:
        resp.raise_for_status()
        return None
    resp.raise_for_status()
    return data


def pretty_print_response(data: dict):
    print("\n=== match_sketch_points response ===")
    print(json.dumps(data, indent=2))
    print("=== end ===\n")


if __name__ == "__main__":
    try:
        pts = load_points(FILE_PATH)
        payload = build_payload(pts, top_k_faiss=5, top_k_final=3)
        resp = send_request(payload)
        pretty_print_response(resp)
    except Exception as e:
        logger.exception(f"Error sending sketch match request: {e}")
