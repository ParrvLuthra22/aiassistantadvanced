#!/usr/bin/env python3
"""Enroll or refresh FRIDAY face-auth profile."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running this script directly: `python scripts/enroll_face.py`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.face_auth import FaceAuthenticator


def main() -> int:
    parser = argparse.ArgumentParser(description="Enroll current face for FRIDAY local auth")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera device index")
    parser.add_argument(
        "--camera-fallback",
        action="store_true",
        default=True,
        help="Try additional camera IDs when enrollment fails on the selected camera",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/face_auth",
        help="Directory for saved face profile",
    )
    args = parser.parse_args()

    auth = FaceAuthenticator(data_dir=args.data_dir, camera_id=args.camera_id)
    result = auth.force_enroll()
    if result.granted:
        print(result.message)
        return 0

    if args.camera_fallback and "no clear face detected" in result.message.lower():
        for alt_camera_id in (1, 2, 3):
            if alt_camera_id == args.camera_id:
                continue
            alt_auth = FaceAuthenticator(data_dir=args.data_dir, camera_id=alt_camera_id)
            alt_result = alt_auth.force_enroll()
            if alt_result.granted:
                print(f"{alt_result.message} (camera-id={alt_camera_id})")
                return 0

    print(
        f"{result.message}\n"
        "Tips: look directly at the camera in good light, close other camera apps, "
        "and grant camera permission to your terminal/IDE. "
        "You can also retry with --camera-id 1."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
