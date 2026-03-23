"""
POC Padel Analytics
===================
Produces rally statistics, shot speed estimates, and player heatmaps from
previously cached tracking results (run main.py first to generate the cache).

Usage:
    python poc.py

Outputs (written to poc_output/):
    rally_summary.json    – all detected rallies with shots and speeds
    best_rally.json       – the best rally (most shots, fastest shots)
    heatmap_player1.png   – player 1 court coverage heatmap
    heatmap_player2.png   – player 2 court coverage heatmap
    heatmap_player3.png   – player 3 court coverage heatmap
    heatmap_player4.png   – player 4 court coverage heatmap
"""

import json
from pathlib import Path

import supervision as sv

from config import (
    INPUT_VIDEO_PATH,
    BALL_TRACKER_LOAD_PATH,
    KEYPOINTS_TRACKER_LOAD_PATH,
    PLAYERS_TRACKER_LOAD_PATH,
    FIXED_COURT_KEYPOINTS_LOAD_PATH,
)
from trackers.ball_tracker.ball_tracker import Ball
from trackers.keypoints_tracker.keypoints_tracker import Keypoints, Keypoint
from trackers.players_tracker.players_tracker import Players
from analytics.projected_court import ProjectedCourt
from analytics.data_analytics import DataAnalytics
from analytics.rally_detector import RallyDetector
from analytics.heatmap_generator import HeatmapGenerator

OUTPUT_DIR = Path("poc_output")


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------

def _load_ball_detections(path: str) -> list:
    with open(path) as f:
        data = json.load(f)
    return [Ball.from_json(obj) for obj in data]


def _load_keypoints_detections(path: str) -> list:
    """Load per-frame keypoints; frames with < 12 keypoints are stored as None."""
    with open(path) as f:
        data = json.load(f)
    result = []
    for obj in data:
        if obj and len(obj) >= 12:
            result.append(Keypoints.from_json(obj))
        else:
            result.append(None)
    return result


def _load_fixed_keypoints(path: str, n_frames: int) -> list:
    """Create a fixed-keypoints list (same Keypoints object repeated)."""
    with open(path) as f:
        raw = json.load(f)
    kp = Keypoints([Keypoint(id=i, xy=tuple(float(c) for c in v)) for i, v in enumerate(raw)])
    return [kp] * n_frames


def _load_players_detections(path: str) -> list:
    with open(path) as f:
        data = json.load(f)
    return [Players.from_json(obj) for obj in data]


# ---------------------------------------------------------------------------
# Analytics helpers
# ---------------------------------------------------------------------------

def _project_ball_to_meters(
    ball_detections: list,
    keypoints_detections: list,
    projected_court: ProjectedCourt,
) -> list:
    """
    Return a list of (x_m, y_m) in court-centred metres for each frame.
    Frames where ball is invisible or projection fails → None.
    """
    positions = []
    for ball, keypoints in zip(ball_detections, keypoints_detections):
        if ball.visibility == 0 or keypoints is None:
            positions.append(None)
            continue
        try:
            H = projected_court.homography_matrix(keypoints)
            proj = projected_court.project_point(ball.asint(), H)
            metres = projected_court.court_keypoints.shift_point_origin(proj, "meters")
            positions.append(metres)
        except Exception:
            positions.append(None)
    return positions


def _build_data_analytics(
    players_detections: list,
    keypoints_detections: list,
    projected_court: ProjectedCourt,
) -> DataAnalytics:
    """
    Reconstruct DataAnalytics by projecting each player's feet to court metres.
    Follows the same accumulation pattern as TrackingRunner.draw_and_collect_data().
    """
    data_analytics = DataAnalytics()
    H = None
    n = len(players_detections)

    for i in range(n):
        keypoints = keypoints_detections[i]
        players = players_detections[i]

        # Recompute homography whenever new keypoints are available
        if keypoints is not None:
            try:
                H = projected_court.homography_matrix(keypoints)
            except Exception:
                H = None

        if H is not None and players is not None:
            for player in players.players:
                try:
                    proj = projected_court.project_point(player.feet, H)
                    metres = projected_court.court_keypoints.shift_point_origin(
                        tuple(float(v) for v in proj), "meters"
                    )
                    data_analytics.add_player_position(
                        id=player.id,
                        position=(float(metres[0]), float(metres[1])),
                    )
                except Exception:
                    pass

        # Advance to next frame (mirrors runner behaviour)
        data_analytics.step(1)

    # Remove the trailing empty frame that the final step() creates
    data_analytics.frames = data_analytics.frames[:-1]

    return data_analytics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== POC Padel Analytics ===\n")

    OUTPUT_DIR.mkdir(exist_ok=True)

    # ---- Video info --------------------------------------------------------
    video_info = sv.VideoInfo.from_video_path(INPUT_VIDEO_PATH)
    fps = video_info.fps
    print(f"Video  : {INPUT_VIDEO_PATH}")
    print(f"Frames : {video_info.total_frames}  FPS: {fps:.1f}")

    # ---- Load cached detections -------------------------------------------
    assert BALL_TRACKER_LOAD_PATH, (
        "BALL_TRACKER_LOAD_PATH must be set in config.py. Run main.py first."
    )
    assert PLAYERS_TRACKER_LOAD_PATH, (
        "PLAYERS_TRACKER_LOAD_PATH must be set in config.py. Run main.py first."
    )
    assert KEYPOINTS_TRACKER_LOAD_PATH or FIXED_COURT_KEYPOINTS_LOAD_PATH, (
        "Either KEYPOINTS_TRACKER_LOAD_PATH or FIXED_COURT_KEYPOINTS_LOAD_PATH "
        "must be set in config.py. Run main.py first."
    )

    print("\nLoading cached detections …")
    ball_detections = _load_ball_detections(BALL_TRACKER_LOAD_PATH)
    n = len(ball_detections)

    if KEYPOINTS_TRACKER_LOAD_PATH:
        keypoints_detections = _load_keypoints_detections(KEYPOINTS_TRACKER_LOAD_PATH)
    else:
        keypoints_detections = _load_fixed_keypoints(FIXED_COURT_KEYPOINTS_LOAD_PATH, n)

    players_detections = _load_players_detections(PLAYERS_TRACKER_LOAD_PATH)

    # Align all lists to the shortest one
    n = min(n, len(keypoints_detections), len(players_detections))
    ball_detections = ball_detections[:n]
    keypoints_detections = keypoints_detections[:n]
    players_detections = players_detections[:n]
    print(f"Frames loaded: {n}")

    # ---- Projected court setup --------------------------------------------
    projected_court = ProjectedCourt(video_info)

    # ---- Project ball to court metres -------------------------------------
    print("Projecting ball positions to court coordinates …")
    ball_positions_m = _project_ball_to_meters(
        ball_detections, keypoints_detections, projected_court
    )
    n_proj = sum(1 for p in ball_positions_m if p is not None)
    print(f"Ball projected: {n_proj}/{n} frames")

    # ---- Rally detection --------------------------------------------------
    print("\nDetecting rallies …")
    detector = RallyDetector()
    rallies = detector.detect(ball_detections, fps, ball_positions_m)
    print(f"Rallies found: {len(rallies)}")

    rally_dicts = [r.to_dict() for r in rallies]
    for i, d in enumerate(rally_dicts):
        print(
            f"  [{i+1:2d}] {d['start_time']:6.1f}s – {d['end_time']:6.1f}s  "
            f"({d['duration_s']:.1f}s, {d['shot_count']} shots, "
            f"max {d['max_shot_speed_kmh']} km/h)"
        )

    summary_path = OUTPUT_DIR / "rally_summary.json"
    with open(summary_path, "w") as f:
        json.dump(rally_dicts, f, indent=2)
    print(f"\nRally summary → {summary_path}")

    # ---- Best rally -------------------------------------------------------
    if rallies:
        best = max(rallies, key=lambda r: (r.shot_count, r.max_shot_speed_kmh or 0))
        best_path = OUTPUT_DIR / "best_rally.json"
        with open(best_path, "w") as f:
            json.dump(best.to_dict(), f, indent=2)
        idx = rallies.index(best) + 1
        print(
            f"Best rally  : #{idx}  "
            f"{best.duration:.1f}s, {best.shot_count} shots  → {best_path}"
        )

    # ---- DataAnalytics for heatmaps --------------------------------------
    print("\nBuilding player position data …")
    data_analytics = _build_data_analytics(
        players_detections, keypoints_detections, projected_court
    )
    print(f"DataAnalytics: {len(data_analytics.datapoints)} datapoints")

    # ---- Heatmaps ---------------------------------------------------------
    print("Generating player heatmaps …")
    generator = HeatmapGenerator()
    heatmaps = generator.generate(data_analytics)
    for hm in heatmaps:
        path = str(OUTPUT_DIR / f"heatmap_player{hm.player_id}.png")
        generator.save_as_png(hm, path)
        print(f"  Player {hm.player_id} → {path}")

    # ---- Match summary ----------------------------------------------------
    print("\n=== Match Summary ===")
    print(f"Total rallies       : {len(rallies)}")
    if rallies:
        total_rally_time = sum(r.duration for r in rallies)
        total_shots = sum(r.shot_count for r in rallies)
        all_speeds = [
            s.speed_kmh
            for r in rallies
            for s in r.shots
            if s.speed_kmh is not None
        ]
        longest = max(rallies, key=lambda r: r.duration)
        print(f"Total rally time    : {total_rally_time:.1f}s")
        print(f"Total shots         : {total_shots}")
        if all_speeds:
            print(f"Max shot speed      : {max(all_speeds):.1f} km/h")
            print(f"Mean shot speed     : {sum(all_speeds)/len(all_speeds):.1f} km/h")
        print(
            f"Longest rally       : #{rallies.index(longest)+1}  "
            f"{longest.start_time:.1f}s – {longest.end_time:.1f}s  "
            f"({longest.duration:.1f}s)"
        )

    print(f"\nAll outputs saved in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
