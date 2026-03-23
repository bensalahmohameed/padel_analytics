"""Rally detection from ball tracking data."""

from dataclasses import dataclass, field
from typing import Optional

from trackers.ball_tracker.ball_tracker import Ball
from trackers.velocity_in_time import VelocityVector


@dataclass
class Shot:
    frame_index: int
    timestamp: float  # seconds
    speed_kmh: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "timestamp": round(self.timestamp, 3),
            "speed_kmh": round(self.speed_kmh, 1) if self.speed_kmh is not None else None,
        }


@dataclass
class Rally:
    start_frame: int
    end_frame: int
    start_time: float  # seconds
    end_time: float    # seconds
    shots: list = field(default_factory=list)  # list[Shot]

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def shot_count(self) -> int:
        return len(self.shots)

    @property
    def max_shot_speed_kmh(self) -> Optional[float]:
        speeds = [s.speed_kmh for s in self.shots if s.speed_kmh is not None]
        return max(speeds) if speeds else None

    def to_dict(self) -> dict:
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3),
            "duration_s": round(self.duration, 3),
            "shot_count": self.shot_count,
            "max_shot_speed_kmh": round(self.max_shot_speed_kmh, 1) if self.max_shot_speed_kmh is not None else None,
            "shots": [s.to_dict() for s in self.shots],
        }


class RallyDetector:
    """
    Detects rallies and shots from ball tracking data.

    A rally is a continuous period of ball activity. Gaps longer than
    INVISIBLE_THRESHOLD frames are treated as rally boundaries.
    Shots are detected as direction changes in the ball trajectory.
    """

    # Max consecutive invisible frames that can be part of the same rally
    INVISIBLE_THRESHOLD: int = 25
    # Minimum visible frames for a segment to be considered a valid rally
    MIN_RALLY_FRAMES: int = 30
    # Minimum angle change (degrees) to count as a shot direction change
    MIN_ANGLE_CHANGE: float = 90.0

    def detect(
        self,
        ball_detections: list,  # list[Ball]
        fps: float,
        ball_positions_meters: Optional[list] = None,  # list[Optional[tuple[float, float]]]
    ) -> list:  # list[Rally]
        """
        Detect rallies from ball detections.

        Parameters:
            ball_detections: flat list of Ball objects, one per frame
            fps: video fps
            ball_positions_meters: projected ball positions in court meters,
                one per frame; None entries mean ball not projected for that frame

        Returns:
            list of Rally objects sorted by start frame
        """
        visible_ranges = self._find_visible_ranges(ball_detections)
        merged = self._merge_ranges(visible_ranges)

        rallies = []
        for start, end in merged:
            if end - start < self.MIN_RALLY_FRAMES:
                continue

            shots = self._detect_shots(
                ball_detections=ball_detections,
                fps=fps,
                start=start,
                end=end,
                ball_positions_meters=ball_positions_meters,
            )

            rallies.append(Rally(
                start_frame=start,
                end_frame=end,
                start_time=start / fps,
                end_time=end / fps,
                shots=shots,
            ))

        return rallies

    def _find_visible_ranges(self, ball_detections: list) -> list:
        """Find contiguous ranges where ball.visibility == 1."""
        ranges = []
        i = 0
        n = len(ball_detections)
        while i < n:
            if ball_detections[i].visibility == 1:
                j = i
                while j < n and ball_detections[j].visibility == 1:
                    j += 1
                ranges.append((i, j - 1))
                i = j
            else:
                i += 1
        return ranges

    def _merge_ranges(self, ranges: list) -> list:
        """Merge adjacent ranges whose gap is <= INVISIBLE_THRESHOLD."""
        if not ranges:
            return []
        merged = [list(ranges[0])]
        for start, end in ranges[1:]:
            gap = start - merged[-1][1]
            if gap <= self.INVISIBLE_THRESHOLD:
                merged[-1][1] = end
            else:
                merged.append([start, end])
        return [tuple(r) for r in merged]

    def _detect_shots(
        self,
        ball_detections: list,
        fps: float,
        start: int,
        end: int,
        ball_positions_meters: Optional[list],
    ) -> list:  # list[Shot]
        """Detect shots within a rally via velocity direction changes."""

        # Collect frames with visible ball in this rally range
        visible_frames = [
            i for i in range(start, end + 1)
            if ball_detections[i].visibility == 1
        ]

        if len(visible_frames) < 3:
            return []

        # Build velocity vectors between consecutive visible frames
        vectors = []  # list of (f0, f1, VelocityVector)
        for i in range(len(visible_frames) - 1):
            f0 = visible_frames[i]
            f1 = visible_frames[i + 1]

            if ball_positions_meters is not None:
                p0 = ball_positions_meters[f0]
                p1 = ball_positions_meters[f1]
                if p0 is None or p1 is None:
                    continue
            else:
                p0 = ball_detections[f0].xy
                p1 = ball_detections[f1].xy

            vectors.append((f0, f1, VelocityVector(r0=p0, r1=p1)))

        shots = []
        for i in range(len(vectors) - 1):
            f0_a, f1_a, v0 = vectors[i]
            f0_b, f1_b, v1 = vectors[i + 1]
            # f1_a == f0_b: common frame where direction change happens

            if v0.norm < 1e-6 or v1.norm < 1e-6:
                continue

            try:
                angle = v0.angle(v1)
            except (ZeroDivisionError, ValueError):
                continue

            if angle >= self.MIN_ANGLE_CHANGE:
                shot_frame = f1_a  # frame where direction changes
                timestamp = shot_frame / fps

                speed_kmh = None
                if ball_positions_meters is not None:
                    delta_time = (f1_b - f0_b) / fps
                    if delta_time > 0:
                        speed_kmh = (v1.norm / delta_time) * 3.6

                shots.append(Shot(
                    frame_index=shot_frame,
                    timestamp=timestamp,
                    speed_kmh=speed_kmh,
                ))

        return shots
