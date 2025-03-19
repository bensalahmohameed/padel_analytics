from enum import Enum
from dataclasses import dataclass
import numpy as np
import cv2
import supervision as sv
from typing import Optional, Tuple

from trackers import Ball, Player, Keypoints
from trackers.velocity_in_time import VelocityVector
from analytics.projected_court import ProjectedCourt

class ImpactType(Enum):
    FLOOR = "floor"
    RACKET = "racket"

@dataclass
class BallVelocityData:
    position_t0: Tuple[float, float]
    position_t1: Tuple[float, float]
    position_t0_proj: Tuple[float, float]
    position_t1_proj: Tuple[float, float]
    velocity_vector: VelocityVector

    def draw_velocity(self, frame: np.ndarray) -> np.ndarray:
        return self.velocity_vector.draw_velocity_vector(frame)

class BallVelocityEstimator:
    def __init__(
        self,
        source_video_fps: float,
        players_detections: list[list[Player]],
        ball_detections: list[list[Ball]],
        keypoints_detections: list[list[Keypoints]],
        video_info: Optional[sv.VideoInfo] = None,
    ):
        self.fps = source_video_fps
        self.players_detections = players_detections
        self.ball_detections = ball_detections
        self.keypoints_detections = keypoints_detections
        
        # Initialize ProjectedCourt with video_info
        if video_info is None:
            # Create a default VideoInfo if none provided
            video_info = sv.VideoInfo(
                width=1920,  # Default width
                height=1080,  # Default height
                fps=source_video_fps,
                total_frames=0,  # Not needed for projection
            )
        self.projected_court = ProjectedCourt(video_info)

    def estimate_velocity(
        self,
        frame_index_t0: int,
        frame_index_t1: int,
        impact_type: ImpactType,
        get_Vz: bool = False,
    ) -> Tuple[BallVelocityData, VelocityVector]:
        """
        Estimate ball velocity between two frames
        
        Args:
            frame_index_t0: First frame index
            frame_index_t1: Second frame index
            impact_type: Type of impact (floor or racket)
            get_Vz: Whether to consider vertical velocity
            
        Returns:
            Tuple of (BallVelocityData, VelocityVector)
        """
        # Get ball positions at both frames
        ball_t0 = self.ball_detections[frame_index_t0][0]  # Assuming one ball per frame
        ball_t1 = self.ball_detections[frame_index_t1][0]
        
        # Get keypoints for projection
        keypoints_t0 = self.keypoints_detections[frame_index_t0][0]
        keypoints_t1 = self.keypoints_detections[frame_index_t1][0]
        
        # Calculate homography matrices
        H_t0 = self.projected_court.homography_matrix(keypoints_t0)
        H_t1 = self.projected_court.homography_matrix(keypoints_t1)
        
        # Convert float coordinates to integers for projection
        ball_t0_xy_int = (int(ball_t0.xy[0]), int(ball_t0.xy[1]))
        ball_t1_xy_int = (int(ball_t1.xy[0]), int(ball_t1.xy[1]))
        
        # Project ball positions to court plane
        position_t0_proj = self.projected_court.project_point(ball_t0_xy_int, H_t0)
        position_t1_proj = self.projected_court.project_point(ball_t1_xy_int, H_t1)
        
        # Calculate time difference
        delta_time = (frame_index_t1 - frame_index_t0) / self.fps
        
        # Create velocity vector
        velocity_vector = VelocityVector(
            r0=position_t0_proj,
            r1=position_t1_proj
        )
        
        # Create velocity data object
        velocity_data = BallVelocityData(
            position_t0=ball_t0.xy,
            position_t1=ball_t1.xy,
            position_t0_proj=position_t0_proj,
            position_t1_proj=position_t1_proj,
            velocity_vector=velocity_vector
        )
        
        return velocity_data, velocity_vector 