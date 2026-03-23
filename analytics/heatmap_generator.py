"""Player heatmap generation from position data."""

from dataclasses import dataclass
import numpy as np
import cv2

from analytics.data_analytics import DataAnalytics
from constants import BASE_LINE, SIDE_LINE


@dataclass
class PlayerHeatmap:
    player_id: int
    heatmap: np.ndarray  # 2D float32 array, values in [0, 1]


class HeatmapGenerator:
    """
    Generates 2D court heatmaps per player from position data.

    The court coordinate system (in meters, centered at origin):
        x: [-BASE_LINE/2, +BASE_LINE/2]  →  [-5, +5]
        y: [-SIDE_LINE/2, +SIDE_LINE/2]  →  [-10, +10]
    """

    GRID_WIDTH: int = 100   # pixels wide
    GRID_HEIGHT: int = 200  # pixels tall (2:1 aspect ratio matches 10x20m court)
    BLUR_KERNEL: int = 15   # Gaussian blur kernel size (must be odd)

    X_MIN: float = -BASE_LINE / 2   # -5 m
    X_MAX: float = BASE_LINE / 2    # +5 m
    Y_MIN: float = -SIDE_LINE / 2   # -10 m
    Y_MAX: float = SIDE_LINE / 2    # +10 m

    def generate(self, data_analytics: DataAnalytics) -> list:  # list[PlayerHeatmap]
        """
        Generate per-player heatmaps from collected tracking data.

        Returns:
            list of PlayerHeatmap, one for each player ID in (1, 2, 3, 4)
        """
        heatmaps = []
        for player_id in (1, 2, 3, 4):
            positions = self._get_player_positions(data_analytics, player_id)
            heatmap = self._build_heatmap(positions)
            heatmaps.append(PlayerHeatmap(player_id=player_id, heatmap=heatmap))
        return heatmaps

    def _get_player_positions(
        self,
        data_analytics: DataAnalytics,
        player_id: int,
    ) -> list:  # list[tuple[float, float]]
        positions = []
        for dp in data_analytics.datapoints:
            if dp.players_position is None:
                continue
            for pp in dp.players_position:
                if pp.id == player_id:
                    positions.append(pp.position)
        return positions

    def _build_heatmap(self, positions: list) -> np.ndarray:
        """Build a smoothed, normalized 2D heatmap from a list of (x, y) positions."""
        grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=np.float32)

        x_range = self.X_MAX - self.X_MIN
        y_range = self.Y_MAX - self.Y_MIN

        for x, y in positions:
            col = int((x - self.X_MIN) / x_range * self.GRID_WIDTH)
            row = int((y - self.Y_MIN) / y_range * self.GRID_HEIGHT)
            col = int(np.clip(col, 0, self.GRID_WIDTH - 1))
            row = int(np.clip(row, 0, self.GRID_HEIGHT - 1))
            grid[row, col] += 1.0

        grid = cv2.GaussianBlur(grid, (self.BLUR_KERNEL, self.BLUR_KERNEL), 0)

        max_val = grid.max()
        if max_val > 0:
            grid /= max_val

        return grid

    def save_as_png(
        self,
        heatmap: PlayerHeatmap,
        output_path: str,
        colormap: int = cv2.COLORMAP_JET,
    ) -> None:
        """
        Save a heatmap as a colorized PNG with court overlay.

        Parameters:
            heatmap: PlayerHeatmap to save
            output_path: destination file path
            colormap: OpenCV colormap constant
        """
        heatmap_uint8 = (heatmap.heatmap * 255).astype(np.uint8)
        colored = cv2.applyColorMap(heatmap_uint8, colormap)

        h, w = colored.shape[:2]

        # Draw net line at vertical center
        net_y = h // 2
        cv2.line(colored, (0, net_y), (w, net_y), (255, 255, 255), 2)

        # Draw court boundary
        cv2.rectangle(colored, (0, 0), (w - 1, h - 1), (255, 255, 255), 1)

        # Label
        cv2.putText(
            colored,
            f"Player {heatmap.player_id}",
            (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.imwrite(output_path, colored)
