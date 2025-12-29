from typing import Dict, List, Any, Optional
from osrparse import Replay, Mod

class ReplayParser:
    def __init__(self, replay_path: str, timestep_ms: int = 16) -> None:
        try:
            self.replay = Replay.from_path(replay_path)
        except Exception as e:
            raise ValueError(f"Failed to parse replay file: {e}")

        self.timestep_ms: int = timestep_ms
        self._trajectory: Optional[List[Dict[str, Any]]] = None
        self._resampled: Optional[List[Dict[str, Any]]] = None

    def get_cursor_trajectory(self) -> List[Dict[str, Any]]:
        # coordinates are normalized to [0, 1].
        if self._trajectory is not None:
            return self._trajectory

        if not self.replay.replay_data:
            raise ValueError("Replay has no cursor data")

        trajectory = []
        current_time = 0

        for event in self.replay.replay_data:
            current_time += event.time_delta

            trajectory.append({
                "time": current_time,
                "x": event.x / 512.0,   # normalize
                "y": event.y / 384.0,
                "keys": event.keys
            })

        self._trajectory = trajectory
        return trajectory

    def resample(self) -> List[Dict[str, Any]]:
        # fixed timestep
        if self._resampled is not None:
            return self._resampled

        trajectory = self.get_cursor_trajectory()
        if not trajectory:
            return []

        resampled = []
        idx = 0

        t_start = trajectory[0]["time"]
        t_end = trajectory[-1]["time"]
        t = t_start

        while t <= t_end:
            while idx + 1 < len(trajectory) and trajectory[idx + 1]["time"] <= t:
                idx += 1

            frame = trajectory[idx]

            resampled.append({
                "time": t,
                "x": frame["x"],
                "y": frame["y"],
                "keys": frame["keys"]
            })

            t += self.timestep_ms

        self._resampled = resampled
        return resampled

    def extract_click(self, keys: int) -> int:
        # convert key bitmask to binary
        return int(
            (keys & 1) or
            (keys & 2) or
            (keys & 4) or
            (keys & 8)
        )

    def is_relax(self) -> bool:
        return Mod.Relax in self.replay.mods

    def get_relax_trajectory(self) -> List[Dict[str, Any]]:
        if not self.is_relax():
            raise ValueError("Replay is not played with Relax mod")

        frames = self.resample()
        return [
            {
                "time": f["time"],
                "x": f["x"],
                "y": f["y"]
            }
            for f in frames
        ]

    def get_aiming_trajectory(self) -> List[Dict[str, Any]]:
        frames = self.resample()
        return [
            {
                "time": f["time"],
                "x": f["x"],
                "y": f["y"]
            }
            for f in frames
        ]

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "player_name": self.replay.username,
            "beatmap_hash": self.replay.beatmap_hash,
            "score": self.replay.score,
            "max_combo": self.replay.max_combo,
            "count_300": self.replay.count_300,
            "count_100": self.replay.count_100,
            "count_50": self.replay.count_50,
            "count_miss": self.replay.count_miss,
            "mods": self.replay.mods,
            "perfect": self.replay.perfect,
            "timestamp": self.replay.timestamp
        }
