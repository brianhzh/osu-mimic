import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import os
import pickle
from pathlib import Path

from app.core.parser import Parser
from app.core.replay_parser import ReplayParser

INPUT_FEATURES = [
    'cursor_x', 'cursor_y',
    'target_x', 'target_y',
    'time_to_target',
    'distance_to_target',
    'angle_sin', 'angle_cos',
]

OUTPUT_FEATURES = ['cursor_x', 'cursor_y']

SEQUENCE_LENGTH = 512
TIMESTEP_MS = 16
TARGET_WINDOW_MS = 800
SEQUENCE_STRIDE = 256


class OsuSequenceDataset(Dataset):

    def __init__(
        self,
        beatmap_paths: List[str],
        replay_paths: List[str],
        sequence_length: int = SEQUENCE_LENGTH,
        sequence_stride: int = SEQUENCE_STRIDE,
        target_window_ms: int = TARGET_WINDOW_MS,
        cache_dir: str = '.data',
        training_noise: float = 0.0
    ):
        self.sequence_length = sequence_length
        self.sequence_stride = sequence_stride
        self.target_window_ms = target_window_ms
        self.training_noise = training_noise
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.pairs = list(zip(beatmap_paths, replay_paths))
        self.beatmap_ids = self._get_beatmap_ids()
        self.sequences = self._build_all_sequences()

        print(f"dataset created: {len(self.sequences)} sequences from {len(self.pairs)} beatmap/replay pairs")

    def _get_beatmap_ids(self) -> List[str]:
        ids = []
        for beatmap_path, _ in self.pairs:
            beatmap_id = os.path.basename(beatmap_path)
            ids.append(beatmap_id)
        return ids

    def _build_all_sequences(self) -> List[Dict[str, np.ndarray]]:
        all_sequences = []

        for i, (beatmap_path, replay_path) in enumerate(self.pairs):
            print(f"[{i+1}/{len(self.pairs)}] processing {os.path.basename(beatmap_path)}")

            cache_file = self.cache_dir / f"seq_{os.path.basename(beatmap_path)}.pkl"

            if cache_file.exists():
                print(f"  loading from cache...")
                with open(cache_file, 'rb') as f:
                    sequences = pickle.load(f)
            else:
                sequences = self._build_sequences(beatmap_path, replay_path)

                with open(cache_file, 'wb') as f:
                    pickle.dump(sequences, f)

                print(f"  created {len(sequences)} sequences")

            all_sequences.extend(sequences)

        return all_sequences

    def _build_sequences(
        self,
        beatmap_path: str,
        replay_path: str
    ) -> List[Dict[str, np.ndarray]]:
        # parse beatmap and replay
        beatmap = Parser(beatmap_path).parse()
        replay = ReplayParser(replay_path)

        trajectory = replay.get_aiming_trajectory()

        if len(trajectory) == 0:
            print(f"  warning: empty trajectory")
            return []

        input_frames = []
        output_frames = []

        # track last 4 positions for velocity history
        position_history = [(0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5)]

        for i, frame in enumerate(trajectory):
            target = self._find_next_target(frame['time'], beatmap.hit_objects)

            if target is None:
                continue

            cursor_x = frame['x']
            cursor_y = frame['y']

            # calculate 4-frame velocity history
            vel_0_x = cursor_x - position_history[0][0]
            vel_0_y = cursor_y - position_history[0][1]
            vel_1_x = position_history[0][0] - position_history[1][0]
            vel_1_y = position_history[0][1] - position_history[1][1]
            vel_2_x = position_history[1][0] - position_history[2][0]
            vel_2_y = position_history[1][1] - position_history[2][1]
            vel_3_x = position_history[2][0] - position_history[3][0]
            vel_3_y = position_history[2][1] - position_history[3][1]

            target_x = target.x / 512.0
            target_y = target.y / 384.0
            time_to_target = (target.time - frame['time']) / 1000.0

            dx = target_x - cursor_x
            dy = target_y - cursor_y
            distance = np.sqrt(dx**2 + dy**2)
            angle = np.arctan2(dy, dx)

            input_feat = np.array([
                cursor_x, cursor_y,
                target_x, target_y,
                time_to_target, distance,
                np.sin(angle), np.cos(angle),
                vel_0_x, vel_0_y,
                vel_1_x, vel_1_y,
                vel_2_x, vel_2_y,
                vel_3_x, vel_3_y
            ], dtype=np.float32)

            position_history.pop()
            position_history.insert(0, (cursor_x, cursor_y))

            # output delta position
            if i + 1 < len(trajectory):
                next_frame = trajectory[i + 1]
                next_x = next_frame['x']
                next_y = next_frame['y']
                delta_x = next_x - cursor_x
                delta_y = next_y - cursor_y
                output_feat = np.array([delta_x, delta_y], dtype=np.float32)
            else:
                output_feat = np.array([0.0, 0.0], dtype=np.float32)

            input_frames.append(input_feat)
            output_frames.append(output_feat)

        sequences = []
        num_frames = len(input_frames)

        if num_frames < self.sequence_length:
            return sequences

        start_idx = 0
        while start_idx + self.sequence_length <= num_frames:
            end_idx = start_idx + self.sequence_length

            seq_input = np.stack(input_frames[start_idx:end_idx])
            seq_output = np.stack(output_frames[start_idx:end_idx])

            sequences.append({
                'input': seq_input,
                'output': seq_output,
                'beatmap_id': os.path.basename(beatmap_path)
            })

            start_idx += self.sequence_stride

        return sequences

    def _find_next_target(self, current_time: int, hit_objects: List) -> Optional[Any]:
        for obj in hit_objects:
            time_diff = obj.time - current_time
            if 0 < time_diff <= self.target_window_ms:
                return obj
        return None

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx]

        input_tensor = torch.from_numpy(seq['input'].copy())
        output_tensor = torch.from_numpy(seq['output'])

        # add noise to cursor positions during training
        if self.training_noise > 0:
            noise = torch.randn(input_tensor.shape[0], 2) * self.training_noise
            input_tensor[:, 0:2] += noise

        return input_tensor, output_tensor

    def get_beatmap_id(self, idx: int) -> str:
        return self.sequences[idx]['beatmap_id']


def train_val_split_by_map(
    dataset: OsuSequenceDataset,
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[List[int], List[int]]:
    np.random.seed(seed)
    unique_beatmaps = list(set(dataset.beatmap_ids))
    np.random.shuffle(unique_beatmaps)
    split_idx = int(len(unique_beatmaps) * (1 - val_ratio))

    train_beatmaps = set(unique_beatmaps[:split_idx])
    val_beatmaps = set(unique_beatmaps[split_idx:])

    print(f"\ntrain/val split:")
    print(f"  train maps: {len(train_beatmaps)}")
    print(f"  val maps: {len(val_beatmaps)}")

    train_indices = []
    val_indices = []

    for i in range(len(dataset)):
        beatmap_id = dataset.get_beatmap_id(i)
        if beatmap_id in train_beatmaps:
            train_indices.append(i)
        else:
            val_indices.append(i)

    print(f"  train sequences: {len(train_indices)}")
    print(f"  val sequences: {len(val_indices)}")

    return train_indices, val_indices


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    inputs, outputs = zip(*batch)
    return torch.stack(inputs), torch.stack(outputs)
