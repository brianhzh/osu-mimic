import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import hashlib
from datetime import datetime, timezone
from osrparse import Replay, GameMode, Mod
from osrparse.utils import ReplayEventOsu, Key

from app.core.parser import Parser, calculate_slider_path
from app.models.aim_model import AimGRU

TIMESTEP_MS = 16
TARGET_WINDOW_MS = 800


class AimTarget:
    def __init__(self, x, y, time, is_slider=False):
        self.x = x
        self.y = y
        self.time = time
        self.is_slider = is_slider


def get_beat_length_at_time(timing_points, time):
    beat_length = 500
    for tp in timing_points:
        if tp['time'] <= time:
            if tp['uninherited'] and tp['beat_length'] is not None:
                beat_length = tp['beat_length']
        else:
            break
    return beat_length


def expand_hit_objects_to_targets(beatmap):
    targets = []
    tick_rate = beatmap.difficulty.get('SliderTickRate', 1.0)

    for obj in beatmap.hit_objects:
        if obj.is_circle():
            targets.append(AimTarget(obj.x, obj.y, obj.time, is_slider=False))

        elif obj.is_slider():
            targets.append(AimTarget(obj.x, obj.y, obj.time, is_slider=True))

            if obj.end_time and obj.end_time > obj.time:
                duration = obj.end_time - obj.time
                duration_per_slide = duration / obj.slides
                beat_length = get_beat_length_at_time(beatmap.timing_points, obj.time)
                sample_interval = beat_length / tick_rate / 8
                base_path = calculate_slider_path(obj, num_points=100)

                for slide in range(obj.slides):
                    slide_start_time = obj.time + (slide * duration_per_slide)
                    slide_end_time = slide_start_time + duration_per_slide
                    path = base_path if slide % 2 == 0 else list(reversed(base_path))

                    sample_time = sample_interval
                    while sample_time < duration_per_slide:
                        progress = sample_time / duration_per_slide
                        path_idx = int(progress * (len(path) - 1))
                        px, py = path[path_idx]
                        targets.append(AimTarget(px, py, slide_start_time + sample_time, is_slider=True))
                        sample_time += sample_interval

                    targets.append(AimTarget(path[-1][0], path[-1][1], slide_end_time, is_slider=True))

    targets.sort(key=lambda t: t.time)
    return targets


def find_next_target(current_time, targets, target_idx):
    while target_idx < len(targets) and targets[target_idx].time <= current_time:
        target_idx += 1

    if target_idx < len(targets) and (targets[target_idx].time - current_time) <= TARGET_WINDOW_MS:
        return targets[target_idx], target_idx

    return None, target_idx


def generate_frames(model, device, beatmap, targets):
    if not targets:
        return []

    start_time = max(0, targets[0].time - 2000)
    end_time = targets[-1].time + 1000

    model.eval()
    hidden = model.init_hidden(1, device)

    cursor_x = 0.5
    cursor_y = 0.5
    position_history = [(0.5, 0.5)] * 4

    frames = []
    current_time = start_time
    target_idx = 0

    with torch.no_grad():
        while current_time <= end_time:
            target, target_idx = find_next_target(current_time, targets, target_idx)

            if target is None:
                frames.append({'time': current_time, 'x': cursor_x, 'y': cursor_y})
                current_time += TIMESTEP_MS
                continue

            target_x = target.x / 512.0
            target_y = target.y / 384.0
            time_to_target = (target.time - current_time) / 1000.0

            dx = target_x - cursor_x
            dy = target_y - cursor_y
            distance = np.sqrt(dx**2 + dy**2)
            angle = np.arctan2(dy, dx)

            input_feat = torch.tensor([[
                cursor_x, cursor_y,
                target_x, target_y,
                time_to_target, distance,
                np.sin(angle), np.cos(angle),
                cursor_x - position_history[0][0], cursor_y - position_history[0][1],
                position_history[0][0] - position_history[1][0], position_history[0][1] - position_history[1][1],
                position_history[1][0] - position_history[2][0], position_history[1][1] - position_history[2][1],
                position_history[2][0] - position_history[3][0], position_history[2][1] - position_history[3][1],
                1.4 if target.is_slider else 0.0
            ]], dtype=torch.float32).unsqueeze(0).to(device)

            pred, hidden = model(input_feat, hidden)
            cursor_x = np.clip(cursor_x + float(pred[0, 0, 0]), 0, 1)
            cursor_y = np.clip(cursor_y + float(pred[0, 0, 1]), 0, 1)
            position_history = [(cursor_x, cursor_y)] + position_history[:3]

            frames.append({'time': current_time, 'x': cursor_x, 'y': cursor_y})
            current_time += TIMESTEP_MS

    return frames


def frames_to_replay_events(frames):
    events = []
    prev_time = 0

    for frame in frames:
        time_delta = int(frame['time']) - prev_time
        events.append(ReplayEventOsu(
            time_delta=time_delta,
            x=frame['x'] * 512.0,
            y=frame['y'] * 384.0,
            keys=Key(0)
        ))
        prev_time = int(frame['time'])

    events.append(ReplayEventOsu(time_delta=-12345, x=0, y=0, keys=Key(0)))
    return events


def get_beatmap_hash(filepath):
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = os.path.join(os.path.dirname(__file__), '..', 'aim_model_best.pt')
    if not os.path.exists(model_path):
        print("model not found")
        return

    model = AimGRU(input_size=17, hidden_size=128, num_layers=2, output_size=2, dropout=0.2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    beatmap_path = input("beatmap path: ").strip('"').strip("'")
    if not os.path.exists(beatmap_path) or not beatmap_path.endswith('.osu'):
        print("invalid path")
        return

    beatmap = Parser(beatmap_path).parse()
    if not beatmap.hit_objects:
        print("no hit objects found")
        return

    targets = expand_hit_objects_to_targets(beatmap)
    print(f"loaded {len(beatmap.hit_objects)} objects, {len(targets)} targets")
    print("generating replay...")

    frames = generate_frames(model, device, beatmap, targets)
    events = frames_to_replay_events(frames)

    beatmap_hash = get_beatmap_hash(beatmap_path)

    artist = beatmap.metadata.get('Artist', 'Unknown').strip()
    title = beatmap.metadata.get('Title', 'Unknown').strip()
    version = beatmap.metadata.get('Version', 'Unknown').strip()

    replay = Replay(
        mode=GameMode.STD,
        game_version=20240123,
        beatmap_hash=beatmap_hash,
        username="osu!mimic",
        replay_hash="",
        count_300=0,
        count_100=0,
        count_50=0,
        count_geki=0,
        count_katu=0,
        count_miss=0,
        score=0,
        max_combo=0,
        perfect=False,
        mods=Mod.Relax,
        life_bar_graph=None,
        timestamp=datetime.now(timezone.utc),
        replay_data=events,
        replay_id=0,
        rng_seed=None,
    )

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'generated_replays')
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{artist} - {title} [{version}].osr"
    for char in '<>:"/\\|?*':
        filename = filename.replace(char, '_')

    output_path = os.path.join(output_dir, filename)
    replay.write_path(output_path)
    print(f"saved: {output_path}")
    print(f"frames: {len(events)}")


if __name__ == "__main__":
    main()
