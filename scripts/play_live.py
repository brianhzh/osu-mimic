import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import time
import ctypes
import keyboard
from pathlib import Path

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except:
    pass

from app.core.parser import Parser, calculate_slider_path
from app.models.aim_model import AimGRU


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


PUL = ctypes.POINTER(ctypes.c_ulong)

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long), ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong), ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("mi", MouseInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("ii", Input_I)]


class LivePlayer:
    def __init__(self, model, device, screen_width, screen_height):
        self.model = model
        self.device = device
        self.screen_width = screen_width
        self.screen_height = screen_height

        playfield_aspect = 512.0 / 384.0
        if screen_width / screen_height > playfield_aspect:
            self.playfield_h = int(screen_height * 0.8)
            self.playfield_w = int(self.playfield_h * playfield_aspect)
        else:
            self.playfield_w = int(screen_width * 0.8)
            self.playfield_h = int(self.playfield_w / playfield_aspect)

        self.playfield_x = (screen_width - self.playfield_w) // 2
        self.playfield_y = (screen_height - self.playfield_h) // 2

        self.cursor_x = 0.5
        self.cursor_y = 0.5
        self.hidden = None

    def move_mouse(self, norm_x, norm_y):
        screen_x = self.playfield_x + (norm_x * self.playfield_w)
        screen_y = self.playfield_y + (norm_y * self.playfield_h)
        abs_x = int(screen_x * 65535 / self.screen_width)
        abs_y = int(screen_y * 65535 / self.screen_height)

        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.mi = MouseInput(abs_x, abs_y, 0, 0x0001 | 0x8000, 0, ctypes.pointer(extra))
        command = Input(ctypes.c_ulong(0), ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(command), ctypes.sizeof(command))

    def play_beatmap(self, beatmap_path, audio_offset_ms=0):
        beatmap = Parser(beatmap_path).parse()
        if len(beatmap.hit_objects) == 0:
            print("no hit objects found")
            return

        targets = expand_hit_objects_to_targets(beatmap)
        print(f"loaded {len(beatmap.hit_objects)} objects, {len(targets)} targets")
        print("press ENTER to start, ESC to stop")

        while not keyboard.is_pressed('enter'):
            time.sleep(0.01)

        start_time = time.time() * 1000 - audio_offset_ms
        self.hidden = self.model.init_hidden(1, self.device)
        position_history = [(0.5, 0.5)] * 4

        with torch.no_grad():
            for target in targets:
                if keyboard.is_pressed('esc'):
                    break

                while True:
                    current_time = time.time() * 1000 - start_time
                    time_to_target = (target.time - current_time) / 1000.0
                    if time_to_target <= 0:
                        break
                    if time_to_target > 0.8:
                        time.sleep(0.008)
                        continue

                    target_x = target.x / 512.0
                    target_y = target.y / 384.0
                    dx, dy = target_x - self.cursor_x, target_y - self.cursor_y
                    distance = np.sqrt(dx**2 + dy**2)
                    angle = np.arctan2(dy, dx)

                    input_feat = torch.tensor([[
                        self.cursor_x, self.cursor_y,
                        target_x, target_y,
                        time_to_target, distance,
                        np.sin(angle), np.cos(angle),
                        self.cursor_x - position_history[0][0], self.cursor_y - position_history[0][1],
                        position_history[0][0] - position_history[1][0], position_history[0][1] - position_history[1][1],
                        position_history[1][0] - position_history[2][0], position_history[1][1] - position_history[2][1],
                        position_history[2][0] - position_history[3][0], position_history[2][1] - position_history[3][1],
                        1.4 if target.is_slider else 0.0
                    ]], dtype=torch.float32).unsqueeze(0).to(self.device)

                    pred, self.hidden = self.model(input_feat, self.hidden)
                    self.cursor_x = np.clip(self.cursor_x + float(pred[0, 0, 0]), 0, 1)
                    self.cursor_y = np.clip(self.cursor_y + float(pred[0, 0, 1]), 0, 1)

                    position_history = [(self.cursor_x, self.cursor_y)] + position_history[:3]
                    self.move_mouse(self.cursor_x, self.cursor_y)
                    time.sleep(0.008)

        print("finished")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists('aim_model_best.pt'):
        print("model not found")
        return

    model = AimGRU(input_size=17, hidden_size=128, num_layers=2, output_size=2, dropout=0.2)
    model.load_state_dict(torch.load('aim_model_best.pt', map_location=device))
    model.to(device)
    model.eval()

    user32 = ctypes.windll.user32
    screen_w, screen_h = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    print(f"screen: {screen_w}x{screen_h}")
    print("disable raw input in osu! settings")

    beatmap_path = input("beatmap path: ").strip('"').strip("'")
    if not os.path.exists(beatmap_path) or not beatmap_path.endswith('.osu'):
        print("invalid path")
        return

    offset = input("audio offset ms (0): ").strip()
    player = LivePlayer(model, device, screen_w, screen_h)
    player.play_beatmap(beatmap_path, int(offset) if offset else 0)


if __name__ == "__main__":
    main()
