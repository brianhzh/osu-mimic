import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import time
import ctypes
import keyboard
from pathlib import Path

# set dpi awareness for true screen resolution
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except:
        pass

from app.core.parser import Parser
from app.models.aim_model import AimGRU

PUL = ctypes.POINTER(ctypes.c_ulong)

class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                ("mi", MouseInput),
                ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_ABSOLUTE = 0x8000


class LivePlayer:
    def __init__(self, model, device, screen_width, screen_height):
        self.model = model
        self.device = device
        self.model.eval()

        self.screen_width = screen_width
        self.screen_height = screen_height

        # osu! uses 512x384 playfield
        self.playfield_aspect = 512.0 / 384.0
        screen_aspect = screen_width / screen_height

        if screen_aspect > self.playfield_aspect:
            self.playfield_h = int(screen_height * 0.8)
            self.playfield_w = int(self.playfield_h * self.playfield_aspect)
        else:
            self.playfield_w = int(screen_width * 0.8)
            self.playfield_h = int(self.playfield_w / self.playfield_aspect)

        self.playfield_x = (screen_width - self.playfield_w) // 2
        self.playfield_y = (screen_height - self.playfield_h) // 2

        print(f"  Calculated playfield: ({self.playfield_x}, {self.playfield_y}) size {self.playfield_w}x{self.playfield_h}")

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
        ii_.mi = MouseInput(abs_x, abs_y, 0, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, 0, ctypes.pointer(extra))
        command = Input(ctypes.c_ulong(0), ii_)

        ctypes.windll.user32.SendInput(1, ctypes.pointer(command), ctypes.sizeof(command))

    def play_beatmap(self, beatmap_path, audio_offset_ms=0):
        beatmap = Parser(beatmap_path).parse()
        hit_objects = beatmap.hit_objects

        if len(hit_objects) == 0:
            print("No hit objects found")
            return

        print(f"Loaded beatmap: {Path(beatmap_path).name}")
        print(f"Hit circles: {len(hit_objects)}")
        print(f"First circle at: {hit_objects[0].time}ms")
        print("\nInstructions:")
        print("1. Start the beatmap in osu! with Relax mod")
        print("2. When ready, press F1 or ENTER to begin aiming immediately")
        print("3. Press ESC to stop at any time\n")
        print("Waiting for F1 or ENTER to start...")

        # Wait for F1 or ENTER
        while True:
            if keyboard.is_pressed('f1'):
                print("\nF1 pressed! Starting NOW...")
                time.sleep(0.1)  # Debounce
                break
            if keyboard.is_pressed('enter'):
                print("\nENTER pressed! Starting NOW...")
                break
            time.sleep(0.01)

        start_time = time.time() * 1000 - audio_offset_ms
        print("\nPlaying! Press ESC to stop.")
        self.hidden = self.model.init_hidden(1, self.device)

        current_obj_idx = 0
        timestep_ms = 16
        target_window_ms = 800

        # track 4-frame position history for velocity
        position_history = [
            (self.cursor_x, self.cursor_y),
            (self.cursor_x, self.cursor_y),
            (self.cursor_x, self.cursor_y),
            (self.cursor_x, self.cursor_y)
        ]

        print("Playing! Press ESC to stop.")

        with torch.no_grad():
            while current_obj_idx < len(hit_objects):
                if keyboard.is_pressed('esc'):
                    print("\nStopped by user")
                    break

                current_time = time.time() * 1000 - start_time

                target = None
                for i in range(current_obj_idx, len(hit_objects)):
                    obj = hit_objects[i]
                    time_diff = obj.time - current_time

                    if time_diff < -1000:
                        current_obj_idx = i + 1
                        continue

                    if 0 < time_diff <= target_window_ms:
                        target = obj
                        break

                    if time_diff > target_window_ms:
                        break

                if target is None:
                    time.sleep(timestep_ms / 1000.0)
                    continue

                target_x = target.x / 512.0
                target_y = target.y / 384.0
                time_to_target = (target.time - current_time) / 1000.0

                vel_0_x = self.cursor_x - position_history[0][0]
                vel_0_y = self.cursor_y - position_history[0][1]
                vel_1_x = position_history[0][0] - position_history[1][0]
                vel_1_y = position_history[0][1] - position_history[1][1]
                vel_2_x = position_history[1][0] - position_history[2][0]
                vel_2_y = position_history[1][1] - position_history[2][1]
                vel_3_x = position_history[2][0] - position_history[3][0]
                vel_3_y = position_history[2][1] - position_history[3][1]

                dx = target_x - self.cursor_x
                dy = target_y - self.cursor_y
                distance = np.sqrt(dx**2 + dy**2)
                angle = np.arctan2(dy, dx)

                input_feat = torch.tensor([[
                    self.cursor_x, self.cursor_y,
                    target_x, target_y,
                    time_to_target, distance,
                    np.sin(angle), np.cos(angle),
                    vel_0_x, vel_0_y,
                    vel_1_x, vel_1_y,
                    vel_2_x, vel_2_y,
                    vel_3_x, vel_3_y
                ]], dtype=torch.float32).unsqueeze(0).to(self.device)

                prediction, self.hidden = self.model(input_feat, self.hidden)

                delta_x = float(prediction[0, 0, 0])
                delta_y = float(prediction[0, 0, 1])

                self.cursor_x += delta_x
                self.cursor_y += delta_y

                self.cursor_x = np.clip(self.cursor_x, 0.0, 1.0)
                self.cursor_y = np.clip(self.cursor_y, 0.0, 1.0)

                position_history.pop()
                position_history.insert(0, (self.cursor_x, self.cursor_y))

                self.move_mouse(self.cursor_x, self.cursor_y)

                time.sleep(timestep_ms / 1000.0)

        print("Finished!")


def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists('aim_model_best.pt'):
        print("Error: Model file 'aim_model_best.pt' not found")
        print("Please train the model first using: python scripts/train.py")
        return

    model = AimGRU(input_size=16, hidden_size=128, num_layers=2, output_size=2, dropout=0.2)
    model.load_state_dict(torch.load('aim_model_best.pt', map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    print(f"Model loaded successfully")
    print(f"Device: {DEVICE}\n")

    # Get screen resolution
    user32 = ctypes.windll.user32
    screen_width = user32.GetSystemMetrics(0)
    screen_height = user32.GetSystemMetrics(1)

    print("="*60)
    print("OSU! LIVE PLAYER SETUP")
    print("="*60)
    print(f"\nDetected screen resolution: {screen_width}x{screen_height}")
    print("\nIMPORTANT: In osu! settings, you MUST:")
    print("  1. Go to Options > Mouse")
    print("  2. DISABLE 'Raw Input'")
    print("  3. Set 'Sensitivity' to 1.0x")
    print("\nWithout disabling raw input, the cursor won't move in-game!")
    print("\nThe script will auto-calculate playfield bounds based on")
    print("osu!'s standard 512x384 aspect ratio (4:3).")

    print("\n" + "="*60)
    print("BEATMAP SELECTION")
    print("="*60)
    print("\nYou need to provide the full path to a .osu beatmap file.")
    print("\nExample:")
    print('  C:\\Users\\brian\\AppData\\Local\\osu!\\Songs\\454843 P_Light - Storm Buster\\P_Light - Storm Buster (Asphyxia) [Axarious\' Extra].osu')
    print("\nTip: In the osu! songs folder, open the song directory and")
    print("      copy the full path of the specific .osu difficulty file.")

    beatmap_path = input("\nEnter path to .osu beatmap file: ").strip('"').strip("'")

    if not os.path.exists(beatmap_path):
        print(f"\nError: File not found: {beatmap_path}")
        return

    if not beatmap_path.endswith('.osu'):
        print(f"\nError: Not a .osu file: {beatmap_path}")
        print("You provided a directory or wrong file type.")
        print("Please provide the full path to a .osu beatmap file.")
        return

    audio_offset = input("\nAudio offset in ms (0 if synced): ").strip()
    audio_offset_ms = int(audio_offset) if audio_offset else 0

    player = LivePlayer(model, DEVICE, screen_width, screen_height)
    player.play_beatmap(beatmap_path, audio_offset_ms=audio_offset_ms)


if __name__ == "__main__":
    main()
