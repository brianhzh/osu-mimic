import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import os
import requests

API_URL = "http://localhost:8000"


class OsuMimicApp:
    def __init__(self, root):
        root.title("osu!mimic")
        root.geometry("500x250")
        root.resizable(False, False)

        self.osu_path = None

        # API URL
        url_frame = tk.Frame(root)
        url_frame.pack(fill=tk.X, padx=20, pady=(20, 5))
        tk.Label(url_frame, text="API URL:").pack(side=tk.LEFT)
        self.url_entry = tk.Entry(url_frame)
        self.url_entry.insert(0, API_URL)
        self.url_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))

        # File selection
        file_frame = tk.Frame(root)
        file_frame.pack(fill=tk.X, padx=20, pady=5)
        self.file_label = tk.Label(file_frame, text="No file selected", anchor="w")
        self.file_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(file_frame, text="Browse", command=self.browse_file).pack(side=tk.RIGHT)

        # Generate button
        self.generate_btn = tk.Button(root, text="Generate Replay", command=self.generate, state=tk.DISABLED)
        self.generate_btn.pack(pady=20)

        # Status
        self.status_label = tk.Label(root, text="", fg="gray")
        self.status_label.pack()

    def browse_file(self):
        path = filedialog.askopenfilename(
            title="Select beatmap",
            filetypes=[("osu! beatmap", "*.osu")],
        )
        if path:
            self.osu_path = path
            self.file_label.config(text=os.path.basename(path))
            self.generate_btn.config(state=tk.NORMAL)

    def generate(self):
        if not self.osu_path:
            return

        self.generate_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Generating replay...", fg="gray")
        threading.Thread(target=self._do_generate, daemon=True).start()

    def _do_generate(self):
        try:
            url = self.url_entry.get().rstrip("/") + "/generate"
            with open(self.osu_path, "rb") as f:
                resp = requests.post(url, files={"file": (os.path.basename(self.osu_path), f)})

            if resp.status_code != 200:
                error = resp.json().get("error", f"Server error ({resp.status_code})")
                self.status_label.config(text=error, fg="red")
                self.generate_btn.config(state=tk.NORMAL)
                return

            # Save .osr next to the .osu file
            osr_name = os.path.splitext(os.path.basename(self.osu_path))[0] + ".osr"
            osr_dir = os.path.dirname(self.osu_path)
            osr_path = os.path.join(osr_dir, osr_name)

            with open(osr_path, "wb") as f:
                f.write(resp.content)

            self.status_label.config(text=f"Saved: {osr_name}", fg="green")
        except requests.ConnectionError:
            self.status_label.config(text="Cannot connect to API", fg="red")
        except Exception as e:
            self.status_label.config(text=str(e), fg="red")
        finally:
            self.generate_btn.config(state=tk.NORMAL)


if __name__ == "__main__":
    root = tk.Tk()
    OsuMimicApp(root)
    root.mainloop()
