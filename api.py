import os
import tempfile
import torch
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from datetime import datetime, timezone
from osrparse import Replay, GameMode, Mod

from app.core.parser import Parser
from app.models.aim_model import AimGRU
from scripts.generate_replay import (
    expand_hit_objects_to_targets,
    generate_frames,
    frames_to_replay_events,
    get_beatmap_hash,
)

app = FastAPI(title="osu!mimic")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None


@app.on_event("startup")
def load_model():
    global model
    model_path = os.path.join(os.path.dirname(__file__), 'aim_model_best.pt')
    model = AimGRU(input_size=17, hidden_size=128, num_layers=2, output_size=2, dropout=0.2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/version")
def version():
    return {"version": "1.0.0"}


@app.post("/generate")
async def generate(file: UploadFile):
    with tempfile.NamedTemporaryFile(suffix=".osu", delete=False) as tmp_osu:
        tmp_osu.write(await file.read())
        tmp_osu_path = tmp_osu.name

    try:
        beatmap = Parser(tmp_osu_path).parse()
        if not beatmap.hit_objects:
            return {"error": "no hit objects found"}

        targets = expand_hit_objects_to_targets(beatmap)
        frames = generate_frames(model, device, beatmap, targets)
        events = frames_to_replay_events(frames)
        beatmap_hash = get_beatmap_hash(tmp_osu_path)

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

        filename = f"{artist} - {title} [{version}].osr"
        for char in '<>:"/\\|?*':
            filename = filename.replace(char, '_')

        tmp_osr = tempfile.NamedTemporaryFile(suffix=".osr", delete=False)
        tmp_osr.close()
        replay.write_path(tmp_osr.name)

        cleanup = BackgroundTask(os.unlink, tmp_osr.name)
        return FileResponse(
            tmp_osr.name,
            media_type="application/octet-stream",
            filename=filename,
            background=cleanup,
        )
    finally:
        os.unlink(tmp_osu_path)
