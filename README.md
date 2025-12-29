# osu!mimic

a neural network trained to replicate human aiming behavior in osu! as closely as possible.

## goal

mimic human osu! gameplay with relax mod - not just aim at circles correctly, but aim **like a human would**. this means smooth acceleration curves, natural hovering between targets, and human-like timing and movement patterns.

## approach

instead of hardcoding aiming logic, the model learns purely from watching human replays. a gru-based recurrent network observes how humans move their cursor and learns to reproduce those patterns.

## project timeline

### phase 1: initial implementation
**goal**: basic circle-to-circle aiming

**architecture**:
- 2-layer gru (128 hidden units) with delta-based output
- 10 input features: cursor position, target position, time/distance/angle to target, previous velocity
- trained on 7 beatmap/replay pairs

**result**: model learned to aim in the correct direction and change direction at appropriate times, but movements were too slow

**problem identified**: severe underaiming on distant circles - cursor moved too slowly to arrive on time

---

### phase 2: velocity constraint enforcement
**goal**: fix underaiming by teaching the model the physics of required velocity

**changes**:
- added explicit velocity-time constraint loss (v = d/t)
  - heavily penalizes velocity too low to reach target (5x penalty)
  - light penalty for moving too fast (0.3x penalty)
- primary loss function with weight 5.0

**result**: underaiming fixed - cursor now arrives at circles on time

**new problem**: cursor stayed slow but then abruptly accelerated right before hitting circles (inhuman movement pattern)

**new problem**: cursor wandered and drifted toward top-left when circles were spaced far apart time-wise

---

### phase 3: human-like movement refinement (current)
**goal**: smooth acceleration ramps and stable hovering behavior

**changes**:
- expanded velocity history from 1-frame to 4-frame (64ms history)
  - allows model to learn acceleration trends and momentum
  - input features expanded from 10 → 16 dimensions
- acceleration smoothness loss (weight 2.0)
  - penalizes jerk (rapid changes in acceleration)
  - encourages gradual speed-up when approaching circles
- hovering stability loss (weight 1.0)
  - detects "hovering phase" (time > 0.5s AND distance > 0.3)
  - penalizes unnecessary movement when far from next target
  - prevents wandering/drifting behavior

**expected result**: smooth, human-like acceleration curves with stable hovering between targets

**status**: implemented, ready for training

---

### future phases

**phase 4: expanded dataset**
- increase from 7 maps to 20+ maps
- focus on jump maps with large spacing between circles
- ensure variety in bpm, spacing patterns, and difficulty

**phase 5: slider support**
- add slider-specific input features (path shape, duration, progress)
- implement conditional loss functions (circle vs slider mode)
- train on mixed circle+slider maps

**phase 6: advanced human mimicry**
- implement overshoot correction (humans don't aim perfectly)
- add micro-adjustments and small jitter
- study and replicate aim "wobble" patterns
- per-player style transfer

## current architecture

**input features** (16 dimensions):
- cursor position (x, y)
- target position (x, y)
- time to target, distance to target
- direction to target (sin, cos)
- 4-frame velocity history (8 values)

**model**:
- 2-layer gru (128 hidden units)
- 3-layer decoder (128 → 32 → 2)
- outputs delta movements per frame (16ms timestep)
- ~350k parameters

**training objectives** (phase 3):
```
velocity-time constraint (5.0) ← enforce v=d/t physics
acceleration smoothness (2.0)  ← prevent abrupt speed changes
hovering stability (1.0)       ← reduce wandering
speed matching (1.0)           ← match replay velocity profiles
arrival loss (0.5)             ← penalize under-aiming
direction loss (0.3)           ← encourage movement toward targets
position loss (1.0)            ← mse on delta predictions
```

## requirements

```bash
pip install torch numpy keyboard
```

- python 3.8+
- pytorch
- numpy
- keyboard (for live play)

## usage

### 1. prepare data

place beatmap files (`.osu`) and replay files (`.osr`) in matching order:

```
data/
├── beatmaps/
│   ├── map1.osu
│   ├── map2.osu
│   └── ...
└── replays/
    ├── map1.osr
    ├── map2.osr
    └── ...
```

**important**: currently only supports circle-only maps. sliders and spinners will confuse the model.

### 2. train

```bash
python scripts/train.py
```

- trains for 30 epochs (~20-30 minutes on gpu)
- saves best model to `aim_model_best.pt`
- validation mse typically ~0.001-0.002

### 3. play live

```bash
python scripts/play_live.py path/to/beatmap.osu
```

- runs at 60+ fps (16ms timestep)
- press esc to stop
- use relax mod in osu! (model only handles aiming, not clicking)

### 4. evaluate

```bash
python scripts/evaluate_model.py
```

shows per-map performance metrics

## project structure

```
osu!mimic/
├── app/
│   ├── core/
│   │   ├── parser.py           # beatmap parser
│   │   └── replay_parser.py    # replay parser
│   ├── models/
│   │   └── aim_model.py        # gru architecture
│   └── data/
│       └── dataset.py          # training data pipeline
├── scripts/
│   ├── train.py                # training script
│   ├── play_live.py            # live inference
│   ├── evaluate_model.py       # model evaluation
│   └── test_inference.py       # visualization
└── data/                       # beatmaps and replays (not in repo)
```

## known limitations

- **circle-only**: sliders and spinners not supported (phase 5)
- **relax mod only**: does not handle tapping/clicking
- **limited dataset**: currently 7 maps (expanding in phase 4)
- **no player-specific mimicry**: learns general human behavior, not individual styles

## contributing

the goal is human-like gameplay. when contributing, prioritize:
1. **naturalness over accuracy** - smooth curves > perfect precision
2. **learning from data** - avoid hardcoded heuristics
3. **measurable improvements** - demonstrate before/after comparison

## license

mit
