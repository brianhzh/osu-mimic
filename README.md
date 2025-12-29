# osu!mimic

An RNN trained to replicate human aiming behavior in osu! as closely as possible.

## Approach

trained on human replays, a GRU-based model observes aiming patterns to mimic aim movements, then uses RL to imitate human tapping.

## Project timeline

### Phase 1: Aiming
**Goal**: basic circle-to-circle aiming, capable of ss'ing maps with the relax mod.

**1.1:** circles only (complete)

- capable of 98%+ on high sr (8*+ maps)
  
cursor trajectory from model |  heatmap of objects in a random map
-------------------------|-------------------------
<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/b7b4323e-a4bd-4810-ac6b-bc3baba70440" /> |  <img width="500" height="400" alt="image" src="https://github.com/user-attachments/assets/b51284b4-2484-49a9-aa45-797fc2230083" />

**1.2:** circles and sliders

**Goal:** high accuracy + combo on standard maps of any star rating
- follow sliders naturally
  - shouldn't follow sliderpath exactly, only enough to not miss sliderends

**1.3:** refine movement

**Goal:** refine cursor movement to mimic human characteristics
  - smoother cursor movement, no jitters on most sections
  - more realistic speed up + deceleration
  - cursor stays in place when circles are not present in playfield
  - maybe use RL?

### Phase 2: Tapping
**Goal**: imitate human tapping (unstable rate > 40)

- reward algorithm should reward human-like behaviours
  - reward when it hits 300's
  - increasing penalty for 100's, 50's, and misses
  - increasing penalty as tapping UR (unstable rate) approaches 0
    - discourages unrealistic tapping 

## Requirements

```bash
pip install torch numpy keyboard
```

- python 3.8+
- pytorch
- numpy
- keyboard (for live play)

## usage

not quite ready yet...
