import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam

from app.data.dataset import OsuSequenceDataset, train_val_split_by_map, collate_fn
from app.models.aim_model import AimGRU, count_parameters


def smoothness_loss(predictions):
    # penalize rapid changes in velocity
    velocity = predictions[:, 1:, :] - predictions[:, :-1, :]
    acceleration = velocity[:, 1:, :] - velocity[:, :-1, :]
    return torch.mean(acceleration ** 2)


def direction_loss(batch_input, predictions):
    # penalize predictions that don't move toward target
    cursor_pos = batch_input[:, :, 0:2]
    target_pos = batch_input[:, :, 2:4]
    distance_to_target = batch_input[:, :, 5:6]

    target_direction = target_pos - cursor_pos
    target_norm = torch.norm(target_direction, dim=-1, keepdim=True) + 1e-6
    target_direction_normalized = target_direction / target_norm

    pred_norm = torch.norm(predictions, dim=-1, keepdim=True) + 1e-6
    pred_direction_normalized = predictions / pred_norm

    cosine_sim = torch.sum(target_direction_normalized * pred_direction_normalized, dim=-1)
    distance_weight = torch.clamp(distance_to_target.squeeze(-1), 0.1, 1.0)
    weighted_loss = (1.0 - cosine_sim) * distance_weight
    return torch.mean(weighted_loss)


def arrival_loss(batch_input, predictions, batch_output):
    # penalize not arriving at target on time
    time_to_target = batch_input[:, :, 4:5]
    distance_to_target = batch_input[:, :, 5:6]

    pred_magnitude = torch.norm(predictions, dim=-1, keepdim=True)
    actual_magnitude = torch.norm(batch_output, dim=-1, keepdim=True)

    urgency = distance_to_target / (time_to_target + 0.1)
    urgency_weight = torch.clamp(urgency, 0.1, 5.0)

    magnitude_diff = actual_magnitude - pred_magnitude
    under_aim_penalty = torch.relu(magnitude_diff) * 2.0
    over_aim_penalty = torch.relu(-magnitude_diff) * 0.5

    weighted_error = (under_aim_penalty + over_aim_penalty) * urgency_weight
    return torch.mean(weighted_error ** 2)


def speed_match_loss(batch_output, predictions):
    # match replay speed profile
    pred_magnitude = torch.norm(predictions, dim=-1)
    actual_magnitude = torch.norm(batch_output, dim=-1)
    return torch.mean(torch.abs(pred_magnitude - actual_magnitude))


def velocity_time_constraint_loss(batch_input, predictions):
    # enforce v = d/t relationship
    time_to_target = batch_input[:, :, 4:5]
    distance_to_target = batch_input[:, :, 5:6]

    timestep = 0.016
    required_velocity_per_sec = distance_to_target / (time_to_target + 0.01)
    required_velocity_per_frame = required_velocity_per_sec * timestep

    actual_velocity = torch.norm(predictions, dim=-1, keepdim=True)
    velocity_deficit = required_velocity_per_frame - actual_velocity

    under_penalty = torch.relu(velocity_deficit) * 5.0
    over_penalty = torch.relu(-velocity_deficit) * 0.3
    total_penalty = under_penalty + over_penalty

    return torch.mean(total_penalty ** 2)


def acceleration_smoothness_loss(batch_input, predictions):
    # prevent abrupt acceleration changes
    vel_0 = batch_input[:, :, 8:10]
    vel_1 = batch_input[:, :, 10:12]
    vel_2 = batch_input[:, :, 12:14]
    vel_3 = batch_input[:, :, 14:16]

    pred_velocity = predictions

    accel_0_to_pred = pred_velocity - vel_0
    accel_1_to_0 = vel_0 - vel_1
    accel_2_to_1 = vel_1 - vel_2
    accel_3_to_2 = vel_2 - vel_3

    jerk_0 = accel_0_to_pred - accel_1_to_0
    jerk_1 = accel_1_to_0 - accel_2_to_1
    jerk_2 = accel_2_to_1 - accel_3_to_2

    jerk_loss = torch.mean(jerk_0 ** 2) + torch.mean(jerk_1 ** 2) + torch.mean(jerk_2 ** 2)
    return jerk_loss


def hovering_stability_loss(batch_input, predictions):
    # prevent wandering when far from next target
    time_to_target = batch_input[:, :, 4:5]
    distance_to_target = batch_input[:, :, 5:6]

    hovering_phase = (time_to_target > 0.5).float() * (distance_to_target > 0.3).float()
    movement_magnitude = torch.norm(predictions, dim=-1, keepdim=True)
    hovering_penalty = hovering_phase * (movement_magnitude ** 2)

    return torch.mean(hovering_penalty)


def train_epoch(model, dataloader, optimizer, criterion, device, smoothness_weight=0.01, direction_weight=0.3, arrival_weight=0.5, speed_weight=1.0, velocity_constraint_weight=5.0, accel_smoothness_weight=2.0, hovering_weight=1.0):
    model.train()
    total_loss = 0.0
    total_pos_loss = 0.0
    total_dir_loss = 0.0
    total_arrival_loss = 0.0
    total_speed_loss = 0.0
    total_smooth_loss = 0.0
    total_velocity_constraint_loss = 0.0
    total_accel_smooth_loss = 0.0
    total_hovering_loss = 0.0
    num_batches = 0

    for batch_input, batch_output in dataloader:
        batch_input = batch_input.to(device)
        batch_output = batch_output.to(device)

        optimizer.zero_grad()
        predictions, _ = model(batch_input)

        position_loss = criterion(predictions, batch_output)
        smooth_loss = smoothness_loss(predictions)
        dir_loss = direction_loss(batch_input, predictions)
        arr_loss = arrival_loss(batch_input, predictions, batch_output)
        spd_loss = speed_match_loss(batch_output, predictions)
        vel_constraint_loss = velocity_time_constraint_loss(batch_input, predictions)
        accel_smooth_loss = acceleration_smoothness_loss(batch_input, predictions)
        hover_loss = hovering_stability_loss(batch_input, predictions)

        loss = (position_loss +
                smoothness_weight * smooth_loss +
                direction_weight * dir_loss +
                arrival_weight * arr_loss +
                speed_weight * spd_loss +
                velocity_constraint_weight * vel_constraint_loss +
                accel_smoothness_weight * accel_smooth_loss +
                hovering_weight * hover_loss)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_pos_loss += position_loss.item()
        total_dir_loss += dir_loss.item()
        total_arrival_loss += arr_loss.item()
        total_speed_loss += spd_loss.item()
        total_smooth_loss += smooth_loss.item()
        total_velocity_constraint_loss += vel_constraint_loss.item()
        total_accel_smooth_loss += accel_smooth_loss.item()
        total_hovering_loss += hover_loss.item()
        num_batches += 1

    if num_batches == 0:
        return 0.0, {}

    return total_loss / num_batches, {
        'position': total_pos_loss / num_batches,
        'direction': total_dir_loss / num_batches,
        'arrival': total_arrival_loss / num_batches,
        'speed': total_speed_loss / num_batches,
        'smoothness': total_smooth_loss / num_batches,
        'velocity_constraint': total_velocity_constraint_loss / num_batches,
        'accel_smoothness': total_accel_smooth_loss / num_batches,
        'hovering': total_hovering_loss / num_batches
    }


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_input, batch_output in dataloader:
            batch_input = batch_input.to(device)
            batch_output = batch_output.to(device)

            predictions, _ = model(batch_input)
            loss = criterion(predictions, batch_output)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def main():
    SEQUENCE_LENGTH = 512
    BATCH_SIZE = 4
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 30
    VAL_RATIO = 0.3
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Device: {DEVICE}")
    beatmap_files = sorted(glob.glob('data/beatmaps/*.osu'))
    replay_files = sorted(glob.glob('data/replays/*.osr'))

    print(f"\nFound {len(beatmap_files)} beatmaps")
    print(f"Found {len(replay_files)} replays")

    if len(beatmap_files) < 2:
        print("\n[ERROR] Need at least 2 beatmaps for train/val split")
        print("Please download more beatmap/replay pairs")
        return

    pairs = min(len(beatmap_files), len(replay_files))
    beatmap_paths = beatmap_files[:pairs]
    replay_paths = replay_files[:pairs]

    print(f"\nBuilding dataset with {pairs} pairs...")
    dataset = OsuSequenceDataset(
        beatmap_paths=beatmap_paths,
        replay_paths=replay_paths,
        sequence_length=SEQUENCE_LENGTH,
        training_noise=0.002
    )

    print(f"Total sequences: {len(dataset)}")

    train_idx, val_idx = train_val_split_by_map(dataset, val_ratio=VAL_RATIO)

    print(f"\nTrain/val split:")
    print(f"  Train sequences: {len(train_idx)}")
    print(f"  Val sequences: {len(val_idx)}")

    if len(train_idx) == 0 or len(val_idx) == 0:
        print("\n[ERROR] Need more beatmaps for proper split")
        return
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )
    model = AimGRU(
        input_size=16,
        hidden_size=128,
        num_layers=2,
        output_size=2,
        dropout=0.2
    ).to(DEVICE)

    print(f"\nModel: {count_parameters(model):,} parameters")

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    print(f"\nTraining for {NUM_EPOCHS} epochs...")
    print(f"velocity-time constraint (5.0)")
    print(f"acceleration smoothness (2.0)")
    print(f"hovering stability (1.0)")
    print(f"speed matching (1.0)")
    print(f"arrival (0.5)")
    print(f"direction (0.3)")
    print(f"smoothness (0.01)\n")

    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        train_loss, loss_components = train_epoch(
            model, train_loader, optimizer, criterion, DEVICE,
            smoothness_weight=0.01,
            direction_weight=0.3,
            arrival_weight=0.5,
            speed_weight=1.0,
            velocity_constraint_weight=5.0,
            accel_smoothness_weight=2.0,
            hovering_weight=1.0
        )
        val_loss = validate(model, val_loader, criterion, DEVICE)

        print(f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | "
              f"Total: {train_loss:.4f} | "
              f"Val: {val_loss:.6f}")
        print(f"      Pos: {loss_components['position']:.4f} | "
              f"VelC: {loss_components['velocity_constraint']:.4f} | "
              f"AccelSmo: {loss_components['accel_smoothness']:.4f} | "
              f"Hover: {loss_components['hovering']:.4f}")
        print(f"      Spd: {loss_components['speed']:.4f} | "
              f"Arr: {loss_components['arrival']:.4f} | "
              f"Dir: {loss_components['direction']:.4f} | "
              f"Smo: {loss_components['smoothness']:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'aim_model_best.pt')
            print(f"         >> New best model saved!")

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best validation MSE: {best_val_loss:.6f}")
    print(f"Model saved to: aim_model_best.pt")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
