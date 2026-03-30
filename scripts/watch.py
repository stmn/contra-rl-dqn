"""Watch the agent play Contra in FCEUX with real audio.

FCEUX sends RAM + full-res RGB screen as binary.
Python processes identically to training: overlay → grayscale → resize → stack.

Usage:
    ./scripts/watch.sh
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from contra.env.contra_env import ACTIONS, RAM_PLAYER_STATE

ACTION_FILE = "/tmp/contra_action.txt"
STATE_FILE = "/tmp/contra_state.bin"

RAM_SIZE = 2048
SCREEN_W, SCREEN_H = 256, 224
SCREEN_BYTES = SCREEN_W * SCREEN_H * 3
STATE_SIZE = RAM_SIZE + SCREEN_BYTES  # 174080 bytes

# Overlay config (identical to ContraEnv._OVERLAY_CONFIG)
_OVERLAY_CONFIG = {
    1: (200, "vrect"), 2: (200, "vrect"), 3: (180, "triangle"),
    4: (180, "triangle"), 5: (180, "triangle"), 6: (160, "circle"),
    7: (220, "diamond"),
}


def apply_overlay_rgb(frame: np.ndarray, ram: np.ndarray) -> np.ndarray:
    """Draw overlay on RGB frame (identical to ContraEnv._apply_overlay)."""
    frame = frame.copy()
    for slot in range(16):
        routine = ram[0x80 + slot]
        if routine == 0:
            continue
        ehp = ram[0x500 + slot]
        if ehp == 0:
            continue
        ex, ey = int(ram[0x100 + slot]), int(ram[0x180 + slot])
        if ex < 24 or ex > 240:
            continue

        cfg = _OVERLAY_CONFIG.get(routine, (170, "vrect"))
        shade, shape = cfg
        color = (shade, shade, shade)

        if shape == "vrect":
            x1, y1 = max(ex - 3, 0), max(ey - 6, 0)
            x2, y2 = min(ex + 3, 255), min(ey + 6, 239)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        elif shape == "circle":
            cv2.circle(frame, (ex, ey), 5, color, -1)
        elif shape == "triangle":
            pts = np.array([[ex, ey - 6], [ex - 5, ey + 4], [ex + 5, ey + 4]])
            cv2.fillPoly(frame, [pts], color)
        elif shape == "diamond":
            pts = np.array([[ex, ey - 6], [ex - 4, ey], [ex, ey + 6], [ex + 4, ey]])
            cv2.fillPoly(frame, [pts], color)

    # Bullets
    for slot in range(16):
        bx, by = int(ram[0x300 + slot]), int(ram[0x318 + slot])
        if bx == 0 and by == 0:
            continue
        cv2.circle(frame, (bx, by), 3, (240, 240, 240), -1)

    return frame


def build_features(ram: np.ndarray) -> np.ndarray:
    """28-dim feature vector from RAM (identical to ContraEnv._build_features)."""
    f = np.zeros(28, dtype=np.float32)
    px, py = int(ram[0x334]), int(ram[0x31A])
    f[0] = px / 256.0
    f[1] = py / 240.0
    f[2] = 1.0 if ram[RAM_PLAYER_STATE] == 1 else 0.0
    f[3] = 1.0 if (ram[0xA0] & 0x0F) > 0 else 0.0

    enemies = []
    for slot in range(16):
        if ram[0x80 + slot] == 0 or ram[0x500 + slot] == 0:
            continue
        ex, ey = int(ram[0x100 + slot]), int(ram[0x180 + slot])
        if ex < 24 or ex > 240:
            continue
        dx, dy = (ex - px) / 256.0, (ey - py) / 240.0
        enemies.append((abs(dx) + abs(dy), dx, dy))
    enemies.sort()
    for i, (_, dx, dy) in enumerate(enemies[:3]):
        f[4 + i * 4], f[5 + i * 4] = dx, dy

    bullet_idx = 0
    for slot in range(16):
        bx, by = int(ram[0x300 + slot]), int(ram[0x318 + slot])
        if bx == 0 and by == 0:
            continue
        if bullet_idx < 3:
            f[16 + bullet_idx * 2] = (bx - px) / 256.0
            f[17 + bullet_idx * 2] = (by - py) / 240.0
            bullet_idx += 1

    f[22] = min(len(enemies), 8) / 8.0
    f[23] = enemies[0][0] if enemies else 1.0
    return f


def main():
    parser = argparse.ArgumentParser(description="Watch agent play in FCEUX")
    parser.add_argument("--checkpoint", "-c", default=None)
    parser.add_argument("--epsilon", type=float, default=0.05)
    args = parser.parse_args()

    ckpt = args.checkpoint
    if not ckpt:
        for name in ["contra_dqn_manual.pt", "auto_best.pt"]:
            p = Path("checkpoints") / name
            if p.exists():
                ckpt = str(p)
                break
    if not ckpt:
        print("No checkpoint found. Use --checkpoint.")
        sys.exit(1)

    n_actions = len(ACTIONS)
    device = torch.device("cpu")
    hybrid = settings.hybrid_observation

    if hybrid:
        from contra.training.dqn import HybridQNetwork
        model = HybridQNetwork(n_actions).to(device)
    else:
        from contra.training.dqn import QNetwork
        model = QNetwork(n_actions).to(device)

    ckpt_data = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ckpt_data["q_network"])
    model.eval()
    print(f"Model: {ckpt} ({'hybrid' if hybrid else 'CNN-only'})")

    # Frame stack (identical to training: 128x128, 4 channels)
    frame_stack = np.zeros((128, 128, 4), dtype=np.uint8)

    def stack_frame(gray128):
        frame_stack[:, :, :3] = frame_stack[:, :, 1:]
        frame_stack[:, :, 3] = gray128

    for fp in [ACTION_FILE, STATE_FILE]:
        try:
            os.remove(fp)
        except OSError:
            pass

    print("Select 1 Player in FCEUX — agent takes over during gameplay.")
    print("Ctrl+C to stop.\n")

    actions_sent = 0
    last_mtime = 0
    duplicate_count = 0
    unique_count = 0
    prev_gray_hash = None

    try:
        while True:
            if not os.path.exists(STATE_FILE):
                time.sleep(0.001)
                continue

            try:
                mtime = os.path.getmtime(STATE_FILE)
            except OSError:
                continue
            if mtime == last_mtime:
                time.sleep(0.001)
                continue
            last_mtime = mtime

            # Read binary state
            try:
                with open(STATE_FILE, "rb") as f:
                    data = f.read()
            except (IOError, OSError):
                continue

            if len(data) < STATE_SIZE:
                continue

            ram = np.frombuffer(data[:RAM_SIZE], dtype=np.uint8)
            screen_rgb = np.frombuffer(data[RAM_SIZE:RAM_SIZE + SCREEN_BYTES], dtype=np.uint8)
            screen_rgb = screen_rgb.reshape(SCREEN_H, SCREEN_W, 3)

            # Match cynes 240-line output: FCEUX gives 224 (top 224 of 240)
            # Resize to same proportions as cynes pipeline: 240x256 → 128x128
            # So 224x256 → scale by 128/240 = 119x128, then pad bottom to 128x128
            padded = np.zeros((240, 256, 3), dtype=np.uint8)
            padded[0:224, :, :] = screen_rgb

            # Pipeline identical to training:
            # 1. Overlay on full-res RGB (240x256, same coordinate space as cynes)
            if settings.overlay_sprites:
                padded = apply_overlay_rgb(padded, ram)

            # 2. Grayscale (cv2.COLOR_RGB2GRAY — same as training wrapper)
            gray = cv2.cvtColor(padded, cv2.COLOR_RGB2GRAY)

            # 3. Resize to 128x128 (INTER_AREA — same as training wrapper)
            gray128 = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)

            # 4. Stack
            stack_frame(gray128)

            # Debug: save frames after agent starts playing
            if ram[RAM_PLAYER_STATE] == 1 and actions_sent == 460:
                cv2.imwrite("/tmp/fceux_start.png", gray128)
                cv2.imwrite("/tmp/fceux_start_rgb.png", cv2.cvtColor(padded, cv2.COLOR_RGB2BGR))
                # Save all 4 channels of frame stack
                for ch in range(4):
                    cv2.imwrite(f"/tmp/fceux_stack_{ch}.png", frame_stack[:, :, ch])
                print(f"Frame stack saved! mean={gray128.mean():.1f}")

            # First 450 decisions = NOOP (~15s at 30fps, let start frame save)
            if actions_sent < 450:
                action = 0
                with open(ACTION_FILE, "w") as f:
                    f.write("0\n")
                actions_sent += 1
                continue

            # Select action
            if random.random() < args.epsilon:
                action = random.randint(0, n_actions - 1)
            else:
                with torch.no_grad():
                    img_t = torch.from_numpy(
                        frame_stack.transpose(2, 0, 1)
                    ).unsqueeze(0).float().to(device)

                    if hybrid:
                        features = build_features(ram)
                        feat_t = torch.from_numpy(features).unsqueeze(0).to(device)
                        q_values = model(img_t, feat_t)
                    else:
                        q_values = model(img_t)

                    action = q_values.argmax(dim=1).item()

            with open(ACTION_FILE, "w") as f:
                f.write(f"{action}\n")

            # Track duplicate vs unique frames
            gray_hash = hash(gray128.tobytes())
            if gray_hash == prev_gray_hash:
                duplicate_count += 1
            else:
                unique_count += 1
            prev_gray_hash = gray_hash

            actions_sent += 1
            if actions_sent % 200 == 0:
                scroll = int(ram[0x60]) * 256 + int(ram[0x65])
                dup_pct = 100 * duplicate_count / max(actions_sent, 1)
                print(f"#{actions_sent} | scroll={scroll} action={action} | frames: {unique_count} unique, {duplicate_count} dupes ({dup_pct:.0f}%)")

    except KeyboardInterrupt:
        print(f"\nDone. {actions_sent} actions sent.")
    finally:
        for fp in [ACTION_FILE, STATE_FILE]:
            try:
                os.remove(fp)
            except OSError:
                pass


if __name__ == "__main__":
    main()
