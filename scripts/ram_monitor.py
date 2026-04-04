"""Monitor RAM changes in FCEUX while user plays.
Simple flow: Enter once, then go shoot the turret, Ctrl+C when done."""

import os
import sys
import time
import numpy as np

STATE_FILE = "/tmp/contra_state.bin"
RAM_SIZE = 2048

for f in [STATE_FILE, "/tmp/contra_action.txt"]:
    try:
        os.remove(f)
    except OSError:
        pass

print("Start FCEUX: fceux --loadlua scripts/fceux_agent.lua roms/contra.nes")
print()

while not os.path.exists(STATE_FILE):
    time.sleep(0.1)

last_mtime = 0

def read_ram():
    global last_mtime
    while True:
        if not os.path.exists(STATE_FILE):
            with open("/tmp/contra_action.txt", "w") as f:
                f.write("0\n")
            time.sleep(0.01)
            continue
        try:
            mtime = os.path.getmtime(STATE_FILE)
        except OSError:
            time.sleep(0.01)
            continue
        if mtime == last_mtime:
            with open("/tmp/contra_action.txt", "w") as f:
                f.write("0\n")
            time.sleep(0.01)
            continue
        last_mtime = mtime
        try:
            with open(STATE_FILE, "rb") as f:
                data = f.read()
        except (IOError, OSError):
            continue
        if len(data) >= RAM_SIZE:
            with open("/tmp/contra_action.txt", "w") as f:
                f.write("0\n")
            return np.frombuffer(data[:RAM_SIZE], dtype=np.uint8).copy()

print("FCEUX connected.")
input("Press Enter, then go shoot the turret. Ctrl+C when done.\n")

print("Monitoring... go shoot!\n")

baseline = read_ram()
changes = {}
samples = 0

try:
    while True:
        ram = read_ram()
        samples += 1
        for addr in range(RAM_SIZE):
            if ram[addr] != baseline[addr]:
                if addr not in changes:
                    changes[addr] = {"count": 0, "vals": [], "first": int(baseline[addr])}
                changes[addr]["count"] += 1
                changes[addr]["vals"].append(int(ram[addr]))
        baseline = ram
except KeyboardInterrupt:
    pass

print(f"\n{samples} samples. Results:\n")

# Show addresses sorted by change count, filter out noise (too many changes = position/timer)
interesting = []
for addr, info in sorted(changes.items()):
    unique = len(set(info["vals"]))
    if 2 <= info["count"] <= 200 and unique <= 30:
        interesting.append((addr, info, unique))

print(f"{'Addr':>6} | {'Count':>6} | {'Unique':>6} | {'Start':>5} | Values (sorted)")
print("-" * 80)
for addr, info, unique in interesting:
    vals = sorted(set(info["vals"]))[:15]
    print(f"0x{addr:03X}  | {info['count']:>6} | {unique:>6} | {info['first']:>5} | {vals}")

for f in ["/tmp/contra_action.txt", STATE_FILE]:
    try:
        os.remove(f)
    except OSError:
        pass
