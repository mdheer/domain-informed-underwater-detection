import json
import os
import random


def create_random_data_to_json(file_name, num_entries=29):
    random_data = {
        "file_paths": [],
        "3d_x": [],
        "3d_y": [],
        "3d_z": [],
        "3d_vx": [],
        "3d_vy": [],
        "3d_vz": [],
        "bb_x1": [],
        "bb_y1": [],
        "bb_x2": [],
        "bb_y2": [],
        "class": [],
    }

    for _ in range(num_entries):
        random_data["file_paths"].append(
            f"C:\\...\\maskframe{random.randint(0, 9999):04d}.jpg"
        )
        random_data["3d_x"].append(random.uniform(-1000, 1000))
        random_data["3d_y"].append(random.uniform(-1000, 1000))
        random_data["3d_z"].append(random.uniform(-1000, 1000))
        random_data["3d_vx"].append(random.uniform(-800, 800))
        random_data["3d_vy"].append(random.uniform(-800, 800))
        random_data["3d_vz"].append(random.uniform(-800, 800))
        random_data["bb_x1"].append(random.uniform(0, 640))
        random_data["bb_y1"].append(random.uniform(0, 480))
        random_data["bb_x2"].append(random.uniform(0, 640))
        random_data["bb_y2"].append(random.uniform(0, 480))
        random_data["class"].append(random.choice(["fish", "trash"]))

    with open(file_name, "w") as f:
        json.dump(random_data, f, indent=2)


file_name = "random_data.json"
create_random_data_to_json(file_name)
