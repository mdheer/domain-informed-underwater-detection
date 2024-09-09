import json

path_annotations = r"C:\Users\mathi\Desktop\unity_sim_v6\parsed_data_raw.json"
new_path = r"C:\Users\mathi\Desktop\unity_sim_v6\parsed_data.json"

# Extract annotations
with open(path_annotations, "r") as f:
    annotations_data = json.load(f)

new_dict = {}

for ky in annotations_data.keys():
    lst = annotations_data[ky]["unity_annotations"]["frameTime"]
    if len(lst) <= 2:
        pass
    else:
        new_dict[ky] = annotations_data[ky]

# Save updated master JSON data
with open(new_path, "w") as f:
    json.dump(new_dict, f, indent=2)
