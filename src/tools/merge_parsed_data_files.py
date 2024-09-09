import json

# path_main_file = (
#     r"M:\underwater_simulator\parsed_footage_v2023-10-24\footage\parsed_data.json"
# )

path_main_file = r"/data/mdheer/input_data/v2023_10_24/footage/parsed_data.json"

main_dataset_name = "v2023_10_24"
merge_dataset_name = "v2023_11_02"

# path_file_to_merge = (
#     r"M:\underwater_simulator\parsed_footage_v2023-11-02\parsed_data.json"
# )

path_file_to_merge = r"/data/mdheer/input_data/v2023_11_02/footage/parsed_data.json"

output_file = r"/data/mdheer/input_data/merged_v2023-10-24_v2023-11-02/parsed_data.json"


with open(path_main_file, "r") as f:
    main_data = json.load(f)

with open(path_file_to_merge, "r") as f:
    data_to_merge = json.load(f)

keys_to_merge = data_to_merge.keys()
keys_main_data = main_data.keys()

# Last item of the main data
last_item = list(keys_main_data)[-1]
new_start_number = int(last_item[len("data_") :])

modified_dict = {}

# Add dataset name to the original dict
for or_key in keys_main_data:
    main_data[or_key]["dataset_name"] = main_dataset_name

# Perform renumbering and add dataset name to the new dict
for ky in keys_to_merge:
    number = int(ky[len("data_") :])
    new_number = number + new_start_number
    modified_dict[f"data_{new_number}"] = data_to_merge[ky]
    modified_dict[f"data_{new_number}"]["dataset_name"] = merge_dataset_name


main_data.update(modified_dict)

with open(output_file, "w") as f:
    json.dump(main_data, f, indent=4)
