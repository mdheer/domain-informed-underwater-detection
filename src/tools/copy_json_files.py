import os
import shutil
import json


def copy_folders_and_json_files(src, dst):
    # Ensure the destination directory exists
    if not os.path.exists(dst):
        os.makedirs(dst)

    # Walk through the directories and files in the source directory
    for dirpath, dirnames, filenames in os.walk(src):
        # Create corresponding directories in the destination directory
        for dirname in dirnames:
            src_dir = os.path.join(dirpath, dirname)
            dst_dir = os.path.join(dst, os.path.relpath(src_dir, src))
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)

        # Copy json files to the corresponding directory in the destination directory
        for filename in filenames:
            if filename.endswith(".json"):
                src_file = os.path.join(dirpath, filename)
                dst_file = os.path.join(dst, os.path.relpath(src_file, src))
                shutil.copy2(src_file, dst_file)


# Set your source and destination folders here
src_folder = r"M:\underwater_simulator\footage"
dst_folder = r"C:\Users\mathi\Documents\Files\University of Technology Delft\Master\MSc 2\Thesis\Code\thesis_mathieu\unity_data"

copy_folders_and_json_files(src_folder, dst_folder)
