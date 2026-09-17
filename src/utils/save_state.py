import json
from pathlib import Path


def save_state(filename, data):
    # 1. Get the directory WHERE THIS SCRIPT LIVES
    script_dir = Path(__file__).resolve().parent
    
    # 2. Define the target folder and final file path
    data_folder = script_dir / "data"
    final_path = data_folder / f"{filename}.json"
    
    # 3. Create the folder if it doesn't exist
    # parents=True ensures it creates intermediate folders if needed
    # exist_ok=True prevents an error if the folder already exists
    data_folder.mkdir(parents=True, exist_ok=True)
    
    # 4. Open the file and dump the JSON data
    with open(final_path, 'w') as f:
        json.dump(data, f, indent=4, default = str)  # indent=4 makes the JSON pretty and readable



def load_state(filename):
    # 1. Get the directory where this script lives
    script_dir = Path(__file__).resolve().parent          # current dir
    parent_dir = script_dir.parent                        # parent
    grandparent_dir = script_dir.parent.parent
    
    # 2. Point to the expected file inside the "data" folder
    final_path = grandparent_dir / "data" / f"{filename}.json"
    
    # 3. Check if the file actually exists before trying to open it
    if not final_path.exists():
        print(f"Warning: No save file found at {final_path}")
        return None  # Or return an empty dict {} depending on your preference
        
    # 4. Open and read the JSON data
    with open(final_path, 'r') as f:
        data = json.load(f)
        return data