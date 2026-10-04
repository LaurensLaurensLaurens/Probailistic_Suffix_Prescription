import pickle
from pathlib import Path

def load_suffix_results(path: str) -> None:
        """
        Load all .pkl files from the given directory into self.d_con_results.
        """
        results = []
        print(f"Looking for .pkl files in: {path}")
        print(f"Resolved absolute path: {Path(path).resolve()}")
        file_paths = list(Path(path).glob("*.pkl"))
        print(f"Found {len(file_paths)} files")

        for file_path in file_paths:
            with file_path.open('rb') as f:
                data = pickle.load(f)
                if len(file_paths) == 1:
                    results = data
                else:
                    results.append(data)
                
        print("Loaded all suffix results!")
        return results
    
def load_conformance_results(path: str) -> dict:
    """
    Load all .pkl files from the given directory and merge them into a single dictionary.
    """
    print(f"Looking for .pkl files in: {path}")
    print(f"Resolved absolute path: {Path(path).resolve()}")
    
    # Find and sort the relevant .pkl files by the part number
    file_paths = sorted(
        Path(path).glob("conformance_res_*.pkl"),
        key=lambda p: int(p.stem.split('_')[-1])
    )
    print(f"Found {len(file_paths)} files")
    
    if not file_paths:
        raise FileNotFoundError("No .pkl files found in the directory.")
    
    merged_dict = None
    for file_path in file_paths:
        with file_path.open('rb') as f:
            data = pickle.load(f)
            if merged_dict is None:
                merged_dict = data
            else:
                for key in merged_dict:
                    if key in data:
                        merged_dict[key].extend(data[key])
                    else:
                        raise KeyError(f"Key {key} missing in {file_path}")
    
    print("Loaded all conformance results!")
    return merged_dict
    