from pathlib import Path

print("Script started")

def list_edfs(data_root):
    data_root = Path(data_root)
    edf_files = list(data_root.rglob("*.edf"))
    print(f"Found {len(edf_files)} EDF files")
    for f in edf_files[:10]:  # show only first 10
        print(f)

if __name__ == "__main__":
    list_edfs("data_raw/DATA")
