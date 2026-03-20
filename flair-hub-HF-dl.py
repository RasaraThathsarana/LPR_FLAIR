import os
import time
from huggingface_hub import HfApi, hf_hub_download

DATASET_ID = "IGNF/FLAIR-HUB"
DOWNLOAD_DIR = "./FLAIR-HUB_download"

api = HfApi()


def human_bytes(num):
    if num is None:
        return "unknown"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num < 1024:
            return f"{num:.1f} {unit}"
        num /= 1024


print("\nContacting HuggingFace...")

info = api.dataset_info(DATASET_ID, files_metadata=True)

files = []

for s in info.siblings:

    path = s.rfilename

    if not path.endswith(".zip"):
        continue

    if not path.startswith("data/") and not path.endswith("GLOBAL_ALL_MTD.zip"):
        continue

    size = getattr(s, "size", None)

    files.append((path, size))


print(f"\nFound {len(files)} ZIP files\n")


# Show list
for i, (path, size) in enumerate(files):

    print(f"[{i}] {path} ({human_bytes(size)})")


# Ask user
#selection = input("\nEnter file numbers (comma separated) or 'all': ")


#if selection.lower() == "all":

selected = files

#else:

idx = [1,2,11] #[1, 2, 11, 12, 21, 22, 31, 32, 41, 42, 51, 52, 61, 62, 71, 72, 81, 82, 91, 92, 101, 102, 111, 112, 121, 122, 131, 132, 140, 141, 150, 151, 160, 161, 170, 171, 180, 181, 190, 191, 200, 201, 210, 211, 220, 221, 230, 231, 240, 241, 250, 251, 260, 261, 270, 271, 279, 280, 289, 290, 299, 300, 309, 310, 319, 320, 329, 330, 339, 340, 349, 350, 359, 360, 368, 369, 378, 379, 388, 389, 398, 399, 408, 409, 418, 419, 428, 429, 438, 439, 448, 449, 458, 459, 468, 469, 478, 479, 488, 489, 498, 499, 508, 509, 518, 519, 528, 529, 538, 539, 548, 549, 558, 559, 568, 569, 578, 579, 588, 589, 598, 599, 608, 609, 618, 619, 628, 629, 638, 639, 648, 649, 658, 659, 667, 668, 677, 678, 687, 688, 697, 698, 707, 708, 717, 718, 727, 728, 736] #[int(x.strip()) for x in selection.split(",")]

selected = [files[i] for i in idx]


os.makedirs(DOWNLOAD_DIR, exist_ok=True)


print("\nStarting download...\n")


for path, size in selected:

    name = os.path.basename(path)

    print(f"Downloading {name} ({human_bytes(size)})")

    start = time.time()

    hf_hub_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        filename=path,
        local_dir=DOWNLOAD_DIR,
        force_download=True,
    )

    elapsed = time.time() - start

    print(f"Done in {elapsed:.1f}s\n")


print("\nAll downloads complete.")
