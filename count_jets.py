import uproot

def count_entries(filelist_path):
    total = 0
    n_files = 0
    n_failed = 0
    with open(filelist_path) as f:
        files = [line.strip() for line in f if line.strip()]
    for i, fpath in enumerate(files):
        try:
            with uproot.open(fpath + ":deepntuplizer/tree") as tree:
                total += tree.num_entries
            n_files += 1
        except Exception as e:
            n_failed += 1
        if (i + 1) % 200 == 0:
            print(f"  ...{i+1}/{len(files)} files, running total = {total:,}")
    return total, n_files, n_failed

for name, path in [
    ("train_val", "filelists/dijet/train_val_samples.txt"),
    ("test", "filelists/dijet/test_samples.txt"),
]:
    print(f"Counting {name} ({path}) ...")
    total, n_ok, n_failed = count_entries(path)
    print(f"{name}: {total:,} jets across {n_ok} files ({n_failed} failed to open)\n")
