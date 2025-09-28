import os
import argparse

def get_all_mp4_files(root_dir):
    mp4_files = set()
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.mp4'):
                # ファイルの相対パス（サブフォルダ構造を含めて比較）
                relative_path = os.path.relpath(os.path.join(subdir, file), root_dir)
                mp4_files.add(relative_path)
    return mp4_files

def compare_folders(folder1, folder2):
    files1 = get_all_mp4_files(folder1)
    files2 = get_all_mp4_files(folder2)

    only_in_folder1 = files1 - files2
    only_in_folder2 = files2 - files1

    print("=== Files only in Folder 1 ===")
    for f in sorted(only_in_folder1):
        print(f"[Folder1] {f}")

    print("\n=== Files only in Folder 2 ===")
    for f in sorted(only_in_folder2):
        print(f"[Folder2] {f}")

    print("\n=== Summary ===")
    print(f"Total files in Folder 1: {len(files1)}")
    print(f"Total files in Folder 2: {len(files2)}")
    print(f"Different files: {len(only_in_folder1) + len(only_in_folder2)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare mp4 files in two folder trees.")
    parser.add_argument("folder1", help="Path to the first folder")
    parser.add_argument("folder2", help="Path to the second folder")
    args = parser.parse_args()

    compare_folders(args.folder1, args.folder2)
