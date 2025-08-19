import pickle
from pathlib import Path

def list_indexed_videos(input_dir):
    """
    input_dir 以下を再帰的に検索し、.mp4 ファイルをソート順に並べ、
    (インデックス, ファイルパス) のタプルリストを返す。
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    video_files = sorted(input_path.rglob("*.mp4"))
    return [(i, str(p)) for i, p in enumerate(video_files)]

def save_indexed_list_to_pickle(indexed_list, pickle_path):
    """
    (i, filepath) のリストを pickle 形式で保存。
    """
    with open(pickle_path, "wb") as f:
        pickle.dump(indexed_list, f)

if __name__ == "__main__":
    input_dir = "/work/YamadaU/myamada/Python/dataset/WalkingTours/"       # ← 実際のパスに変更してください
    pickle_path = "/work/YamadaU/myamada/Python/dataset/WalkingTours/walking_tours_indexed.pkl"        # 出力ファイル名（例）

    indexed_list = list_indexed_videos(input_dir)
    print(f"Found {len(indexed_list)} video files. Saving to {pickle_path} ...")
    save_indexed_list_to_pickle(indexed_list, pickle_path)
    print("Done.")
