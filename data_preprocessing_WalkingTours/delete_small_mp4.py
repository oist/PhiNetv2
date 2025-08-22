import os

def delete_small_mp4_files(root_dir, max_size_mb=1):
    """
    root_dir 以下の全ファイルを再帰的に探索し、
    サイズが max_size_mb MB 以下の .mp4 ファイルを削除する。

    :param root_dir: 探索を開始するルートディレクトリ（例："WalkingTours"）
    :param max_size_mb: 削除対象の最大ファイルサイズ（MB）
    """
    max_size_bytes = max_size_mb * 1024 * 1024
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.mp4'):
                filepath = os.path.join(dirpath, filename)
                try:
                    size = os.path.getsize(filepath)
                except OSError as e:
                    print(f"ファイル取得エラー: {filepath} — {e}")
                    continue

                if size <= max_size_bytes:
                    try:
                        os.remove(filepath)
                        print(f"削除しました: {filepath} （{size} バイト）")
                    except OSError as e:
                        print(f"削除エラー: {filepath} — {e}")

if __name__ == "__main__":
    target_dir = "/work/YamadaU/myamada/Python/dataset/WalkingTours/"
    if os.path.isdir(target_dir):
        delete_small_mp4_files(target_dir, max_size_mb=1)
    else:
        print(f"指定されたディレクトリが存在しません: {target_dir}")
