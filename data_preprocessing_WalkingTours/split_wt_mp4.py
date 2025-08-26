import subprocess
from pathlib import Path
import sys
import argparse
import os

CHUNK_FRAMES = 1200  # 1 チャンクあたりのフレーム数

def get_fps(path: str) -> float:
    """ffprobe で avg_frame_rate を取得し、FPS（浮動小数）として返す"""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate",
        "-of", "default=nw=1:nk=1", path
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    if "/" in out:
        num, den = out.split("/")
        return float(num) / float(den) if float(den) != 0 else float(num)
    return float(out)

def split_video_by_frames(input_file: Path, output_root: Path, chunk_frames: int = CHUNK_FRAMES):
    fps = get_fps(str(input_file))
    chunk_seconds = chunk_frames / fps
    folder = output_root / input_file.stem
    folder.mkdir(parents=True, exist_ok=True)
    out_pattern = str(folder / "video_%d.mp4")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_file),
        "-map", "0",
        "-c", "copy",
        "-f", "segment",
        "-segment_time", f"{chunk_seconds:.6f}",
        "-reset_timestamps", "1",
        out_pattern
    ]
    print(f"Splitting '{input_file.name}' into {folder} (≈{chunk_seconds:.3f}s per chunk)")
    subprocess.run(cmd, check=True)

def existing_dir(path: str) -> Path:
    """argparse の型チェック：ディレクトリの存在確認"""
    p = Path(path)
    if not p.is_dir():
        raise argparse.ArgumentTypeError(f"ディレクトリが存在しません: {path}")
    return p

def main():
    parser = argparse.ArgumentParser(
        description="動画を各 300 フレームごとに分割して、フォルダごとに出力するスクリプト",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_dir", type=existing_dir,
        default=Path("/bucket/YamadaU/Datasets/WalkingTours/"),
        help="入力ディレクトリ（.mp4 ファイルが入っている）"
    )
    parser.add_argument(
        "--output_dir", type=Path,
        default=Path("/work/YamadaU/myamada/Python/dataset/WalkingTours/"),
        help="出力先ディレクトリ"
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    mp4_files = list(input_dir.glob("*.mp4"))
    if not mp4_files:
        print("入力ディレクトリに .mp4 ファイルが見つかりません:", input_dir, file=sys.stderr)
        sys.exit(0)

    for mp4 in mp4_files:
        split_video_by_frames(mp4, output_dir)

if __name__ == "__main__":
    main()
