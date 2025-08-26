import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--begin', type=int, default=0)
parser.add_argument('--end', type=int, default=401)
parser.add_argument('--datadir', type=str, default='../../dataset/')

args = parser.parse_args()

root = os.path.join(args.datadir, 'WalkingTours')
output_root = os.path.join(args.datadir, 'WalkingTours3')

os.makedirs(output_root, exist_ok=True)

dirs = ['Dora_WalkingTours_Dataset_Amsterdam','Dora_WalkingTours_Dataset_Istanbul', 'Dora_WalkingTours_Dataset_Stockholm','Dora_WalkingTours_Dataset_Zurich',
        'Dora_WalkingTours_Dataset_Bangkok', 'Dora_WalkingTours_Dataset_KualaLumpur','Dora_WalkingTours_Dataset_Venice',
        'Dora_WalkingTours_Dataset_ChiangMai','Dora_WalkingTours_Dataset_Singapore','Dora_WalkingTours_Dataset_WildLife']

import os

def generate_ffmpeg_command(input_file, output_dir, filename_base):
    lines = []
    lines.append(f"ffmpeg -i {input_file} -filter_complex '")
    # split part
    split_outputs = [f"[out{i}]" for i in range(49)]
    lines.append(f"  [0] split=49 {' '.join(split_outputs)};")
    # crop parts
    for i in range(49):
        x = i % 7
        y = i // 7
        lines.append(f"  [out{i}] crop=iw/7:ih/7:iw/7*{x}:ih/7*{y} [out{i}];")
    # End filter_complex
    # Remove trailing semicolon from last crop
    lines[-1] = lines[-1].rstrip(';')
    lines.append("' \\")
    # map outputs
    for i in range(49):
        idx = i + 1
        output_file = os.path.join(output_dir, f"{filename_base}_{idx}.mp4")
        lines.append(f"  -map '[out{i}]' {output_file} \\")
    # Remove trailing backslash from last line
    lines[-1] = lines[-1].rstrip(" \\")
    return "\n".join(lines)


#if __name__ == "__main__":
#
#    cmd = generate_ffmpeg_command("input.mp4", "outputdir", "video")
#    print(cmd)


for dir in dirs[args.begin:args.end]:
    os.makedirs(os.path.join(output_root, dir), exist_ok=True)
    input_dir = os.path.join(root, dir)
    output_dir = os.path.join(output_root, dir)

    if not os.path.exists(input_dir):
        print(f"Skipping {input_dir} as it does not exist.")
        continue

    for filename in os.listdir(input_dir):
        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, filename)

        basename, _ = os.path.splitext(filename)

        # 入力ファイルと出力ファイルのパスを適切にクォートする
        #cmd = f'ffmpeg -i "{input_file}" -vf "scale=w=256:h=256:force_original_aspect_ratio=decrease,pad=256:256:(ow-iw)/2:(oh-ih)/2" -c:a copy "{output_file}"'
        
        #cmd = f'ffmpeg -i "{input_file}" -vf "scale=trunc(iw/5/2)*2:trunc(ih/5/2)*2" -c:a copy "{output_file}"'
        cmd = generate_ffmpeg_command(input_file, output_dir, basename)
        print(f"Executing: {cmd}")
        os.system(cmd)
