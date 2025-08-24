import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--begin', type=int, default=0)
parser.add_argument('--end', type=int, default=401)
parser.add_argument('--datadir', type=str, default='../../dataset/')

args = parser.parse_args()

root = os.path.join(args.datadir, 'WalkingTours')
output_root = os.path.join(args.datadir, 'WalkingTours2')

os.makedirs(output_root, exist_ok=True)

dirs = ['Dora_WalkingTours_Dataset_Amsterdam','Dora_WalkingTours_Dataset_Istanbul', 'Dora_WalkingTours_Dataset_Stockholm','Dora_WalkingTours_Dataset_Zurich',
        'Dora_WalkingTours_Dataset_Bangkok', 'Dora_WalkingTours_Dataset_KualaLumpur','Dora_WalkingTours_Dataset_Venice',
        'Dora_WalkingTours_Dataset_ChiangMai','Dora_WalkingTours_Dataset_Singapore','Dora_WalkingTours_Dataset_WildLife']


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

        # 入力ファイルと出力ファイルのパスを適切にクォートする
        cmd = f'ffmpeg -i "{input_file}" -vf "scale=w=256:h=256:force_original_aspect_ratio=decrease,pad=256:256:(ow-iw)/2:(oh-ih)/2" -c:a copy "{output_file}"'
        print(f"Executing: {cmd}")
        os.system(cmd)
