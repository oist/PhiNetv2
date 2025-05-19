#!/bin/bash

DEST_DIR="filtered_videos"

# 移動先ディレクトリを作成（存在しない場合）
mkdir -p "$DEST_DIR"

# filter_files.txt を1行ずつ処理
while IFS= read -r filepath; do
    [[ -z "$filepath" || "$filepath" == \#* ]] && continue

    filename=$(basename "$filepath")
    mv "$filepath" "$DEST_DIR/$filename"

done < filter_files.txt
