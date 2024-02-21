#!/bin/bash

directory="outputs/2024-01-22"
size_threshold=$((10 * 1024 * 1024))  # 10MB

delete_small_folders() {
  local dir=$1

  for folder in "$dir"/*; do
    if [ -d "$folder" ]; then
      folder_size=$(du -s "$folder" | awk '{print $1}')
      if [ "$folder_size" -lt "$size_threshold" ]; then
        rm -rf "$folder"
        echo "Deleted folder: $folder"
      else
        delete_small_folders "$folder"
      fi
    fi
  done
}

delete_small_folders "$directory"