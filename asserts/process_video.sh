#!/bin/bash
set -e

SRC="/root/host/ssd2/zhangzhimin04/workspaces_ocs/CiCiAgent/asserts/cici.mp4"
OUT_MP4="/root/host/ssd2/zhangzhimin04/workspaces_ocs/CiCiAgent/asserts/cici_trimmed.mp4"
OUT_GIF="/root/host/ssd2/zhangzhimin04/workspaces_ocs/CiCiAgent/asserts/cici_trimmed.gif"

# Delete 6s~35s and 36s~67s. Keep [0,6], [35,36], [67,end].
# Source has video only (no audio stream).
ffmpeg -y -i "$SRC" -filter_complex "
[0:v]trim=start=0:end=6,setpts=PTS-STARTPTS[v0];
[0:v]trim=start=35:end=36,setpts=PTS-STARTPTS[v1];
[0:v]trim=start=67:end=75,setpts=PTS-STARTPTS[v2];
[v0][v1][v2]concat=n=3:v=1:a=0[outv]
" -map "[outv]" -c:v libx264 -preset veryfast -crf 20 "$OUT_MP4"

# Convert to GIF using palette for better quality
PALETTE="/tmp/palette_cici.png"
ffmpeg -y -i "$OUT_MP4" -vf "fps=20,scale=1080:-1:flags=lanczos,palettegen=max_colors=256:stats_mode=diff" "$PALETTE"
ffmpeg -y -i "$OUT_MP4" -i "$PALETTE" -lavfi "fps=20,scale=1080:-1:flags=lanczos [x]; [x][1:v] paletteuse=dither=sierra2_4a" "$OUT_GIF"
rm -f "$PALETTE"

echo "Done:"
echo "  MP4: $OUT_MP4"
echo "  GIF: $OUT_GIF"
