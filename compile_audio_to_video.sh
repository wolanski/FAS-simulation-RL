#!/bin/sh
ffmpeg -i sequence.wav -pix_fmt yuv420p -loop 1 -i ./output/*.png -vf fps=10 -vcodec mpeg4 -strict -2 -b:a 384k -acodec aac -c:a aac -shortest fas2.mp4

