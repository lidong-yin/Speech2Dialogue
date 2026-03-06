#!/usr/bin/env python
"""
语音转对话脚本 - 主入口

使用方法:
    python transcriber.py audio.mp3
    python transcriber.py video.mp4 --denoise
    python transcriber.py . --batch
"""
import sys

from speech2dialogue import main

if __name__ == "__main__":
    main()
