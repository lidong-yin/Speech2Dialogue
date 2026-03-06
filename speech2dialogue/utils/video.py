"""视频处理工具模块"""
import subprocess
from pathlib import Path
from typing import Optional


class VideoProcessor:
    """视频处理工具"""
    
    @staticmethod
    def extract_audio(
        video_path: str, 
        output_path: Optional[str] = None, 
        audio_rate: int = 16000
    ) -> str:
        """
        从视频中提取音频
        
        Args:
            video_path: 视频文件路径
            output_path: 输出音频路径 (可选)
            audio_rate: 音频采样率
            
        Returns:
            提取的音频文件路径
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        if output_path is None:
            output_path = str(video_path.with_suffix(".wav"))
        else:
            output_path = str(Path(output_path).with_suffix(".wav"))
        
        print(f"🎬 从视频提取音频: {video_path.name}")
        
        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-vn", "-acodec", "pcm_s16le",
                "-ar", str(audio_rate),
                "-ac", "1",
                output_path
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            print(f"✅ 音频提取成功: {Path(output_path).name}")
            return output_path
            
        except subprocess.CalledProcessError as e:
            print(f"❌ 音频提取失败: {e.stderr}")
            raise
        except FileNotFoundError:
            print("❌ ffmpeg 未安装，请安装 ffmpeg")
            raise
    
    @staticmethod
    def is_video(file_path: str) -> bool:
        """检查是否为视频文件"""
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
        return Path(file_path).suffix.lower() in video_extensions
