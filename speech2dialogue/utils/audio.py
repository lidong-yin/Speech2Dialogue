"""音频处理工具模块"""
from pathlib import Path
from typing import Optional, Tuple
import numpy as np


class AudioProcessor:
    """音频处理工具"""
    
    @staticmethod
    def reduce_noise(
        audio_path: str, 
        output_path: Optional[str] = None, 
        strength: float = 0.5
    ) -> str:
        """
        降噪
        
        Args:
            audio_path: 输入音频路径
            output_path: 输出路径
            strength: 降噪强度
            
        Returns:
            降噪后的音频路径
        """
        try:
            import noisereduce as nr
        except ImportError:
            print("⚠ 降噪模块不可用，跳过")
            return audio_path

        if output_path is None:
            output_path = str(Path(audio_path).with_suffix("_denoised.wav"))
        
        print(f"🔊 降噪处理: {Path(audio_path).name}")
        
        try:
            import librosa
            import soundfile as sf
            
            y, sr = librosa.load(audio_path, sr=None)
            
            noise_sample = y[:int(sr * 0.5)]
            
            y_denoised = nr.reduce_noise(
                y=y, 
                sr=sr, 
                y_noise=noise_sample, 
                stationary=True,
                prop_decrease=strength
            )
            
            sf.write(output_path, y_denoised, sr)
            
            print(f"✅ 降噪完成: {Path(output_path).name}")
            return output_path
            
        except Exception as e:
            print(f"⚠ 降噪失败: {e}")
            return audio_path
    
    @staticmethod
    def load_audio(audio_path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
        """
        加载音频
        
        Args:
            audio_path: 音频文件路径
            sr: 采样率
            
        Returns:
            (音频数据, 采样率)
        """
        import librosa
        return librosa.load(audio_path, sr=sr)
    
    @staticmethod
    def is_audio(file_path: str) -> bool:
        """检查是否为音频文件"""
        audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac'}
        return Path(file_path).suffix.lower() in audio_extensions
