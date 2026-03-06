"""
统一配置管理模块

所有模型名称、路径、参数都在此集中管理
"""
import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """模型配置 - 所有模型相关配置"""
    
    # 模型根目录
    MODEL_ROOT: Path = field(default_factory=lambda: Path("./models"))
    
    # ==================== 语音识别模型 ====================
    # faster-whisper 模型
    FASTER_WHISPER_NAME: str = "faster-whisper-large-v3-turbo"
    FASTER_WHISPER_PATH: Path = field(init=False)
    
    # whisperx 模型  
    WHISPERX_NAME: str = "Whisper-large-v3-turbo"
    WHISPERX_PATH: Path = field(init=False)
    
    # wav2vec2 中文模型
    WAV2VEC2_NAME: str = "wav2vec2-large-xlsr-53-chinese-zh-cn"
    WAV2VEC2_PATH: Path = field(init=False)
    
    # ==================== 说话人分离模型 ====================
    PYANNOTE_NAME: str = "pyannote/speaker-diarization-community-1"
    PYANNOTE_PATH: Path = field(init=False)
    
    # ==================== 声纹识别模型 ====================
    SPEECHBRAIN_NAME: str = "speechbrain/spkrec-xvect-voxceleb"
    SPEECHBRAIN_PATH: Path = field(init=False)
    
    def __post_init__(self):
        """初始化路径"""
        self.FASTER_WHISPER_PATH = self.MODEL_ROOT / self.FASTER_WHISPER_NAME
        self.WHISPERX_PATH = self.MODEL_ROOT / self.WHISPERX_NAME
        self.WAV2VEC2_PATH = self.MODEL_ROOT / self.WAV2VEC2_NAME
        # 说话人分离模型 - 优先使用 community 版本
        # 先检查 community-1，不存在则用 3.1
        community_path = self.MODEL_ROOT / "pyannote" / "speaker-diarization-community-1"
        v31_path = self.MODEL_ROOT / "pyannote" / "speaker-diarization-3.1"
        if community_path.exists():
            self.PYANNOTE_PATH = community_path
        else:
            self.PYANNOTE_PATH = v31_path
        self.SPEECHBRAIN_PATH = self.MODEL_ROOT / "speechbrain" / "spkrec-xvect-voxceleb"
    
    def get_model_path(self, model_type: str) -> Optional[Path]:
        """获取指定类型模型的路径"""
        mapping = {
            "faster-whisper": self.FASTER_WHISPER_PATH,
            "whisperx": self.WHISPERX_PATH,
            "wav2vec2": self.WAV2VEC2_PATH,
            "pyannote": self.PYANNOTE_PATH,
            "speechbrain": self.SPEECHBRAIN_PATH,
        }
        return mapping.get(model_type)
    
    def init_dirs(self):
        """初始化所有模型目录"""
        self.MODEL_ROOT.mkdir(exist_ok=True)
        paths = [
            self.FASTER_WHISPER_PATH,
            self.WHISPERX_PATH,
            self.WAV2VEC2_PATH,
            self.PYANNOTE_PATH,
            self.SPEECHBRAIN_PATH,
        ]
        for path in paths:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class ProcessConfig:
    """处理流程配置"""
    
    # 默认语言
    DEFAULT_LANGUAGE: str = "zh"
    
    # 语音识别参数
    VAD_FILTER: bool = False
    MIN_SPEECH_DURATION: float = 0.1
    BEAM_SIZE: int = 5
    
    # 说话人分离参数
    DIARIZATION_MIN_SPEAKERS: int = 1
    DIARIZATION_MAX_SPEAKERS: int = 20
    
    # 对话合并参数
    MERGE_GAP_THRESHOLD: float = 0.8
    
    # 降噪参数
    NOISE_REDUCTION_ENABLE: bool = False
    NOISE_REDUCTION_STRENGTH: float = 0.5


@dataclass
class OutputConfig:
    """输出配置"""
    
    OUTPUT_DIR: Path = field(default_factory=lambda: Path("./outputs"))
    OUTPUT_FORMATS: List[str] = field(default_factory=lambda: ["json", "txt", "srt", "vtt", "csv", "clean"])
    TIMESTAMP_FORMAT: str = "[%06.2f - %06.2f]"
    SRT_ENCODING: str = "utf-8"
    VTT_ENCODING: str = "utf-8"
    
    def init_dirs(self):
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass  
class DeviceConfig:
    """设备配置"""
    
    # 设备选择
    DEVICE: str = field(init=False)
    COMPUTE_TYPE: str = "int8"
    
    # 并行参数
    NUM_WORKERS: int = 2
    CPU_THREADS: int = 2
    
    # 批处理大小
    BATCH_SIZE: int = 8
    
    def __post_init__(self):
        import torch
        if torch.cuda.is_available():
            self.DEVICE = "cuda"
            self.COMPUTE_TYPE = "float16"
        else:
            self.DEVICE = "cpu"
            self.COMPUTE_TYPE = "int8"
    
    @classmethod
    def from_args(cls, cpu: bool = False, device: str = None) -> "DeviceConfig":
        """从命令行参数创建设备配置"""
        config = cls()
        if cpu:
            config.DEVICE = "cpu"
            config.COMPUTE_TYPE = "int8"
        elif device:
            config.DEVICE = device
            config.COMPUTE_TYPE = "float16" if device == "cuda" else "int8"
        return config


@dataclass
class SpeakerConfig:
    """说话人配置"""
    
    # 说话人映射 (pyannote标签 -> 显示名称)
    SPEAKER_MAPPING: Dict[str, str] = field(default_factory=lambda: {
        "SPEAKER_00": "A",
        "SPEAKER_01": "B", 
        "SPEAKER_02": "C",
        "SPEAKER_03": "D",
        "SPEAKER_04": "E",
        "SPEAKER_05": "F",
        "SPEAKER_06": "G",
        "SPEAKER_07": "H",
        "未知": "未知",
    })
    
    def get_display_name(self, speaker_label: str) -> str:
        """获取显示名称"""
        return self.SPEAKER_MAPPING.get(speaker_label, speaker_label)


# 全局配置实例
model_config = ModelConfig()
process_config = ProcessConfig()
output_config = OutputConfig()
device_config = DeviceConfig()
speaker_config = SpeakerConfig()


def get_config() -> Dict:
    """获取所有配置"""
    return {
        "model": model_config,
        "process": process_config,
        "output": output_config,
        "device": device_config,
        "speaker": speaker_config,
    }


def init_all_dirs():
    """初始化所有目录"""
    model_config.init_dirs()
    output_config.init_dirs()
