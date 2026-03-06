"""
Speech2Dialogue - 完全离线的语音/视频转对话脚本生成器

支持:
- 语音转文字 (faster-whisper, whisperx, wav2vec2)
- 视频支持 (自动提取音频)
- 降噪处理
- 说话人分离 (pyannote)
- 多格式输出 (JSON, TXT, SRT, VTT, CSV)
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")

# 离线模式环境变量
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_EVALUATE_OFFLINE'] = '1'
os.environ['PYANNOTE_CACHE'] = './models/pyannote'

# 版本信息
__version__ = "1.0.0"
__author__ = "Speech2Dialogue Team"

# 导出主要类和函数
from .configs import (
    ModelConfig,
    ProcessConfig, 
    OutputConfig,
    DeviceConfig,
    SpeakerConfig,
    model_config,
    process_config,
    output_config,
    device_config,
    speaker_config,
    get_config,
    init_all_dirs,
)

from .core import (
    OfflineAudioProcessor,
    SpeakerDiarizer,
    DialogueExporter,
)

from .utils import (
    VideoProcessor,
    AudioProcessor,
    VoicePrintRecognizer,
)

from .cli import main, parse_args

__all__ = [
    # 版本
    "__version__",
    
    # 配置
    "ModelConfig",
    "ProcessConfig",
    "OutputConfig", 
    "DeviceConfig",
    "SpeakerConfig",
    "model_config",
    "process_config",
    "output_config",
    "device_config",
    "speaker_config",
    "get_config",
    "init_all_dirs",
    
    # 核心
    "OfflineAudioProcessor",
    "SpeakerDiarizer",
    "DialogueExporter",
    
    # 工具
    "VideoProcessor",
    "AudioProcessor",
    "VoicePrintRecognizer",
    
    # CLI
    "main",
    "parse_args",
]


def run(*args, **kwargs):
    """便捷运行入口"""
    from .cli import main
    return main(*args, **kwargs)
