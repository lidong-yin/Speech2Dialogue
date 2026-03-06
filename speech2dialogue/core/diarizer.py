"""说话人分离模块"""
from pathlib import Path
from typing import Dict, List, Optional, Any
import torch


class SpeakerDiarizer:
    """说话人分离器"""
    
    def __init__(
        self, 
        model_path: Optional[str] = None,
        device: str = "cpu",
        config: Optional[Dict] = None
    ):
        """
        初始化说话人分离器
        
        Args:
            model_path: 模型路径
            device: 设备
            config: 配置字典
        """
        # 确保 model_path 是 Path 对象
        if model_path is not None:
            self.model_path = Path(model_path)
        else:
            self.model_path = None
        self.device = device
        self.config = config or {}
        self.pipeline = None
    
    def load_model(self, model_name: str = "pyannote/speaker-diarization-community-1", token: Optional[str] = None) -> bool:
        """
        加载说话人分离模型
        
        Args:
            model_name: 模型名称
            token: HuggingFace token (用于访问需要认证的模型)
            
        Returns:
            是否加载成功
        """
        if self.model_path and not self.model_path.exists():
            print(f"⚠ 说话人分离模型不存在: {self.model_path}")
            return False
        
        print(f"🔄 加载说话人分离模型: {model_name}")
        
        try:
            from pyannote.audio import Pipeline
            import os
            
            # 强制离线模式
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            
            # 设置本地缓存路径
            os.environ['PYANNOTE_CACHE'] = str(self.model_path.parent if self.model_path else "./models/pyannote")
            
            if self.model_path and self.model_path.exists():
                # 检查必要文件
                required_files = ['config.yaml']
                missing_files = [f for f in required_files if not (self.model_path / f).exists()]
                
                if missing_files:
                    print(f"⚠ 模型文件缺失: {missing_files}")
                
                # 尝试加载 - 不使用 local_files_only 参数
                self.pipeline = Pipeline.from_pretrained(
                    str(self.model_path.absolute())
                )
            else:
                print(f"⚠ 模型路径不存在: {self.model_path}")
                return False
            
            if self.device == "cuda":
                self.pipeline = self.pipeline.to(torch.device("cuda"))
            else:
                self.pipeline = self.pipeline.to(torch.device("cpu"))
            
            print(f"✅ 说话人分离模型加载成功")
            return True
            
        except Exception as e:
            error_msg = str(e)
            if "authentication token" in error_msg.lower() or "gated" in error_msg.lower() or "xvec_transform" in error_msg.lower():
                print(f"❌ 说话人分离模型需要认证或缺少文件")
                print(f"   本地模型可能不完整，请检查以下文件是否存在:")
                print(f"   - pytorch_model.bin")
                print(f"   - xvec_transform.npz")
                print(f"   - preprocessor_config.yaml")
                print(f"   如果需要认证，请:")
                print(f"   1. 访问 https://hf.co/pyannote/speaker-diarization-3.1 接受用户协议")
                print(f"   2. 创建 token: https://hf.co/settings/tokens")
                print(f"   3. 使用 --token 参数或设置 HF_TOKEN 环境变量")
            else:
                print(f"❌ 说话人分离模型加载失败: {e}")
            return False
            
            if self.device == "cuda":
                self.pipeline = self.pipeline.to(torch.device("cuda"))
            else:
                self.pipeline = self.pipeline.to(torch.device("cpu"))
            
            print(f"✅ 说话人分离模型加载成功")
            return True
            
        except Exception as e:
            error_msg = str(e)
            if "authentication token" in error_msg.lower() or "gated" in error_msg.lower():
                print(f"❌ 说话人分离模型需要认证")
                print(f"   请访问 https://hf.co/pyannote/speaker-diarization-3.1 接受用户协议")
                print(f"   然后创建 token: https://hf.co/settings/tokens")
                print(f"   使用方式: 在代码中传入 token 参数或设置 HF_TOKEN 环境变量")
            else:
                print(f"❌ 说话人分离模型加载失败: {e}")
            return False
    
    def diarize(
        self, 
        audio_path: str, 
        num_speakers: Optional[int] = None
    ) -> Optional[Any]:
        """
        执行说话人分离
        
        Args:
            audio_path: 音频路径
            num_speakers: 说话人数量(可选)
            
        Returns:
            分离结果
        """
        if self.pipeline is None:
            return None
        
        print("👥 进行说话人分离...")
        
        try:
            import librosa
            
            y, sr = librosa.load(audio_path, sr=16000)
            audio = {"waveform": torch.tensor(y).unsqueeze(0), "sample_rate": sr}
            
            if num_speakers:
                diarization = self.pipeline(
                    audio,
                    num_speakers=num_speakers
                )
            else:
                diarization = self.pipeline(audio)
            
            speakers = set()
            
            # Handle different pyannote versions
            diarization_result = diarization
            
            # pyannote 4.x uses DiarizeOutput with speaker_diarization property
            if hasattr(diarization, 'speaker_diarization'):
                diarization_result = diarization.speaker_diarization
            
            for turn, _, speaker in diarization_result.itertracks(yield_label=True):
                speakers.add(speaker)
            
            print(f"✅ 检测到 {len(speakers)} 个说话人: {', '.join(sorted(speakers))}")
            return diarization
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"❌ 说话人分离失败: {e}")
            return None
    
    def assign_speakers(self, diarization: Any, result: Dict) -> Dict:
        """
        为识别结果分配说话人
        
        Args:
            diarization: 分离结果
            result: 识别结果
            
        Returns:
            带说话人标签的结果
        """
        if diarization is None:
            return result
        
        print("🏷️ 分配说话人标签...")
        
        try:
            from pyannote.core import Annotation
            
            annotation = Annotation()
            
            # Handle different pyannote versions
            diarization_result = diarization
            
            if hasattr(diarization, 'speaker_diarization'):
                diarization_result = diarization.speaker_diarization
            
            for turn, _, speaker in diarization_result.itertracks(yield_label=True):
                annotation[turn] = speaker
            
            # Try whisperx method, fallback to manual
            try:
                import whisperx
                result_with_speaker = whisperx.assign_word_speakers(
                    annotation,
                    result
                )
            except Exception:
                result_with_speaker = self._assign_speakers_manual(annotation, result)
            
            print("✅ 说话人分配完成")
            return result_with_speaker
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"❌ 说话人分配失败: {e}")
            return result
    
    def _assign_speakers_manual(self, annotation: Any, result: Dict) -> Dict:
        """手动根据时间重叠分配说话人"""
        if not result.get("segments"):
            return result
        
        for segment in result["segments"]:
            seg_start = segment.get("start", 0)
            seg_end = segment.get("end", 0)
            
            speaker = "未知"
            
            for turn, _, spk in annotation.itertracks(yield_label=True):
                if seg_start < turn.end and seg_end > turn.start:
                    speaker = spk
                    break
            
            segment["speaker"] = speaker
        
        return result
