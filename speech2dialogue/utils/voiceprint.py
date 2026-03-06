"""声纹识别模块

说明:
- speechbrain 库在加载模型时会尝试连接 HuggingFace 验证元数据
- 即使本地模型文件完整，也需要网络访问
- 在完全离线环境下，声纹功能不可用，但主流程 (pyannote) 不受影响
"""
from pathlib import Path
from typing import Optional
import numpy as np
import torch


class VoicePrintRecognizer:
    """声纹识别器"""
    
    def __init__(self, model_dir: str = "./models", device: Optional[str] = None):
        self.model_dir = Path(model_dir)
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
    
    def load_model(self, model_name: str = "speechbrain/spkrec-xvect-voxceleb") -> bool:
        """加载声纹模型
        
        注意: 由于 speechbrain 库的设计，即使模型文件在本地，
        加载时仍会尝试连接 HuggingFace 验证。
        如果网络不可用，声纹功能将不可用，但主流程不受影响。
        """
        print(f"🔄 加载声纹模型: {model_name}")
        
        # 检查本地模型文件是否存在
        model_folder = model_name.split("/")[-1] if "/" in model_name else model_name
        local_path = self.model_dir / "speechbrain" / model_folder
        
        if not local_path.exists():
            print(f"⚠ 声纹模型目录不存在: {local_path}")
            return False
        
        required_files = ['embedding_model.ckpt', 'classifier.ckpt']
        missing = [f for f in required_files if not (local_path / f).exists()]
        if missing:
            print(f"⚠ 模型文件缺失: {missing}")
            return False
        
        # 尝试加载
        try:
            from speechbrain.inference import EncoderClassifier
            
            self.model = EncoderClassifier.from_hparams(
                source=str(local_path),
                savedir=str(local_path),
                run_opts={"device": self.device}
            )
            
            print("✅ 声纹模型加载成功")
            return True
            
        except Exception as e:
            error_msg = str(e)
            if any(x in error_msg.lower() for x in ['connection', 'timeout', 'huggingface', 'hub', 'network']):
                print("⚠ 声纹模型加载失败: 需要网络连接")
                print("   原因: speechbrain 库加载时会验证元数据")
                print("   在完全离线环境下，声纹功能不可用")
                print("   ✅ 主流程不受影响，pyannote 说话人分离正常工作")
            else:
                print(f"⚠ 声纹模型加载失败: {e}")
            return False
    
    def extract_embedding(self, audio_path: str) -> Optional[np.ndarray]:
        """从音频文件提取声纹特征"""
        return self.extract_embedding_from_file(audio_path)
    
    def extract_embedding_from_file(self, audio_path: str) -> Optional[np.ndarray]:
        """从音频文件提取声纹特征"""
        if self.model is None:
            raise ValueError("请先加载声纹模型")
        
        try:
            import librosa
            
            signal, sr = librosa.load(audio_path, sr=16000)
            
            embeddings = self.model.encode_batch(torch.tensor(signal).unsqueeze(0).to(self.device))
            return embeddings.squeeze().cpu().numpy()
            
        except Exception as e:
            print(f"❌ 声纹提取失败: {e}")
            return None
    
    def extract_embedding_from_array(self, audio_data: np.ndarray, sr: int = 16000) -> Optional[np.ndarray]:
        """从音频数组提取声纹特征"""
        if self.model is None:
            raise ValueError("请先加载声纹模型")
        
        try:
            # 重采样如果需要
            if sr != 16000:
                import librosa
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
            
            # 确保是一维数组
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)
            
            embeddings = self.model.encode_batch(
                torch.tensor(audio_data).unsqueeze(0).to(self.device)
            )
            return embeddings.squeeze().cpu().numpy()
            
        except Exception as e:
            print(f"❌ 声纹提取失败: {e}")
            return None
    
    def compare_speakers(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """比较两个声纹特征的相似度"""
        if emb1 is None or emb2 is None:
            return 0.0
        
        emb1 = emb1 / (np.linalg.norm(emb1) + 1e-8)
        emb2 = emb2 / (np.linalg.norm(emb2) + 1e-8)
        
        return float(np.dot(emb1, emb2))
