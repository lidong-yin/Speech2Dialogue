"""
模型下载工具

提供离线模型下载和验证功能
"""
import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass


# 模型信息配置
MODELS_INFO = {
    "faster-whisper": {
        "name": "Faster Whisper Large V3 Turbo",
        "local_name": "faster-whisper-large-v3-turbo",
        "source": "huggingface",
        "repo_id": "Systran/faster-whisper-large-v3-turbo",
        "description": "高速 Whisper 模型，支持多语言",
    },
    "pyannote": {
        "name": "PyAnnote Speaker Diarization 3.1",
        "local_name": "speaker-diarization-3.1",
        "source": "huggingface",
        "repo_id": "pyannote/speaker-diarization-3.1",
        "description": "说话人分离模型",
    },
    "speechbrain": {
        "name": "SpeechBrain Speaker Embedding",
        "local_name": "spkrec-xvect-voxceleb",
        "source": "huggingface",
        "repo_id": "speechbrain/spkrec-xvect-voxceleb",
        "description": "声纹特征提取模型",
    },
    "wav2vec2": {
        "name": "Wav2Vec2 Chinese",
        "local_name": "wav2vec2-large-xlsr-53-chinese-zh-cn",
        "source": "modelscope",
        "repo_id": "damo/speech_wav2vec2_large_xlsr_53-zh-cn",
        "description": "中文语音识别模型",
    },
}


@dataclass
class ModelDownloader:
    """模型下载器"""
    
    model_dir: Path = None
    
    def __post_init__(self):
        if self.model_dir is None:
            self.model_dir = Path("./models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def verify_models(self) -> Dict[str, bool]:
        """验证已下载的模型"""
        print("=" * 60)
        print("🔍 模型验证")
        print("=" * 60)
        
        results = {}
        
        # 检查 faster-whisper
        faster_path = self.model_dir / "faster-whisper-large-v3-turbo"
        results["faster-whisper"] = faster_path.exists() and any(faster_path.iterdir())
        
        # 检查 pyannote
        pyannote_path = self.model_dir / "pyannote" / "speaker-diarization-3.1"
        results["pyannote"] = pyannote_path.exists() and any(pyannote_path.iterdir())
        
        # 检查 speechbrain
        speechbrain_path = self.model_dir / "speechbrain" / "spkrec-xvect-voxceleb"
        results["speechbrain"] = speechbrain_path.exists() and any(speechbrain_path.iterdir())
        
        # 检查 wav2vec2
        wav2vec2_path = self.model_dir / "wav2vec2-large-xlsr-53-chinese-zh-cn"
        results["wav2vec2"] = wav2vec2_path.exists() and any(wav2vec2_path.iterdir())
        
        # 打印结果
        for name, exists in results.items():
            status = "✅ 已安装" if exists else "❌ 未安装"
            model_info = MODELS_INFO.get(name, {})
            print(f"  {name:20s} {model_info.get('name', '')[:30]:30s} {status}")
        
        print("=" * 60)
        
        all_ok = all(results.values())
        if all_ok:
            print("✅ 所有模型已就绪")
        else:
            print("⚠️ 部分模型缺失，请运行下载命令")
        
        return results
    
    def download_faster_whisper(self, use_hf_transfer: bool = False):
        """下载 Faster Whisper 模型"""
        print("\n📥 下载 Faster Whisper 模型...")
        print(f"   模型: {MODELS_INFO['faster-whisper']['repo_id']}")
        print(f"   路径: {self.model_dir / 'faster-whisper-large-v3-turbo'}")
        
        try:
            from huggingface_hub import snapshot_download
            
            model_path = snapshot_download(
                repo_id="Systran/faster-whisper-large-v3-turbo",
                cache_dir=str(self.model_dir),
                local_dir=str(self.model_dir / "faster-whisper-large-v3-turbo"),
                local_dir_use_symlinks=False,
            )
            print(f"✅ 下载完成: {model_path}")
            return True
            
        except ImportError:
            print("❌ 需要安装 huggingface_hub: pip install huggingface-hub")
            return False
        except Exception as e:
            print(f"❌ 下载失败: {e}")
            return False
    
    def download_pyannote(self, token: Optional[str] = None):
        """下载 PyAnnote 说话人分离模型
        
        Args:
            token: HuggingFace token (用于访问需要认证的模型)
        """
        print("\n📥 下载 PyAnnote 说话人分离模型...")
        print(f"   模型: {MODELS_INFO['pyannote']['repo_id']}")
        print(f"   路径: {self.model_dir / 'pyannote' / 'speaker-diarization-3.1'}")
        
        # 检查 token
        if not token:
            token = os.environ.get("HF_TOKEN")
        
        if not token:
            print("\n⚠️  PyAnnote 模型需要用户认证")
            print("   请先:")
            print("   1. 访问 https://hf.co/pyannote/speaker-diarization-3.1 接受用户协议")
            print("   2. 创建 token: https://hf.co/settings/tokens")
            print("   3. 使用以下方式之一:")
            print("      - python download_models.py --download pyannote --token YOUR_TOKEN")
            print("      - 设置环境变量: export HF_TOKEN=YOUR_TOKEN")
            return False
        
        try:
            from huggingface_hub import snapshot_download
            
            # 下载主模型
            print("📥 下载主模型...")
            model_path = snapshot_download(
                repo_id="pyannote/speaker-diarization-3.1",
                cache_dir=str(self.model_dir / "pyannote"),
                local_dir=str(self.model_dir / "pyannote" / "speaker-diarization-3.1"),
                local_dir_use_symlinks=False,
                token=token,
            )
            
            # 下载 pyannote/segmentation-3.0
            print("\n📥 下载 PyAnnote segmentation 模型...")
            snapshot_download(
                repo_id="pyannote/segmentation-3.0",
                cache_dir=str(self.model_dir / "pyannote"),
                local_dir_use_symlinks=False,
                token=token,
            )
            
            # 下载 pyannote/embedding-3.0
            print("📥 下载 PyAnnote embedding 模型...")
            snapshot_download(
                repo_id="pyannote/embedding-3.0",
                cache_dir=str(self.model_dir / "pyannote"),
                local_dir_use_symlinks=False,
                token=token,
            )
            
            print(f"✅ 下载完成")
            return True
            
        except ImportError:
            print("❌ 需要安装 huggingface_hub: pip install huggingface-hub")
            return False
        except Exception as e:
            error_msg = str(e)
            if "authentication" in error_msg.lower() or "gated" in error_msg.lower():
                print(f"❌ 认证失败，请检查 token 是否正确")
                print(f"   确保已在 https://hf.co/pyannote/speaker-diarization-3.1 接受用户协议")
            else:
                print(f"❌ 下载失败: {e}")
            return False
    
    def download_speechbrain(self):
        """下载 SpeechBrain 声纹模型"""
        print("\n📥 下载 SpeechBrain 声纹模型...")
        print(f"   模型: {MODELS_INFO['speechbrain']['repo_id']}")
        print(f"   路径: {self.model_dir / 'speechbrain' / 'spkrec-xvect-voxceleb'}")
        
        try:
            from huggingface_hub import snapshot_download
            
            model_path = snapshot_download(
                repo_id="speechbrain/spkrec-xvect-voxceleb",
                cache_dir=str(self.model_dir / "speechbrain"),
                local_dir=str(self.model_dir / "speechbrain" / "spkrec-xvect-voxceleb"),
                local_dir_use_symlinks=False,
            )
            print(f"✅ 下载完成: {model_path}")
            return True
            
        except ImportError:
            print("❌ 需要安装 huggingface_hub: pip install huggingface-hub")
            return False
        except Exception as e:
            print(f"❌ 下载失败: {e}")
            return False
    
    def download_all(self):
        """下载所有模型"""
        print("=" * 60)
        print("📥 开始下载所有模型")
        print("=" * 60)
        
        self.download_faster_whisper()
        self.download_pyannote()
        self.download_speechbrain()
        
        print("\n" + "=" * 60)
        print("✅ 所有模型下载完成")
        print("=" * 60)
        
        self.verify_models()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="模型下载和验证工具")
    parser.add_argument("--verify", "-v", action="store_true", help="验证已安装的模型")
    parser.add_argument("--download", "-d", choices=["all", "faster-whisper", "pyannote", "speechbrain"], 
                       help="下载指定模型")
    parser.add_argument("--model-dir", default="./models", help="模型目录")
    parser.add_argument("--token", "-t", default=None, help="HuggingFace token (用于下载 pyannote)")
    
    args = parser.parse_args()
    
    downloader = ModelDownloader(Path(args.model_dir))
    token = args.token or os.environ.get("HF_TOKEN")
    
    if args.verify:
        downloader.verify_models()
    elif args.download:
        if args.download == "all":
            downloader.download_all()
        elif args.download == "faster-whisper":
            downloader.download_faster_whisper()
        elif args.download == "pyannote":
            downloader.download_pyannote(token=token)
        elif args.download == "speechbrain":
            downloader.download_speechbrain()
    else:
        downloader.verify_models()
        
        print("\n使用方法:")
        print("  python download_models.py --verify      # 验证模型")
        print("  python download_models.py --download all # 下载所有模型")
        print("  python download_models.py --download faster-whisper")


if __name__ == "__main__":
    main()
