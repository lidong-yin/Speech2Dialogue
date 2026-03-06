"""音频处理核心模块"""
from pathlib import Path
from typing import Dict, Optional, Any
import torch
import numpy as np

from ..configs import ModelConfig, ProcessConfig
from ..utils.video import VideoProcessor
from ..utils.audio import AudioProcessor
from ..core.diarizer import SpeakerDiarizer
from ..utils.voiceprint import VoicePrintRecognizer


class OfflineAudioProcessor:
    """完全离线的音频处理类"""
    
    def __init__(self, 
                 model_dir: str = "./models",
                 device: Optional[str] = None,
                 compute_type: Optional[str] = None,
                 model_config: Optional[ModelConfig] = None,
                 process_config: Optional[ProcessConfig] = None):
        """
        初始化处理器
        
        Args:
            model_dir: 模型目录
            device: 设备 (cpu/cuda)
            compute_type: 计算类型 (int8/float16)
            model_config: 模型配置
            process_config: 处理配置
        """
        self.model_config = model_config or ModelConfig()
        self.process_config = process_config or ProcessConfig()
        
        self.model_dir = Path(model_dir)
        
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        if compute_type:
            self.compute_type = compute_type
        else:
            self.compute_type = "float16" if self.device == "cuda" else "int8"
        
        print(f"📱 设备: {self.device}, 计算类型: {self.compute_type}")
        print(f"📁 模型目录: {self.model_dir.absolute()}")
        
        # 模型实例
        self.whisper_model = None
        self.diarizer = None
        self.voiceprint_recognizer = None
        self.wav2vec2_model = None
        self.wav2vec2_processor = None
        self.active_model = None
        
        # 当前使用的模型名称
        self.current_model_name = None
        
        # 说话人声纹库 (用于聚类)
        self.speaker_embeddings = {}
    
    def load_faster_whisper(self, model_name: Optional[str] = None) -> bool:
        """加载 faster-whisper 模型"""
        model_name = model_name or self.model_config.FASTER_WHISPER_NAME
        model_path = self.model_config.FASTER_WHISPER_PATH
        
        if not model_path.exists():
            print(f"⚠ 模型不存在: {model_path}")
            return False
        
        print(f"🔄 加载 faster-whisper: {model_name}")
        
        try:
            from faster_whisper import WhisperModel
            
            self.whisper_model = WhisperModel(
                str(model_path),
                device=self.device,
                compute_type=self.compute_type,
                num_workers=2,
                cpu_threads=2,
                download_root=None
            )
            self.active_model = "faster-whisper"
            self.current_model_name = model_name
            print(f"✅ faster-whisper 加载成功")
            return True
            
        except Exception as e:
            print(f"❌ faster-whisper 加载失败: {e}")
            return False
    
    def load_whisperx(self, model_name: Optional[str] = None) -> bool:
        """加载 whisperx 模型"""
        model_name = model_name or self.model_config.WHISPERX_NAME
        model_path = self.model_config.WHISPERX_PATH
        
        if not model_path.exists():
            print(f"⚠ 模型不存在: {model_path}")
            return False
        
        print(f"🔄 加载 whisperx: {model_name}")
        
        try:
            import whisperx
            import os
            
            os.environ['WHISPER_MODEL_DIR'] = str(self.model_dir)
            
            self.whisper_model = whisperx.load_model(
                str(model_path),
                device=self.device,
                compute_type=self.compute_type,
                download_root=None
            )
            self.active_model = "whisperx"
            self.current_model_name = model_name
            print(f"✅ whisperx 加载成功")
            return True
            
        except Exception as e:
            print(f"❌ whisperx 加载失败: {e}")
            return False
    
    def load_wav2vec2(self, model_name: Optional[str] = None) -> bool:
        """加载 wav2vec2 模型"""
        model_name = model_name or self.model_config.WAV2VEC2_NAME
        model_path = self.model_config.WAV2VEC2_PATH
        
        if not model_path.exists():
            print(f"⚠ wav2vec2 模型不存在: {model_path}")
            return False
        
        print(f"🔄 加载 wav2vec2: {model_name}")
        
        try:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
            
            self.wav2vec2_model = Wav2Vec2ForCTC.from_pretrained(
                str(model_path),
                local_files_only=True
            )
            
            self.wav2vec2_processor = Wav2Vec2Processor.from_pretrained(
                str(model_path),
                local_files_only=True
            )
            
            if self.device == "cuda":
                self.wav2vec2_model = self.wav2vec2_model.cuda()
            
            self.wav2vec2_model.eval()
            self.active_model = "wav2vec2"
            self.current_model_name = model_name
            print(f"✅ wav2vec2 中文模型加载成功")
            return True
            
        except Exception as e:
            print(f"❌ wav2vec2 加载失败: {e}")
            return False
    
    def load_diarization(self, model_name: Optional[str] = None, token: Optional[str] = None) -> bool:
        """
        加载说话人分离模型
        
        Args:
            model_name: 模型名称
            token: HuggingFace token
        """
        model_path = self.model_config.PYANNOTE_PATH
        
        self.diarizer = SpeakerDiarizer(
            model_path=model_path,
            device=self.device
        )
        return self.diarizer.load_model(
            model_name or self.model_config.PYANNOTE_NAME,
            token=token
        )
    
    def load_voiceprint(self) -> bool:
        """
        加载声纹识别模型
        
        Returns:
            是否加载成功
        """
        print(f"🔄 加载声纹识别模型...")
        
        self.voiceprint_recognizer = VoicePrintRecognizer(
            model_dir=str(self.model_dir),
            device=self.device
        )
        
        result = self.voiceprint_recognizer.load_model()
        if result:
            print(f"✅ 声纹识别模型加载成功")
        return result
    
    def extract_speaker_embedding(self, audio_path: str, segment: Dict) -> Optional[np.ndarray]:
        """
        从音频片段提取声纹特征
        
        Args:
            audio_path: 音频文件路径
            segment: 音频片段信息
            
        Returns:
            声纹特征向量
        """
        if self.voiceprint_recognizer is None:
            return None
        
        try:
            import librosa
            
            # 加载完整音频
            y, sr = librosa.load(audio_path, sr=16000)
            
            # 提取片段
            start_sample = int(segment.get("start", 0) * sr)
            end_sample = int(segment.get("end", 0) * sr)
            
            if start_sample >= len(y) or start_sample >= end_sample:
                return None
            
            segment_audio = y[start_sample:min(end_sample, len(y))]
            
            # 提取声纹
            embedding = self.voiceprint_recognizer.extract_embedding_from_array(segment_audio, sr)
            return embedding
            
        except Exception as e:
            print(f"⚠ 声纹提取失败: {e}")
            return None
    
    def cluster_speakers_by_voice(self, result: Dict, audio_path: str, threshold: float = 0.75) -> Dict:
        """
        使用声纹特征对说话人进行聚类
        
        Args:
            result: 识别结果
            audio_path: 音频文件路径
            threshold: 相似度阈值 (0-1)
            
        Returns:
            聚类后的结果
        """
        if self.voiceprint_recognizer is None:
            print("⚠ 声纹模型未加载，跳过聚类")
            return result
        
        if not result.get("segments"):
            return result
        
        print("🔊 使用声纹特征进行说话人聚类...")
        
        # 为每个片段提取声纹
        for i, segment in enumerate(result["segments"]):
            embedding = self.extract_speaker_embedding(audio_path, segment)
            if embedding is not None:
                segment["_embedding"] = embedding
        
        # 聚类
        embeddings = []
        segment_indices = []
        
        for i, segment in enumerate(result["segments"]):
            if "_embedding" in segment:
                embeddings.append(segment["_embedding"])
                segment_indices.append(i)
        
        if len(embeddings) < 2:
            return result
        
        # 简单的层次聚类
        import numpy as np
        
        # 计算相似度矩阵
        n = len(embeddings)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                similarity_matrix[i][j] = self.voiceprint_recognizer.compare_speakers(
                    embeddings[i], embeddings[j]
                )
        
        # 基于相似度重新标记说话人
        speaker_id = 0
        visited = set()
        
        for i in range(n):
            if i in visited:
                continue
            
            # 找相似片段
            similar_indices = [i]
            for j in range(i + 1, n):
                if j not in visited and similarity_matrix[i][j] >= threshold:
                    similar_indices.append(j)
                    visited.add(j)
            
            # 标记相同说话人
            speaker_label = f"SPEAKER_{speaker_id:02d}"
            for idx in similar_indices:
                result["segments"][segment_indices[idx]]["speaker"] = speaker_label
            
            visited.add(i)
            speaker_label = f"SPEAKER_{speaker_id:02d}"
            speaker_id += 1
        
        # 清理临时数据
        for segment in result["segments"]:
            if "_embedding" in segment:
                del segment["_embedding"]
        
        print(f"✅ 声纹聚类完成，识别到 {speaker_id} 个独立说话人")
        return result
    
    def transcribe(self, audio_path: str, language: Optional[str] = None) -> Dict:
        """
        转录音频
        
        Args:
            audio_path: 音频路径
            language: 语言代码
            
        Returns:
            转录结果
        """
        if self.whisper_model is None and self.wav2vec2_model is None:
            raise ValueError("请先加载语音识别模型")
        
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        
        print(f"🎤 转录音频: {audio_path.name}")
        
        try:
            if self.active_model == "wav2vec2":
                return self._transcribe_wav2vec2(str(audio_path))
            elif self.active_model == "faster-whisper":
                return self._transcribe_faster_whisper(str(audio_path), language)
            else:
                return self._transcribe_whisperx(str(audio_path), language)
                
        except Exception as e:
            print(f"❌ 转录失败: {e}")
            raise
    
    def _transcribe_faster_whisper(self, audio_path: str, language: Optional[str] = None) -> Dict:
        """使用 faster-whisper 转录"""
        segments, info = self.whisper_model.transcribe(
            audio_path,
            language=language,
            beam_size=self.process_config.BEAM_SIZE,
            vad_filter=self.process_config.VAD_FILTER,
            word_timestamps=False
        )
        
        result_segments = []
        for segment in segments:
            result_segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })
        
        result = {
            "segments": result_segments,
            "language": getattr(info, 'language', language or "unknown")
        }
        
        print(f"✅ 转录完成: {len(result_segments)} 个片段")
        return result
    
    def _transcribe_whisperx(self, audio_path: str, language: Optional[str] = None) -> Dict:
        """使用 whisperx 转录"""
        import whisperx
        
        audio = whisperx.load_audio(audio_path)
        result = self.whisper_model.transcribe(audio, batch_size=4, language=language)
        return result
    
    def _transcribe_wav2vec2(self, audio_path: str) -> Dict:
        """使用 wav2vec2 转录"""
        print(f"🎤 使用 wav2vec2 转录")
        
        import librosa
        
        speech, sr = librosa.load(audio_path, sr=16000)
        
        inputs = self.wav2vec2_processor(
            speech, 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding=True
        )
        
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = self.wav2vec2_model(inputs.input_values).logits
        
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.wav2vec2_processor.batch_decode(predicted_ids)[0]
        
        result = {
            "segments": [{
                "start": 0.0,
                "end": len(speech) / sr,
                "text": transcription
            }],
            "language": "zh"
        }
        
        print(f"✅ wav2vec2 转录完成")
        return result
    
    def align_segments(self, result: Dict, audio_path: str) -> Dict:
        """对齐时间戳"""
        if not result.get("segments"):
            return result
        
        lang = result.get("language", "unknown")
        
        supported_langs = ["zh", "en"]
        if lang not in supported_langs:
            print(f"⚠ 语言 {lang} 不支持对齐，跳过")
            return result
        
        try:
            print("⏭️ 对齐跳过（使用原始时间戳）")
            return result
            
        except Exception as e:
            print(f"⚠ 对齐失败: {e}")
            return result
    
    def process_audio(
        self, 
        audio_path: str,
        use_model: str = "faster-whisper",
        language: Optional[str] = None,
        enable_diarization: bool = True,
        num_speakers: Optional[int] = None,
        enable_noise_reduction: bool = False,
        is_video: bool = False,
        enable_voiceprint: bool = False,
        voiceprint_threshold: float = 0.75
    ) -> Dict:
        """
        处理音频/视频文件
        
        Args:
            audio_path: 输入文件路径
            use_model: 使用的模型
            language: 语言代码
            enable_diarization: 是否启用说话人分离
            num_speakers: 说话人数量
            enable_noise_reduction: 是否启用降噪
            is_video: 是否是视频文件
            enable_voiceprint: 是否使用声纹识别辅助分离
            voiceprint_threshold: 声纹相似度阈值
            
        Returns:
            处理结果
        """
        input_path = Path(audio_path)
        temp_audio = None
        
        try:
            # Step 0: 视频处理
            if is_video or VideoProcessor.is_video(audio_path):
                print(f"\n[0/6] 检测到视频文件，提取音频...")
                temp_audio = VideoProcessor.extract_audio(audio_path)
                audio_path = temp_audio
                input_path = Path(audio_path)
            
            # Step 1: 降噪处理
            if enable_noise_reduction:
                print(f"\n[1/6] 降噪处理...")
                audio_path = AudioProcessor.reduce_noise(audio_path)
            
            # Step 2: 语音识别
            print(f"\n[2/6] 语音识别...")
            if use_model == "wav2vec2":
                if self.wav2vec2_model is None:
                    self.load_wav2vec2()
                result = self.transcribe(audio_path, language)
            else:
                if self.whisper_model is None:
                    if use_model == "faster-whisper":
                        self.load_faster_whisper()
                    else:
                        self.load_whisperx()
                result = self.transcribe(audio_path, language)
            
            if not result.get("segments"):
                raise ValueError("语音识别失败，无结果")
            
            # Step 3: 时间戳对齐
            print(f"\n[3/6] 时间戳对齐...")
            if use_model != "wav2vec2":
                result = self.align_segments(result, audio_path)
            
            # Step 4: 说话人分离 (pyannote)
            print(f"\n[4/6] 说话人分离...")
            if enable_diarization and self.diarizer:
                diarization = self.diarizer.diarize(audio_path, num_speakers)
                if diarization:
                    result = self.diarizer.assign_speakers(diarization, result)
            else:
                print("⏭️  跳过说话人分离")
            
            # Step 5: 声纹识别辅助 (可选)
            if enable_voiceprint:
                print(f"\n[5/6] 声纹识别辅助分离...")
                if self.voiceprint_recognizer is None:
                    self.load_voiceprint()
                
                if self.voiceprint_recognizer and self.voiceprint_recognizer.model:
                    result = self.cluster_speakers_by_voice(
                        result, 
                        audio_path, 
                        threshold=voiceprint_threshold
                    )
                else:
                    print("⚠️ 声纹模型未加载，跳过声纹聚类")
            else:
                print("⏭️  跳过声纹识别")
            
            # Step 6: 后处理
            print(f"\n[6/6] 后处理...")
            result = self._post_process(result)
            
            print(f"\n🎉 处理完成!")
            return result
            
        except Exception as e:
            print(f"\n❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()
            raise
            
        finally:
            if temp_audio and Path(temp_audio).exists():
                try:
                    Path(temp_audio).unlink()
                except:
                    pass
    
    def _post_process(self, result: Dict) -> Dict:
        """后处理"""
        if not result.get("segments"):
            return result
        
        for segment in result["segments"]:
            if "speaker" not in segment:
                segment["speaker"] = "未知"
        
        return result
