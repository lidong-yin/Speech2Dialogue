"""
命令行入口模块

提供 CLI 接口给用户调用
"""
import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional

from .configs import (
    ModelConfig, ProcessConfig, OutputConfig, 
    DeviceConfig, SpeakerConfig, init_all_dirs
)
from .core.processor import OfflineAudioProcessor
from .core.exporter import DialogueExporter
from .utils.video import VideoProcessor


def find_audio_files(directory: str = ".") -> List[Path]:
    """查找音频/视频文件"""
    extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac', '.m4a', '.mp4', '.avi', '.mov', '.mkv'}
    
    audio_files = []
    for ext in extensions:
        audio_files.extend(Path(directory).glob(f"*{ext}"))
        audio_files.extend(Path(directory).glob(f"*{ext.upper()}"))
    
    return sorted(audio_files, key=lambda x: x.name)


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="📼 完全离线语音转对话脚本生成器 (支持视频输入)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python -m speech2dialogue audio.mp3
  python -m speech2dialogue video.mp4
  python -m speech2dialogue . --batch
  python -m speech2dialogue audio.mp3 --model wav2vec2 --denoise
        """
    )
    
    parser.add_argument(
        "input", 
        nargs="?", 
        help="音频/视频文件或目录路径"
    )
    parser.add_argument(
        "--model", 
        default="faster-whisper",
        choices=["faster-whisper", "whisperx", "wav2vec2"],
        help="使用的语音识别模型"
    )
    parser.add_argument(
        "--language", 
        help="语言代码 (zh, en, ja 等)"
    )
    parser.add_argument(
        "--speakers", 
        type=int, 
        help="说话人数量"
    )
    parser.add_argument(
        "--no-diarization", 
        action="store_true",
        help="不进行说话人分离"
    )
    parser.add_argument(
        "--output", 
        default="./outputs", 
        help="输出目录"
    )
    parser.add_argument(
        "--cpu", 
        action="store_true", 
        help="使用CPU"
    )
    parser.add_argument(
        "--batch", 
        action="store_true", 
        help="批量处理模式"
    )
    parser.add_argument(
        "--denoise", 
        action="store_true", 
        help="启用降噪"
    )
    parser.add_argument(
        "--model-dir",
        default="./models",
        help="模型目录"
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HuggingFace token (用于下载说话人分离模型)"
    )
    parser.add_argument(
        "--voiceprint",
        action="store_true",
        help="使用声纹识别辅助说话人分离 (基于 speechbrain)"
    )
    parser.add_argument(
        "--voiceprint-threshold",
        type=float,
        default=0.75,
        help="声纹相似度阈值 (0-1, 默认 0.75)"
    )
    
    return parser.parse_args(args)


def single_process(args: argparse.Namespace, processor: OfflineAudioProcessor):
    """单文件处理"""
    # 选择输入文件
    if args.input and Path(args.input).exists():
        audio_file = args.input
    else:
        audio_files = find_audio_files()
        if not audio_files:
            print("❌ 没有找到音视频文件")
            return
        
        if len(audio_files) == 1:
            audio_file = str(audio_files[0])
            print(f"📁 使用找到的文件: {audio_file}")
        else:
            print(f"📁 找到 {len(audio_files)} 个音视频文件:")
            for i, f in enumerate(audio_files[:10], 1):
                print(f"  {i:2d}. {f.name}")
            
            if len(audio_files) > 10:
                print(f"  ... 还有 {len(audio_files)-10} 个")
            
            try:
                choice = int(input(f"\n请选择 (1-{min(10, len(audio_files))}): "))
                if 1 <= choice <= len(audio_files):
                    audio_file = str(audio_files[choice-1])
                else:
                    print("❌ 无效选择")
                    return
            except:
                print("❌ 无效输入")
                return
    
    is_video = VideoProcessor.is_video(audio_file)
    
    try:
        result = processor.process_audio(
            audio_path=audio_file,
            use_model=args.model,
            language=args.language,
            enable_diarization=not args.no_diarization,
            num_speakers=args.speakers,
            enable_noise_reduction=args.denoise,
            is_video=is_video,
            enable_voiceprint=args.voiceprint,
            voiceprint_threshold=args.voiceprint_threshold
        )
        
        if not result.get("segments"):
            print("❌ 没有生成对话")
            return
        
        # 使用配置中的说话人映射
        speaker_config = SpeakerConfig()
        dialogues = DialogueExporter.format_dialogues(
            result,
            speaker_mapping=speaker_config.SPEAKER_MAPPING,
            merge_gap=ProcessConfig.MERGE_GAP_THRESHOLD
        )
        
        if not dialogues:
            print("❌ 没有可输出的对话")
            return
        
        base_name = Path(audio_file).stem
        saved_files = DialogueExporter.save_all_formats(
            dialogues, base_name, args.output
        )
        
        DialogueExporter.print_preview(dialogues)
        
        print(f"\n📊 统计信息:")
        print(f"   文件: {Path(audio_file).name}")
        print(f"   总时长: {dialogues[-1]['end']:.1f} 秒")
        print(f"   对话片段: {len(dialogues)} 个")
        print(f"   输出文件: {len(saved_files)} 个")
        print(f"   输出目录: {Path(args.output).absolute()}")
        
        print(f"\n🎉 处理完成!")
        
    except Exception as e:
        print(f"\n❌ 处理失败: {e}")


def batch_process(args: argparse.Namespace, processor: OfflineAudioProcessor):
    """批量处理"""
    input_dir = args.input if args.input else "."
    audio_files = find_audio_files(input_dir)
    
    if not audio_files:
        print(f"❌ 在 {input_dir} 中没有找到音视频文件")
        return
    
    print(f"📁 找到 {len(audio_files)} 个音视频文件")
    
    success_count = 0
    speaker_config = SpeakerConfig()
    
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(audio_files)}] 处理: {audio_file.name}")
        
        is_video = VideoProcessor.is_video(str(audio_file))
        
        try:
            result = processor.process_audio(
                audio_path=str(audio_file),
                use_model=args.model,
                language=args.language,
                enable_diarization=not args.no_diarization,
                num_speakers=args.speakers,
                enable_noise_reduction=args.denoise,
                is_video=is_video,
                enable_voiceprint=args.voiceprint,
                voiceprint_threshold=args.voiceprint_threshold
            )
            
            if result.get("segments"):
                dialogues = DialogueExporter.format_dialogues(
                    result, 
                    speaker_mapping=speaker_config.SPEAKER_MAPPING,
                    merge_gap=ProcessConfig.MERGE_GAP_THRESHOLD
                )
                
                if dialogues:
                    base_name = audio_file.stem
                    DialogueExporter.save_all_formats(dialogues, base_name, args.output)
                    
                    print(f"✅ 成功: {len(dialogues)} 个对话片段")
                    success_count += 1
                    
        except Exception as e:
            print(f"❌ 失败: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"📊 批量处理完成!")
    print(f"   总文件: {len(audio_files)} 个")
    print(f"   成功: {success_count} 个")
    print(f"   失败: {len(audio_files) - success_count} 个")


def main(args: Optional[List[str]] = None):
    """主函数"""
    args = args or sys.argv[1:]
    
    # 如果没有参数，显示帮助
    if not args:
        print("请提供输入文件或使用 --batch 参数")
        print("示例: python -m speech2dialogue audio.mp3")
        print("帮助: python -m speech2dialogue --help")
        return
    
    parsed_args = parse_args(args)
    
    # 初始化目录
    init_all_dirs()
    Path(parsed_args.output).mkdir(parents=True, exist_ok=True)
    
    print(f"{'='*60}")
    print(f"🎵 语音转对话脚本生成器 V1.0")
    print(f"{'='*60}")
    
    # 创建设备配置
    device_config = DeviceConfig.from_args(cpu=parsed_args.cpu)
    
    # 创建处理器
    processor = OfflineAudioProcessor(
        model_dir=parsed_args.model_dir,
        device=device_config.DEVICE,
        compute_type=device_config.COMPUTE_TYPE,
        model_config=ModelConfig(),
        process_config=ProcessConfig()
    )
    
    print(f"\n📦 加载模型...")
    
    # 加载语音识别模型
    if parsed_args.model == "faster-whisper":
        if not processor.load_faster_whisper():
            print("⚠ 尝试加载 whisperx...")
            processor.load_whisperx()
    elif parsed_args.model == "whisperx":
        processor.load_whisperx()
    elif parsed_args.model == "wav2vec2":
        processor.load_wav2vec2()
    
    # 加载说话人分离模型
    if not parsed_args.no_diarization:
        # 优先使用命令行传入的 token，其次使用环境变量
        token = parsed_args.token or os.environ.get("HF_TOKEN")
        if token:
            print(f"🔑 使用 HuggingFace token")
        processor.load_diarization(token=token)
    
    # 执行处理
    if parsed_args.batch or (parsed_args.input and Path(parsed_args.input).is_dir()):
        batch_process(parsed_args, processor)
    else:
        single_process(parsed_args, processor)


if __name__ == "__main__":
    main()
