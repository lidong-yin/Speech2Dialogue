"""对话导出模块"""
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional


class DialogueExporter:
    """对话导出器"""
    
    @staticmethod
    def format_dialogues(
        result: Dict, 
        speaker_mapping: Optional[Dict[str, str]] = None,
        merge_gap: float = 0.8
    ) -> List[Dict]:
        """
        格式化对话数据
        
        Args:
            result: 识别结果
            speaker_mapping: 说话人映射
            merge_gap: 合并间隔(秒)
            
        Returns:
            对话列表
        """
        if not result.get("segments"):
            return []
        
        dialogues = []
        
        for segment in result["segments"]:
            speaker = segment.get("speaker", "未知")
            text = segment.get("text", "").strip()
            
            if not text:
                continue
            
            if speaker_mapping and speaker in speaker_mapping:
                speaker = speaker_mapping[speaker]
            
            dialogue = {
                "speaker": speaker,
                "text": text,
                "start": round(float(segment.get("start", 0)), 2),
                "end": round(float(segment.get("end", 0)), 2),
                "duration": round(
                    float(segment.get("end", 0)) - float(segment.get("start", 0)), 
                    2
                )
            }
            dialogues.append(dialogue)
        
        if merge_gap > 0 and len(dialogues) > 1:
            dialogues = DialogueExporter._merge_consecutive(dialogues, merge_gap)
        
        return dialogues
    
    @staticmethod
    def _merge_consecutive(dialogues: List[Dict], max_gap: float) -> List[Dict]:
        """合并相邻的同一说话人片段"""
        if not dialogues:
            return []
            
        merged = [dialogues[0].copy()]
        
        for i in range(1, len(dialogues)):
            current = dialogues[i]
            last = merged[-1]
            
            if (current["speaker"] == last["speaker"] and 
                current["start"] - last["end"] <= max_gap):
                last["text"] += " " + current["text"]
                last["end"] = current["end"]
                last["duration"] = last["end"] - last["start"]
            else:
                merged.append(current.copy())
        
        return merged
    
    @staticmethod
    def save_all_formats(
        dialogues: List[Dict], 
        base_name: str, 
        output_dir: str = "./outputs"
    ) -> List[Path]:
        """
        保存所有格式
        
        Args:
            dialogues: 对话列表
            base_name: 基础文件名
            output_dir: 输出目录
            
        Returns:
            保存的文件路径列表
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        formats = {
            "json": DialogueExporter.save_json,
            "txt": DialogueExporter.save_txt,
            "clean": DialogueExporter.save_clean,
            "csv": DialogueExporter.save_csv,
            "srt": DialogueExporter.save_srt,
            "vtt": DialogueExporter.save_vtt
        }
        
        saved_files = []
        for fmt, save_func in formats.items():
            try:
                file_path = save_func(dialogues, base_name, output_dir)
                saved_files.append(file_path)
            except Exception as e:
                print(f"⚠ 保存 {fmt} 格式失败: {e}")
        
        return saved_files
    
    @staticmethod
    def save_json(dialogues: List[Dict], base_name: str, output_dir: Path) -> Path:
        """保存JSON格式"""
        file_path = output_dir / f"{base_name}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(dialogues, f, ensure_ascii=False, indent=2)
        print(f"📄 JSON: {file_path.name}")
        return file_path
    
    @staticmethod
    def save_txt(dialogues: List[Dict], base_name: str, output_dir: Path) -> Path:
        """保存带时间戳的文本格式"""
        file_path = output_dir / f"{base_name}.txt"
        lines = []
        for d in dialogues:
            time_str = f"[{d['start']:06.2f}-{d['end']:06.2f}]"
            lines.append(f"{time_str} {d['speaker']}: {d['text']}")
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"📄 文本: {file_path.name}")
        return file_path
    
    @staticmethod
    def save_clean(dialogues: List[Dict], base_name: str, output_dir: Path) -> Path:
        """保存纯对话文本"""
        file_path = output_dir / f"{base_name}_clean.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            for d in dialogues:
                f.write(f"{d['speaker']}: {d['text']}\n")
        print(f"📄 纯文本: {file_path.name}")
        return file_path
    
    @staticmethod
    def save_csv(dialogues: List[Dict], base_name: str, output_dir: Path) -> Path:
        """保存CSV格式"""
        file_path = output_dir / f"{base_name}.csv"
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["开始时间", "结束时间", "说话人", "文本", "时长"])
            for d in dialogues:
                writer.writerow([
                    f"{d['start']:.2f}", 
                    f"{d['end']:.2f}", 
                    d["speaker"], 
                    d["text"], 
                    f"{d['duration']:.2f}"
                ])
        print(f"📄 CSV: {file_path.name}")
        return file_path
    
    @staticmethod
    def save_srt(dialogues: List[Dict], base_name: str, output_dir: Path) -> Path:
        """保存SRT字幕格式"""
        file_path = output_dir / f"{base_name}.srt"
        
        def sec_to_srt(seconds: float) -> str:
            hrs = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            msecs = int((seconds - int(seconds)) * 1000)
            return f"{hrs:02d}:{mins:02d}:{secs:02d},{msecs:03d}"
        
        lines = []
        for i, d in enumerate(dialogues, 1):
            start_time = sec_to_srt(d["start"])
            end_time = sec_to_srt(d["end"])
            
            lines.append(str(i))
            lines.append(f"{start_time} --> {end_time}")
            lines.append(f"{d['speaker']}: {d['text']}")
            lines.append("")
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"📄 SRT字幕: {file_path.name}")
        return file_path
    
    @staticmethod
    def save_vtt(dialogues: List[Dict], base_name: str, output_dir: Path) -> Path:
        """保存VTT字幕格式"""
        file_path = output_dir / f"{base_name}.vtt"
        
        def sec_to_vtt(seconds: float) -> str:
            hrs = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            msecs = int((seconds - int(seconds)) * 1000)
            return f"{hrs:02d}:{mins:02d}:{secs:02d}.{msecs:03d}"
        
        lines = ["WEBVTT", ""]
        
        for i, d in enumerate(dialogues, 1):
            start_time = sec_to_vtt(d["start"])
            end_time = sec_to_vtt(d["end"])
            
            lines.append(str(i))
            lines.append(f"{start_time} --> {end_time}")
            lines.append(f"{d['speaker']}: {d['text']}")
            lines.append("")
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"📄 VTT字幕: {file_path.name}")
        return file_path
    
    @staticmethod
    def print_preview(dialogues: List[Dict], max_lines: int = 15):
        """打印对话预览"""
        if not dialogues:
            print("📭 没有对话内容")
            return
        
        print(f"\n{'='*80}")
        print(f"对话预览 (共 {len(dialogues)} 个片段):")
        print(f"{'='*80}")
        
        for i, d in enumerate(dialogues[:max_lines], 1):
            print(f"{i:3d}. [{d['start']:6.1f}s] {d['speaker']}: {d['text']}")
        
        if len(dialogues) > max_lines:
            print(f"... 还有 {len(dialogues)-max_lines} 个片段")
        
        print(f"{'='*80}")
