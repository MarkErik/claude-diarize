"""
Comparison Logic for Diarization Pipelines

Extracted from diarization_test.py, this module provides the comparison framework
for evaluating Granite Speech vs Whisper pipelines with comprehensive output saving.

Features:
- DiarizationEvaluator class for metrics calculation
- run_comparison_test() function for fair pipeline comparison
- Import and use new pipeline modules (granite_pipeline, whisper_pipeline)
- Comprehensive output saving with descriptive filenames
- Maintains fairness principles between pipelines

Installation:
pip install torch transformers pyannote.audio librosa peft torchaudio soundfile

For pyannote.audio, you need to:
1. Get HuggingFace token: https://huggingface.co/settings/tokens
2. Accept model terms: https://hf.co/pyannote/speaker-diarization-community-1
   and https://huggingface.co/pyannote/segmentation-3.0
"""

import torch
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import os
import gc
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import the new pipeline modules
from granite_pipeline import GraniteSpeechDiarizer
from whisper_pipeline import WhisperPipelineDiarizer


class DiarizationEvaluator:
    """Compare and evaluate diarization results with comprehensive output saving"""
    
    @staticmethod
    def calculate_metrics(
        hypothesis: List[Dict], 
        reference: List[Dict]
    ) -> Dict[str, float]:
        """
        Calculate diarization error metrics
        
        Metrics:
        - Word Error Rate (WER) for transcription
        - Diarization Error Rate (DER)
        - Speaker Confusion
        - Boundary accuracy
        """
        # This is a simplified version
        # Full implementation would use pyannote.metrics
        
        total_words = len(reference)
        correct_speaker = 0
        correct_transcription = 0
        
        for hyp, ref in zip(hypothesis, reference):
            if hyp['speaker'] == ref['speaker']:
                correct_speaker += 1
            if hyp['word'].lower() == ref['word'].lower():
                correct_transcription += 1
        
        return {
            'speaker_accuracy': correct_speaker / total_words if total_words > 0 else 0,
            'transcription_accuracy': correct_transcription / total_words if total_words > 0 else 0,
            'total_words': total_words
        }
    
    @staticmethod
    def format_output(segments: List[Dict], output_path: str):
        """Format and save results in multiple formats"""
        
        # JSON format
        with open(output_path + '.json', 'w') as f:
            json.dump(segments, f, indent=2)
        
        # Human-readable format
        with open(output_path + '.txt', 'w') as f:
            current_speaker = None
            current_text = []
            
            for seg in segments:
                if seg['speaker'] != current_speaker:
                    if current_text:
                        f.write(f"{current_speaker}: {' '.join(current_text)}\n\n")
                    current_speaker = seg['speaker']
                    current_text = []
                current_text.append(seg['word'])
            
            if current_text:
                f.write(f"{current_speaker}: {' '.join(current_text)}\n")
        
        # SRT subtitle format (with speakers)
        with open(output_path + '.srt', 'w') as f:
            for i, seg in enumerate(segments, 1):
                start_time = format_timestamp(seg['start'])
                end_time = format_timestamp(seg['end'])
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"[{seg['speaker']}] {seg['word']}\n\n")
    
    @staticmethod
    def save_comparison_results(
        granite_results: List[Dict],
        whisper_results: List[Dict],
        output_dir: str = "outputs"
    ):
        """
        Save comprehensive comparison results including individual pipeline results
        and comparison metrics
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save individual pipeline results
        if granite_results:
            with open(output_path / "comparison_granite_results.json", 'w') as f:
                json.dump(granite_results, f, indent=2)
            
            # Save formatted outputs for granite
            DiarizationEvaluator.format_output(
                granite_results,
                str(output_path / "comparison_granite_formatted")
            )
        
        if whisper_results:
            with open(output_path / "comparison_whisper_results.json", 'w') as f:
                json.dump(whisper_results, f, indent=2)
            
            # Save formatted outputs for whisper
            DiarizationEvaluator.format_output(
                whisper_results,
                str(output_path / "comparison_whisper_formatted")
            )
        
        # Calculate and save comparison metrics
        if granite_results and whisper_results:
            metrics = {
                'granite_stats': {
                    'total_words': len(granite_results),
                    'boundary_cases': len([s for s in granite_results if s.get('boundary_case', False)]),
                    'smoothed_words': len([s for s in granite_results if s.get('smoothed', False)]),
                    'avg_confidence': np.mean([s.get('confidence', 0) for s in granite_results])
                },
                'whisper_stats': {
                    'total_words': len(whisper_results),
                    'boundary_cases': len([s for s in whisper_results if s.get('boundary_case', False)]),
                    'smoothed_words': len([s for s in whisper_results if s.get('smoothed', False)]),
                    'avg_confidence': np.mean([s.get('confidence', 0) for s in whisper_results])
                },
                'comparison': {
                    'word_count_diff': abs(len(granite_results) - len(whisper_results)),
                    'granite_more_words': len(granite_results) > len(whisper_results),
                    'confidence_diff': (
                        np.mean([s.get('confidence', 0) for s in granite_results]) - 
                        np.mean([s.get('confidence', 0) for s in whisper_results])
                    )
                }
            }
            
            with open(output_path / "comparison_metrics.json", 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Save human-readable summary
            with open(output_path / "comparison_summary.txt", 'w') as f:
                f.write("DIARIZATION PIPELINE COMPARISON SUMMARY\n")
                f.write("=" * 50 + "\n\n")
                
                f.write("GRANITE SPEECH PIPELINE:\n")
                f.write("-" * 30 + "\n")
                f.write(f"  Total words: {metrics['granite_stats']['total_words']}\n")
                f.write(f"  Boundary cases: {metrics['granite_stats']['boundary_cases']}\n")
                f.write(f"  Smoothed words: {metrics['granite_stats']['smoothed_words']}\n")
                f.write(f"  Average confidence: {metrics['granite_stats']['avg_confidence']:.3f}\n")
                f.write(f"  Model: IBM Granite Speech 3.3-8B (8B parameters)\n")
                f.write(f"  Transcription: Optimized for enterprise/multilingual\n\n")
                
                f.write("WHISPER PIPELINE:\n")
                f.write("-" * 30 + "\n")
                f.write(f"  Total words: {metrics['whisper_stats']['total_words']}\n")
                f.write(f"  Boundary cases: {metrics['whisper_stats']['boundary_cases']}\n")
                f.write(f"  Smoothed words: {metrics['whisper_stats']['smoothed_words']}\n")
                f.write(f"  Average confidence: {metrics['whisper_stats']['avg_confidence']:.3f}\n")
                f.write(f"  Model: OpenAI Whisper large-v3 (1.5B parameters)\n")
                f.write(f"  Transcription: Battle-tested, widely used\n\n")
                
                f.write("COMPARISON:\n")
                f.write("-" * 30 + "\n")
                f.write(f"  Word count difference: {metrics['comparison']['word_count_diff']}\n")
                f.write(f"  {'Granite' if metrics['comparison']['granite_more_words'] else 'Whisper'} has more words\n")
                f.write(f"  Confidence difference: {metrics['comparison']['confidence_diff']:.3f}\n")
                f.write(f"  {'Granite' if metrics['comparison']['confidence_diff'] > 0 else 'Whisper'} has higher confidence\n\n")
                
                f.write("FAIR COMPARISON PRINCIPLES:\n")
                f.write("-" * 30 + "\n")
                f.write("  Both pipelines use identical forced alignment (Wav2Vec2)\n")
                f.write("  Both use pyannote.audio for diarization\n")
                f.write("  Both implement identical boundary handling algorithms\n")
                f.write("  Both apply the same speaker transition smoothing\n\n")
                
                f.write("OUTPUT FILES:\n")
                f.write("-" * 30 + "\n")
                f.write("  comparison_granite_results.json - Raw Granite results\n")
                f.write("  comparison_whisper_results.json - Raw Whisper results\n")
                f.write("  comparison_granite_formatted.json/txt/srt - Formatted Granite output\n")
                f.write("  comparison_whisper_formatted.json/txt/srt - Formatted Whisper output\n")
                f.write("  comparison_metrics.json - Detailed metrics and statistics\n")
                f.write("  comparison_summary.txt - This summary file\n")


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def run_comparison_test(audio_path: str, hf_token: str, output_dir: str = "outputs"):
    """
    Run fair comparison between Granite Speech and Whisper pipelines
    Both use the same multi-component approach for accurate comparison
    
    Args:
        audio_path: Path to the audio file
        hf_token: HuggingFace token for model access
        output_dir: Directory to save comparison results
        
    Returns:
        Tuple of (granite_results, whisper_results)
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    print("="*80)
    print("FAIR DIARIZATION COMPARISON TEST")
    print("="*80)
    print("\nBoth pipelines use:")
    print("  - Forced alignment for precise word timestamps")
    print("  - pyannote community-1 (latest model) for diarization")
    print("  - Intelligent boundary handling")
    print("\nThe only difference is the transcription model:")
    print("  Option A: Granite Speech 3.3-8B")
    print("  Option B: Whisper large-v3")
    print()
    
    # Test Option A: Granite Speech pipeline
    print("\n\nTesting OPTION A: Granite Speech Multi-Component Pipeline")
    print("-"*80)
    try:
        granite = GraniteSpeechDiarizer(hf_token, output_dir)
        granite_results = granite.transcribe_and_diarize(audio_path)
        
        print(f"✓ Granite pipeline completed: {len(granite_results)} words")
        
        # Show boundary cases
        granite_boundary = [s for s in granite_results if s.get('boundary_case', False)]
        print(f"  Boundary cases flagged: {len(granite_boundary)}")
        
    except Exception as e:
        print(f"✗ Granite pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        granite_results = None
    
    # Test Option B: Whisper pipeline
    print("\n\nTesting OPTION B: Whisper Multi-Component Pipeline")
    print("-"*80)
    try:
        whisper = WhisperPipelineDiarizer(hf_token, output_dir)
        whisper_results = whisper.transcribe_and_diarize(audio_path)
        
        print(f"✓ Whisper pipeline completed: {len(whisper_results)} words")
        
        # Show boundary cases
        whisper_boundary = [s for s in whisper_results if s.get('boundary_case', False)]
        print(f"  Boundary cases flagged: {len(whisper_boundary)}")
        
    except Exception as e:
        print(f"✗ Whisper pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        whisper_results = None
    
    # Save comprehensive comparison results
    if granite_results or whisper_results:
        print("\n\nSaving comparison results...")
        DiarizationEvaluator.save_comparison_results(
            granite_results, 
            whisper_results, 
            output_dir
        )
        print(f"✓ Comparison results saved to: {output_dir}/")
    
    # Summary
    print("\n\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    if granite_results and whisper_results:
        print(f"\nGranite Speech Pipeline:")
        print(f"  - Words: {len(granite_results)}")
        print(f"  - Boundary cases: {len([s for s in granite_results if s.get('boundary_case', False)])}")
        print(f"  - Model: IBM Granite Speech 3.3-8B (8B parameters)")
        print(f"  - Transcription: Optimized for enterprise/multilingual")
        
        print(f"\nWhisper Pipeline:")
        print(f"  - Words: {len(whisper_results)}")
        print(f"  - Boundary cases: {len([s for s in whisper_results if s.get('boundary_case', False)])}")
        print(f"  - Model: OpenAI Whisper large-v3 (1.5B parameters)")
        print(f"  - Transcription: Battle-tested, widely used")
        
        print(f"\nBoth pipelines use:")
        print(f"  - Forced alignment for word timestamps")
        print(f"  - pyannote community-1 for diarization")
        print(f"  - Same boundary handling algorithms")
        
        print(f"\nResults saved to: {output_dir}/")
        print(f"  - comparison_granite_results.json")
        print(f"  - comparison_whisper_results.json")
        print(f"  - comparison_metrics.json")
        print(f"  - comparison_summary.txt")
        print(f"\nCompare both outputs to see which transcription model")
        print(f"performs better for your specific audio characteristics.")
    
    return granite_results, whisper_results


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compare Granite Speech and Whisper diarization pipelines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python comparison_logic.py audio_file.mp3
  python comparison_logic.py /path/to/audio.wav --output custom_outputs
  python comparison_logic.py interview.mp3 --token your_hf_token
        """
    )
    
    parser.add_argument(
        "audio_file",
        help="Path to the audio file to process (MP3, WAV, etc.)"
    )
    
    parser.add_argument(
        "--output",
        "-o",
        default="outputs",
        help="Output directory for results (default: outputs)"
    )
    
    parser.add_argument(
        "--token",
        "-t",
        help="HuggingFace token (can also be set via HF_TOKEN environment variable)"
    )
    
    args = parser.parse_args()
    
    # Check if audio file exists
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file '{args.audio_file}' not found.")
        exit(1)
    
    # Get HuggingFace token from command line or environment
    HF_TOKEN = args.token or os.getenv("HF_TOKEN")
    
    if not HF_TOKEN:
        print("Error: HF_TOKEN not found.")
        print("Please provide your HuggingFace token via:")
        print("  1. Command line: --token your_token_here")
        print("  2. Environment variable: HF_TOKEN=your_token_here")
        print("  3. .env file: HF_TOKEN=your_token_here")
        print("\nYou can get your token from: https://huggingface.co/settings/tokens")
        exit(1)
    
    # Run comparison
    print(f"Processing audio file: {args.audio_file}")
    print(f"Output directory: {args.output}")
    print()
    
    granite_results, whisper_results = run_comparison_test(
        audio_path=args.audio_file,
        hf_token=HF_TOKEN,
        output_dir=args.output
    )