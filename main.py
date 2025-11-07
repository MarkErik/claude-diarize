"""
Main Entry Point for Claude Diarize Application

This module serves as the primary entry point for the speech diarization application,
providing a unified command-line interface to run individual pipelines (Granite or Whisper)
or compare both pipelines with comprehensive output options.

Features:
- Command-line interface with argparse
- Support for Granite Speech and Whisper pipelines
- Pipeline comparison mode
- Multiple output formats (JSON, TXT, SRT)
- Comprehensive error handling
- Progress feedback and user-friendly output
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional

# Import the new pipeline modules
from granite_pipeline import GraniteSpeechDiarizer
from whisper_pipeline import WhisperPipelineDiarizer
from combined_pipeline import CombinedGraniteWhisperDiarizer
from comparison_logic import run_comparison_test, DiarizationEvaluator


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments with comprehensive validation.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Speech Diarization Pipeline - Compare Granite Speech and Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run comparison between both pipelines (default)
  python main.py audio_file.mp3
  
  # Run only Granite pipeline
  python main.py interview.wav --pipeline granite
  
  # Run only Whisper pipeline with custom output
  python main.py podcast.mp3 --pipeline whisper --output results/
  
  # Run comparison with specific output formats
  python main.py audio.wav --pipeline compare --formats json,txt
  
  # Custom output directory and formats
  python main.py audio.mp3 --output custom_output/ --formats json,txt,srt

Pipeline Options:
  granite    - IBM Granite Speech 3.3-8B pipeline
  whisper    - OpenAI Whisper large-v3 pipeline
  combined   - Hybrid Granite-Whisper pipeline for maximum accuracy
  compare    - Run both pipelines and compare results (default)

Output Formats:
  json       - Machine-readable JSON format
  txt        - Human-readable text format
  srt        - Subtitle format with timestamps
        """
    )
    
    # Required argument: audio file path
    parser.add_argument(
        "--audio-file",
        required=True,
        help="Path to the audio file to process (MP3, WAV, etc.)"
    )
    
    # Optional: pipeline choice
    parser.add_argument(
        "--pipeline",
        choices=["granite", "whisper", "combined", "compare"],
        default="compare",
        help="Pipeline to run (default: compare)"
    )
    
    # Optional: output directory
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Output directory for results (default: outputs)"
    )
    
    # Optional: output formats
    parser.add_argument(
        "--formats",
        default="json,txt,srt",
        help="Comma-separated list of output formats (default: json,txt,srt)"
    )
    
    # Optional: HuggingFace token
    parser.add_argument(
        "--token",
        help="HuggingFace token (can also be set via HF_TOKEN environment variable)"
    )
    
    # Optional: number of speakers (for combined pipeline)
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=2,
        help="Number of speakers to detect (default: 2)"
    )
    
    # Optional: minimum confidence threshold
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence threshold for boundary cases (default: 0.5)"
    )
    
    # Set default number of speakers to 2
    parser.set_defaults(num_speakers=2)
    
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> bool:
    """
    Validate command-line arguments and check for common issues.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        bool: True if valid, False otherwise
    """
    # Check if audio file exists
    if not os.path.exists(args.audio_file):
        print(f"❌ Error: Audio file '{args.audio_file}' not found.")
        return False
    
    # Check if audio file is readable
    if not os.access(args.audio_file, os.R_OK):
        print(f"❌ Error: Audio file '{args.audio_file}' is not readable.")
        return False
    
    # Validate output formats
    valid_formats = {"json", "txt", "srt"}
    requested_formats = [f.strip().lower() for f in args.formats.split(",")]
    invalid_formats = set(requested_formats) - valid_formats
    
    if invalid_formats:
        print(f"❌ Error: Invalid output format(s): {', '.join(invalid_formats)}")
        print(f"Valid formats are: {', '.join(valid_formats)}")
        return False
    
    return True


def get_huggingface_token(args: argparse.Namespace) -> Optional[str]:
    """
    Get HuggingFace token from command line or environment variable.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Optional[str]: HuggingFace token or None if not found
    """
    # Try command line first, then environment variable
    token = args.token or os.getenv("HF_TOKEN")
    
    if not token:
        print("❌ Error: HuggingFace token not found.")
        print("Please provide your HuggingFace token via:")
        print("  1. Command line: --token your_token_here")
        print("  2. Environment variable: HF_TOKEN=your_token_here")
        print("  3. .env file: HF_TOKEN=your_token_here")
        print("\nYou can get your token from: https://huggingface.co/settings/tokens")
        return None
    
    return token


def create_output_directory(output_dir: str) -> bool:
    """
    Create output directory if it doesn't exist.
    
    Args:
        output_dir: Path to output directory
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Check if directory is writable
        if not os.access(output_path, os.W_OK):
            print(f"❌ Error: Output directory '{output_dir}' is not writable.")
            return False
            
        print(f"📁 Output directory: {output_path.absolute()}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating output directory '{output_dir}': {e}")
        return False


def save_formatted_outputs(
    results: List[Dict], 
    output_dir: str, 
    pipeline_name: str,
    formats: List[str]
) -> bool:
    """
    Save results in specified formats.
    
    Args:
        results: Pipeline results
        output_dir: Output directory path
        pipeline_name: Name of the pipeline for filename prefix
        formats: List of output formats
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        output_path = Path(output_dir)
        
        for fmt in formats:
            if fmt == "json":
                json_path = output_path / f"{pipeline_name}_results.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    import json
                    json.dump(results, f, indent=2, ensure_ascii=False, default=str)
                print(f"  ✅ Saved: {json_path.name}")
                
            elif fmt == "txt":
                txt_path = output_path / f"{pipeline_name}_results.txt"
                with open(txt_path, 'w', encoding='utf-8') as f:
                    current_speaker = None
                    current_text = []
                    
                    for seg in results:
                        if seg['speaker'] != current_speaker:
                            if current_text:
                                f.write(f"{current_speaker}: {' '.join(current_text)}\n\n")
                            current_speaker = seg['speaker']
                            current_text = []
                        current_text.append(seg['word'])
                    
                    if current_text:
                        f.write(f"{current_speaker}: {' '.join(current_text)}\n")
                print(f"  ✅ Saved: {txt_path.name}")
                
            elif fmt == "srt":
                srt_path = output_path / f"{pipeline_name}_results.srt"
                with open(srt_path, 'w', encoding='utf-8') as f:
                    for i, seg in enumerate(results, 1):
                        start_time = format_timestamp(seg['start'])
                        end_time = format_timestamp(seg['end'])
                        f.write(f"{i}\n")
                        f.write(f"{start_time} --> {end_time}\n")
                        f.write(f"[{seg['speaker']}] {seg['word']}\n\n")
                print(f"  ✅ Saved: {srt_path.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving formatted outputs: {e}")
        return False


def format_timestamp(seconds: float) -> str:
    """
    Convert seconds to SRT timestamp format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        str: Formatted timestamp (HH:MM:SS,mmm)
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def run_granite_pipeline(
    audio_path: str, 
    hf_token: str, 
    output_dir: str,
    formats: List[str]
) -> bool:
    """
    Run the Granite Speech pipeline.
    
    Args:
        audio_path: Path to audio file
        hf_token: HuggingFace token
        output_dir: Output directory
        formats: List of output formats
        
    Returns:
        bool: True if successful, False otherwise
    """
    print("\n🚀 Running Granite Speech Pipeline...")
    print("-" * 50)
    
    try:
        # Initialize pipeline
        diarizer = GraniteSpeechDiarizer(hf_token, output_dir)
        
        # Run transcription and diarization
        results = diarizer.transcribe_and_diarize(audio_path)
        
        if not results:
            print("❌ Granite pipeline produced no results.")
            return False
        
        # Save formatted outputs
        success = save_formatted_outputs(results, output_dir, "granite", formats)
        
        if success:
            print(f"\n✅ Granite pipeline completed successfully!")
            print(f"   - Total words processed: {len(results)}")
            boundary_cases = [s for s in results if s.get('boundary_case', False)]
            print(f"   - Boundary cases flagged: {len(boundary_cases)}")
            
            # Show speaker distribution
            speakers = {}
            for seg in results:
                speaker = seg.get('speaker', 'Unknown')
                speakers[speaker] = speakers.get(speaker, 0) + 1
            
            print(f"   - Speaker distribution:")
            for speaker, count in speakers.items():
                print(f"     {speaker}: {count} words")
        
        return success
        
    except Exception as e:
        print(f"❌ Granite pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_whisper_pipeline(
    audio_path: str, 
    hf_token: str, 
    output_dir: str,
    formats: List[str]
) -> bool:
    """
    Run the Whisper pipeline.
    
    Args:
        audio_path: Path to audio file
        hf_token: HuggingFace token
        output_dir: Output directory
        formats: List of output formats
        
    Returns:
        bool: True if successful, False otherwise
    """
    print("\n🚀 Running Whisper Pipeline...")
    print("-" * 50)
    
    try:
        # Initialize pipeline
        diarizer = WhisperPipelineDiarizer(hf_token, output_dir)
        
        # Run transcription and diarization
        results = diarizer.transcribe_and_diarize(audio_path)
        
        if not results:
            print("❌ Whisper pipeline produced no results.")
            return False
        
        # Save formatted outputs
        success = save_formatted_outputs(results, output_dir, "whisper", formats)
        
        if success:
            print(f"\n✅ Whisper pipeline completed successfully!")
            print(f"   - Total words processed: {len(results)}")
            boundary_cases = [s for s in results if s.get('boundary_case', False)]
            print(f"   - Boundary cases flagged: {len(boundary_cases)}")
            
            # Show speaker distribution
            speakers = {}
            for seg in results:
                speaker = seg.get('speaker', 'Unknown')
                speakers[speaker] = speakers.get(speaker, 0) + 1
            
            print(f"   - Speaker distribution:")
            for speaker, count in speakers.items():
                print(f"     {speaker}: {count} words")
        
        return success
        
    except Exception as e:
        print(f"❌ Whisper pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_combined_pipeline(
    audio_path: str,
    hf_token: str,
    output_dir: str,
    formats: List[str],
    num_speakers: int = 2,
    min_confidence: float = 0.5
) -> bool:
    """
    Run the combined Granite-Whisper pipeline.
    
    Args:
        audio_path: Path to audio file
        hf_token: HuggingFace token
        output_dir: Output directory
        formats: List of output formats
        num_speakers: Number of speakers to detect
        min_confidence: Minimum confidence threshold
        
    Returns:
        bool: True if successful, False otherwise
    """
    print("\n🚀 Running Combined Granite-Whisper Pipeline...")
    print("-" * 50)
    
    try:
        # Initialize pipeline
        diarizer = CombinedGraniteWhisperDiarizer(
            hf_token,
            output_dir,
            num_speakers=num_speakers,
            min_confidence_threshold=min_confidence
        )
        
        # Run transcription and diarization
        results = diarizer.transcribe_and_diarize(audio_path)
        
        if not results:
            print("❌ Combined pipeline produced no results.")
            return False
        
        # Save formatted outputs
        success = save_formatted_outputs(results, output_dir, "combined", formats)
        
        if success:
            print(f"\n✅ Combined pipeline completed successfully!")
            print(f"   - Total words processed: {len(results)}")
            high_confidence = [s for s in results if s['confidence'] > 0.8]
            boundary_cases = [s for s in results if s.get('boundary_case', False)]
            context_smoothed = [s for s in results if s.get('context_smoothed', False)]
            
            print(f"   - High confidence words (>0.8): {len(high_confidence)}")
            print(f"   - Boundary cases flagged: {len(boundary_cases)}")
            print(f"   - Context smoothed words: {len(context_smoothed)}")
            
            # Show speaker distribution
            speakers = {}
            for seg in results:
                speaker = seg.get('speaker', 'Unknown')
                speakers[speaker] = speakers.get(speaker, 0) + 1
            
            print(f"   - Speaker distribution:")
            for speaker, count in speakers.items():
                print(f"     {speaker}: {count} words")
        
        return success
        
    except Exception as e:
        print(f"❌ Combined pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_comparison_mode(
    audio_path: str,
    hf_token: str,
    output_dir: str,
    formats: List[str]
) -> bool:
    """
    Run comparison between all pipelines.
    
    Args:
        audio_path: Path to audio file
        hf_token: HuggingFace token
        output_dir: Output directory
        formats: List of output formats
        
    Returns:
        bool: True if successful, False otherwise
    """
    print("\n🔍 Running Pipeline Comparison...")
    print("-" * 50)
    
    try:
        # Run individual pipelines
        granite_success = run_granite_pipeline(audio_path, hf_token, output_dir, formats)
        whisper_success = run_whisper_pipeline(audio_path, hf_token, output_dir, formats)
        combined_success = run_combined_pipeline(audio_path, hf_token, output_dir, formats)
        
        if not granite_success and not whisper_success and not combined_success:
            print("❌ All pipelines failed. No comparison results available.")
            return False
        
        print(f"\n✅ Comparison completed successfully!")
        print(f"   Results saved to: {output_dir}/")
        
        # Show which pipelines succeeded
        succeeded = []
        if granite_success:
            succeeded.append("Granite")
        if whisper_success:
            succeeded.append("Whisper")
        if combined_success:
            succeeded.append("Combined")
        
        print(f"   Successful pipelines: {', '.join(succeeded)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Comparison mode failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    Main entry point for the application.
    """
    print("🎙️  Claude Diarize - Speech Diarization Pipeline")
    print("=" * 60)
    
    # Parse command-line arguments
    args = parse_arguments()
    
    # Validate arguments
    if not validate_arguments(args):
        sys.exit(1)
    
    # Get HuggingFace token
    hf_token = get_huggingface_token(args)
    if not hf_token:
        sys.exit(1)
    
    # Create output directory
    if not create_output_directory(args.output_dir):
        sys.exit(1)
    
    # Parse output formats
    formats = [f.strip().lower() for f in args.formats.split(",")]
    
    # Show configuration
    print(f"\n📋 Configuration:")
    print(f"   Audio file: {args.audio_file}")
    print(f"   Pipeline: {args.pipeline}")
    print(f"   Output directory: {args.output_dir}")
    print(f"   Output formats: {', '.join(formats)}")
    
    if args.pipeline == "combined":
        print(f"   Number of speakers: {args.num_speakers}")
        print(f"   Minimum confidence threshold: {args.min_confidence}")
    
    print()
    
    # Run the selected pipeline
    success = False
    
    try:
        if args.pipeline == "granite":
            success = run_granite_pipeline(args.audio_file, hf_token, args.output_dir, formats)
            
        elif args.pipeline == "whisper":
            success = run_whisper_pipeline(args.audio_file, hf_token, args.output_dir, formats)
            
        elif args.pipeline == "combined":
            success = run_combined_pipeline(
                args.audio_file,
                hf_token,
                args.output_dir,
                formats,
                args.num_speakers,
                args.min_confidence
            )
            
        elif args.pipeline == "compare":
            success = run_comparison_mode(args.audio_file, hf_token, args.output_dir, formats)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user.")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Final status
    if success:
        print(f"\n🎉 Processing completed successfully!")
        print(f"   Check the output directory: {args.output_dir}")
    else:
        print(f"\n💥 Processing failed. Check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
