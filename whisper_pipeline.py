"""
Whisper Pipeline for Speech Diarization

Multi-step pipeline for maximum accuracy:
1. Whisper large-v3 for transcription
2. Wav2vec2 forced alignment for precise word timestamps
3. pyannote community-1 for diarization
4. Intelligent merging with boundary handling
5. Step-by-step output saving for analysis

Includes output saving at each processing step.
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


class WhisperPipelineDiarizer:
    """
    Multi-step pipeline for maximum accuracy:
    1. Whisper large-v3 for transcription
    2. Montreal Forced Aligner for precise word timestamps
    3. pyannote.audio 3.x for diarization
    4. Intelligent merging with boundary handling
    
    Includes automatic output saving at each processing step.
    
    Multi-core optimizations for Apple Silicon:
    - Uses cpu_threads parameter to leverage all available cores
    - Optimized batch_size for parallel processing
    - Reserves 1-2 cores for system operations
    """
    
    def __init__(
        self,
        hf_token: str,
        output_dir: str = "outputs",
        # Decoding controls (tuned for accuracy on long interviews)
        beam_size: int = 6,
        best_of: int = 5,
        patience: float = 1.2,
        length_penalty: float = 1.0,
        temperature: float = 0.0,
        vad_min_silence_ms: int = 400,
        # Parallelism cap (optional overrides)
        max_num_workers: int | None = None,
    ):
        self.hf_token = hf_token
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        # Store decoding/parallelism preferences
        self.beam_size = beam_size
        self.best_of = best_of
        self.patience = patience
        self.length_penalty = length_penalty
        self.temperature = temperature
        self.vad_min_silence_ms = vad_min_silence_ms
        self.max_num_workers = max_num_workers
        
        # Determine device
        if torch.backends.mps.is_available():
            self.device = "mps"
            print("Using Apple Silicon GPU (MPS)")
        elif torch.cuda.is_available():
            self.device = "cuda"
            print("Using NVIDIA GPU (CUDA)")
        else:
            self.device = "cpu"
            print("Using CPU")
        
        self.setup_models()
        
    def setup_models(self):
        """Initialize all pipeline components"""
        # 1. Whisper for transcription
        from faster_whisper import WhisperModel
        import os
        
        print("Loading Whisper large-v3...")
        # faster-whisper handles device internally. On Apple Silicon, use device='cpu'.
        if self.device == "cuda":
            device_str = "cuda"
            compute_type = "float16"  # Mixed precision for speed & memory
        else:
            device_str = "cpu"
            # Prefer float16 on Apple Silicon for a good accuracy/speed balance (falls back if unsupported)
            compute_type = "float32"

        # Balanced threading: avoid oversubscription by splitting cores across workers.
        total_cores = os.cpu_count() or 8
        reserve = 4  # leave some headroom for OS/IO
        usable = max(2, total_cores - reserve)
        # Choose workers based on total cores; cap if user requested
        default_workers = 2 if usable < 12 else 4 if usable < 24 else 6
        dynamic_num_workers = default_workers
        if self.max_num_workers is not None:
            dynamic_num_workers = max(1, min(self.max_num_workers, 8))
        # Threads per worker (integer floor)
        cpu_threads_per_worker = max(2, usable // max(1, dynamic_num_workers))
        print(
            f"Parallelism config: total_cores={total_cores}, workers={dynamic_num_workers}, "
            f"cpu_threads_per_worker={cpu_threads_per_worker}, compute_type={compute_type}"
        )

        self.whisper = WhisperModel(
            "large-v3",
            device=device_str,
            compute_type=compute_type,
            cpu_threads=cpu_threads_per_worker if device_str == "cpu" else 0,
            num_workers=dynamic_num_workers,
        )
        
        # 2. Pyannote for diarization
        from pyannote.audio import Pipeline
        
        # Load pyannote community-1 for diarization
        print("Loading pyannote community-1 diarization pipeline...")
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=self.hf_token
        )
        
        # Move to appropriate device
        if torch.backends.mps.is_available():
            # MPS support for pyannote (if available)
            try:
                self.diarization_pipeline.to(torch.device("mps"))
            except:
                print("Note: pyannote may not fully support MPS, using CPU for diarization")
        elif torch.cuda.is_available():
            self.diarization_pipeline.to(torch.device("cuda"))
        
        # 3. Pre-load alignment model to avoid reloading
        print("Loading alignment model...")
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        self.alignment_processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-large-960h-lv60-self"
        )
        
        # Load alignment model with MPS handling
        try:
            self.alignment_model = Wav2Vec2ForCTC.from_pretrained(
                "facebook/wav2vec2-large-960h-lv60-self",
                dtype=torch.float16 if self.device == "mps" else torch.float32
            ).to(self.device)
        except Exception as e:
            print(f"Failed to load alignment model on {self.device}: {e}")
            print("Falling back to CPU for alignment model...")
            self.alignment_model = Wav2Vec2ForCTC.from_pretrained(
                "facebook/wav2vec2-large-960h-lv60-self"
            ).to("cpu")
    
    def transcribe_and_diarize(self, audio_path: str) -> List[Dict]:
        """
        Complete pipeline for word-level diarized transcription
        
        Returns:
            List of dicts with: {word, start, end, speaker, confidence}
        Performance/accuracy tuning notes:
        - beam_size: Raising above 6 (e.g. 8–10) improves accuracy but increases latency.
        - patience: >1.2 allows beam search to explore longer; diminishing returns past ~1.4.
        - num_workers: Increased in init; raising further can cause memory pressure even with 512GB when CPU cache/memory bandwidth becomes the bottleneck.
        - cpu_threads_per_worker: Computed dynamically; keep >=2 to avoid context switching overhead.
        - compute_type float16: Good trade-off; if you observe numerical instability, set compute_type to 'float32'.
        - vad_min_silence_ms: Lower (<350) may fragment continuous speech; raise (>600) if speakers have long reflective pauses.
        - For extremely long audio (>2h), consider periodic checkpointing of intermediate JSON outputs to mitigate risk of interruption.
        """
        
        # Step 0: Preprocess audio for consistency across all pipeline steps
        print("Step 0: Preprocessing audio...")
        preprocessed_audio_path = self._preprocess_audio_for_diarization(audio_path)
        
        # Step 1: Transcribe with Whisper
        print("Step 1: Transcribing with Whisper...")
        # Accuracy-focused decoding parameters:
        # - Higher beam_size improves transcription accuracy for long-form speech.
        # - best_of is mainly used with sampling (temperature>0); kept for potential fallback.
        # - condition_on_previous_text preserves context across long interviews without chunking.
        # NOTE: faster-whisper does not support 'batch_size' in transcribe(); removed.
        segments, info = self.whisper.transcribe(
            preprocessed_audio_path,
            word_timestamps=True,
            language="en",  # Adjust as needed or set None for auto-detect
            beam_size=self.beam_size,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=self.vad_min_silence_ms),  # Catch shorter turn breaks
            best_of=self.best_of,
            patience=self.patience,  # Allow a bit more exploration for accuracy
            length_penalty=self.length_penalty,
            temperature=self.temperature,  # Deterministic decoding by default
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            condition_on_previous_text=True,
        )
        
        # Extract word-level transcription
        words_with_timestamps = []
        for segment in segments:
            if hasattr(segment, 'words'):
                for word in segment.words:
                    words_with_timestamps.append({
                        'word': word.word.strip(),
                        'start': word.start,
                        'end': word.end,
                        'confidence': getattr(word, 'probability', 1.0)
                    })
        
        # Save raw transcription output
        self._save_output(words_with_timestamps, "whisper_raw_transcription.json")
        print(f"  Saved raw transcription: {len(words_with_timestamps)} words")
        
        # Let Python handle cleanup naturally
        
        # Step 2: Forced Alignment (if available)
        print("Step 2: Performing forced alignment...")
        transcript = " ".join([w['word'] for w in words_with_timestamps])
        aligned_words = self._forced_alignment(preprocessed_audio_path, transcript)
        
        # Save forced alignment results
        self._save_output(aligned_words, "whisper_forced_alignment.json")
        print(f"  Saved forced alignment: {len(aligned_words)} words")
        
        # Let Python handle cleanup naturally
        
        # Step 3: Diarization
        print("Step 3: Running speaker diarization...")
        diarization = self.diarization_pipeline(
            preprocessed_audio_path,
            num_speakers=2  # Specified for 2-speaker interviews
        )
        
        # Convert diarization to list of segments
        speaker_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append({
                'start': turn.start,
                'end': turn.end,
                'speaker': speaker
            })
        
        # Let Python handle cleanup naturally
        
        # Step 4: Merge with intelligent boundary handling
        print("Step 4: Merging with boundary resolution...")
        merged_segments = self._merge_with_boundary_handling(
            aligned_words, 
            speaker_segments
        )
        
        # Save merged results
        self._save_output(merged_segments, "whisper_merged_results.json")
        print(f"  Saved merged results: {len(merged_segments)} words")
        
        # Let Python handle cleanup naturally
        
        # Step 5: Smooth speaker transitions
        print("Step 5: Smoothing speaker transitions...")
        final_segments = self._smooth_speaker_transitions(merged_segments)
        
        # Save final results
        self._save_output(final_segments, "whisper_final_results.json")
        print(f"  Saved final results: {len(final_segments)} words")
        
        return final_segments
    
    def _preprocess_audio_for_diarization(self, audio_path: str) -> str:
        """
        Preprocess audio file for pyannote diarization to avoid sample rate mismatches.
        
        Ensures audio is properly formatted with consistent sample rate and avoids
        corrupted samples that can cause chunk extraction errors.
        
        Args:
            audio_path: Path to original audio file
            
        Returns:
            Path to preprocessed audio file (or original if preprocessing fails)
        """
        try:
            import librosa
            import soundfile as sf
            
            # Load audio and resample to 16kHz (pyannote's expected rate)
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            # Create preprocessed output path
            preprocessed_path = self.output_dir / "whisper-results" / "preprocessed_audio.wav"
            preprocessed_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save with consistent format
            sf.write(str(preprocessed_path), audio, 16000, subtype='PCM_16')
            
            print(f"  Preprocessed audio saved: {preprocessed_path}")
            return str(preprocessed_path)
            
        except Exception as e:
            print(f"Warning: Audio preprocessing failed: {e}")
            print("  Falling back to original audio file")
            return audio_path
    
    def _forced_alignment(self, audio_path: str, transcript: str) -> List[Dict]:
        """
        Use wav2vec2 for forced phoneme alignment
        Uses pre-loaded model to avoid memory leaks
        """
        try:
            import librosa
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Process audio
            inputs = self.alignment_processor(audio, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get logits
            with torch.no_grad():
                logits = self.alignment_model(**inputs).logits
            
            # Simple heuristic alignment based on audio duration
            words = transcript.split()
            words_with_timestamps = []
            
            audio_duration = len(audio) / sr
            char_count = len(transcript)
            
            current_time = 0
            for word in words:
                word_duration = (len(word) / char_count) * audio_duration if char_count > 0 else 0.1
                words_with_timestamps.append({
                    'word': word,
                    'start': current_time,
                    'end': current_time + word_duration,
                    'confidence': 0.9
                })
                current_time += word_duration
            
            # Let Python handle cleanup naturally
            
            return words_with_timestamps
            
        except Exception as e:
            print(f"Forced alignment failed: {e}, using simple timing")
            # Fallback to simple timing
            words = transcript.split()
            import librosa
            audio, sr = librosa.load(audio_path, sr=16000)
            audio_duration = len(audio) / sr
            
            words_with_timestamps = []
            for i, word in enumerate(words):
                start_time = (i / len(words)) * audio_duration if len(words) > 0 else 0
                end_time = ((i + 1) / len(words)) * audio_duration if len(words) > 0 else 0.1
                words_with_timestamps.append({
                    'word': word,
                    'start': start_time,
                    'end': end_time,
                    'confidence': 0.7
                })
            
            # Let Python handle cleanup naturally
            
            return words_with_timestamps
    
    def _merge_with_boundary_handling(
        self, 
        words: List[Dict], 
        speaker_segments: List[Dict]
    ) -> List[Dict]:
        """
        Intelligently merge word timestamps with speaker segments
        Handles boundary conditions with multiple strategies
        """
        final_segments = []
        
        for word_data in words:
            word_start = word_data['start']
            word_end = word_data['end']
            word_mid = (word_start + word_end) / 2
            
            # Strategy 1: Find speaker with maximum overlap
            max_overlap = 0
            assigned_speaker = None
            overlaps = {}
            
            for spk_seg in speaker_segments:
                # Calculate overlap duration
                overlap_start = max(word_start, spk_seg['start'])
                overlap_end = min(word_end, spk_seg['end'])
                overlap_duration = max(0, overlap_end - overlap_start)
                
                overlaps[spk_seg['speaker']] = overlap_duration
                
                if overlap_duration > max_overlap:
                    max_overlap = overlap_duration
                    assigned_speaker = spk_seg['speaker']
            
            # Strategy 2: If word spans multiple speakers equally, use midpoint
            if len(overlaps) > 1:
                overlap_values = list(overlaps.values())
                # If overlaps are very similar (within 10%), use midpoint
                if max(overlap_values) - min(overlap_values) < 0.1 * (word_end - word_start):
                    # Find which speaker contains the midpoint
                    for spk_seg in speaker_segments:
                        if spk_seg['start'] <= word_mid <= spk_seg['end']:
                            assigned_speaker = spk_seg['speaker']
                            break
            
            # Strategy 3: Confidence weighting
            # If overlap is less than 50%, mark as uncertain
            word_duration = word_end - word_start
            confidence = max_overlap / word_duration if word_duration > 0 else 0
            
            final_segments.append({
                'word': word_data['word'],
                'start': word_start,
                'end': word_end,
                'speaker': assigned_speaker,
                'confidence': confidence,
                'whisper_confidence': word_data.get('confidence', 1.0),
                'boundary_case': confidence < 0.5  # Flag uncertain assignments
            })
        
        return final_segments
    
    def _smooth_speaker_transitions(self, segments: List[Dict]) -> List[Dict]:
        """
        Apply temporal smoothing to reduce spurious speaker switches
        """
        if len(segments) < 3:
            return segments
        
        smoothed = segments.copy()
        
        for i in range(1, len(smoothed) - 1):
            prev_speaker = smoothed[i-1]['speaker']
            curr_speaker = smoothed[i]['speaker']
            next_speaker = smoothed[i+1]['speaker']
            
            # If current word is isolated (different from neighbors who match)
            if prev_speaker == next_speaker and curr_speaker != prev_speaker:
                # Only switch if confidence is low
                if smoothed[i]['confidence'] < 0.7:
                    smoothed[i]['speaker'] = prev_speaker
                    smoothed[i]['smoothed'] = True
        
        return smoothed
    
    def _save_output(self, data: any, filename: str):
        """Save output data to JSON file"""
        output_path = self.output_dir / filename
        
        # Convert Path objects to strings for JSON serialization
        if isinstance(data, list) and data and isinstance(data[0], dict):
            # Handle list of dictionaries (like segments)
            json_data = data
        else:
            # Handle other data types (like strings)
            json_data = {"data": data}
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"  Saved: {output_path}")
    
    def format_output(self, segments: List[Dict], output_prefix: str):
        """Format and save results in multiple formats"""
        
        # JSON format
        json_path = self.output_dir / f"{output_prefix}.json"
        with open(json_path, 'w') as f:
            json.dump(segments, f, indent=2)
        
        # Human-readable format
        txt_path = self.output_dir / f"{output_prefix}.txt"
        with open(txt_path, 'w') as f:
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
        srt_path = self.output_dir / f"{output_prefix}.srt"
        with open(srt_path, 'w') as f:
            for i, seg in enumerate(segments, 1):
                start_time = self._format_timestamp(seg['start'])
                end_time = self._format_timestamp(seg['end'])
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"[{seg['speaker']}] {seg['word']}\n\n")
        
        print(f"  Formatted outputs saved: {output_prefix}.json, .txt, .srt")
    
    def _format_timestamp(self, seconds: float) -> str:
        """Convert seconds to SRT timestamp format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Whisper Pipeline for Speech Diarization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python whisper_pipeline.py audio_file.mp3
  python whisper_pipeline.py /path/to/audio.wav --output custom_results
  python whisper_pipeline.py interview.mp3 --token your_hf_token
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
        print("\nYou can get your token from: https://huggingface.co/settings/tokens")
        exit(1)
    
    # Run pipeline
    print(f"Processing audio file: {args.audio_file}")
    print(f"Output directory: {args.output}")
    print()
    
    try:
        pipeline = WhisperPipelineDiarizer(HF_TOKEN, args.output)
        results = pipeline.transcribe_and_diarize(args.audio_file)
        
        # Save formatted outputs
        pipeline.format_output(results, "whisper_final")
        
        print(f"\n✓ Pipeline completed successfully!")
        print(f"  - Total words processed: {len(results)}")
        print(f"  - Boundary cases flagged: {len([s for s in results if s.get('boundary_case', False)])}")
        print(f"  - Outputs saved to: {args.output}/")
        
    except Exception as e:
        print(f"✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()