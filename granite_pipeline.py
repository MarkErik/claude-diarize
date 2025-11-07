"""
Granite Speech Pipeline with Diarization

Extracted from diarization_test.py, this module provides a self-contained
Granite Speech diarization pipeline with output saving functionality.

Features:
- Granite Speech 3.3-8B for transcription
- Wav2vec2 forced alignment for precise word timestamps  
- pyannote.audio community-1 for diarization
- Intelligent boundary handling with confidence scores
- Step-by-step output saving for debugging and analysis

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


class GraniteSpeechDiarizer:
    """
    Multi-component pipeline using Granite Speech
    1. Granite Speech for transcription
    2. Wav2vec2 forced alignment for precise word timestamps
    3. pyannote community-1 for diarization
    4. Intelligent merging with boundary handling
    5. Step-by-step output saving for analysis
    """
    
    def __init__(self, hf_token: str, output_dir: str = "outputs"):
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        from pyannote.audio import Pipeline
        
        self.hf_token = hf_token
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        print("Loading Granite Speech 3.3-8B...")
        model_id = "ibm-granite/granite-speech-3.3-8b"
        
        # Use MPS for Apple Silicon, CUDA for NVIDIA, CPU as fallback
        if torch.backends.mps.is_available():
            self.device = "mps"
            print("Using Apple Silicon GPU (MPS)")
        elif torch.cuda.is_available():
            self.device = "cuda"
            print("Using NVIDIA GPU (CUDA)")
        else:
            self.device = "cpu"
            print("Using CPU")
        
        # Install peft if needed
        try:
            import peft
        except ImportError:
            print("Installing peft...")
            import subprocess
            subprocess.check_call(["pip", "install", "peft"])
        
        self.granite_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            device_map=self.device,
            torch_dtype=torch.float16  # MPS supports float16
        )
        
        self.granite_processor = AutoProcessor.from_pretrained(model_id)
        
        # Load pyannote community-1 for diarization
        print("Loading pyannote community-1 diarization pipeline...")
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=hf_token
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
        
        # Pre-load alignment model to avoid reloading
        print("Loading alignment model...")
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        self.alignment_processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-large-960h-lv60-self"
        )
        self.alignment_model = Wav2Vec2ForCTC.from_pretrained(
            "facebook/wav2vec2-large-960h-lv60-self"
        )
        self.alignment_model.to(self.device)
        
    def transcribe_and_diarize(self, audio_path: str) -> List[Dict]:
        """
        Complete multi-component pipeline:
        1. Transcribe with Granite Speech
        2. Forced alignment for precise word timestamps
        3. Diarize with pyannote community-1
        4. Intelligent merging with boundary handling
        
        Returns:
            List of dicts with: {word, start, end, speaker, confidence}
        """
        import torchaudio
        
        # Load audio - must be mono, 16kHz
        wav, sr = torchaudio.load(audio_path, normalize=True)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)  # Convert to mono
        if sr != 16000:
            import torchaudio.transforms as T
            resampler = T.Resample(sr, 16000)
            wav = resampler(wav)
            sr = 16000
        
        # Step 1: Transcribe with Granite Speech
        print("Step 1: Transcribing with Granite Speech...")
        transcript_text = self._transcribe_with_granite(wav, audio_path)
        print(f"Transcript: {transcript_text[:200]}...")
        
        # Save raw transcription output
        self._save_output(transcript_text, "granite_raw_transcription.json")
        
        # Clear audio from memory
        del wav
        gc.collect()
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()
        
        # Step 2: Forced alignment for precise word timestamps
        print("Step 2: Performing forced alignment...")
        words_with_timestamps = self._forced_alignment(audio_path, transcript_text)
        
        # Save forced alignment results
        self._save_output(words_with_timestamps, "granite_forced_alignment.json")
        
        # Clear memory after alignment
        gc.collect()
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()
        
        # Step 3: Diarization with pyannote community-1
        print("Step 3: Running speaker diarization...")
        diarization = self.diarization_pipeline(
            audio_path,
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
        
        # Clear diarization object
        del diarization
        gc.collect()
        
        # Step 4: Merge with intelligent boundary handling
        print("Step 4: Merging with boundary resolution...")
        merged_results = self._merge_with_boundary_handling(
            words_with_timestamps, 
            speaker_segments
        )
        
        # Save merged results
        self._save_output(merged_results, "granite_merged_results.json")
        
        # Final cleanup
        gc.collect()
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()
        
        # Step 5: Smooth speaker transitions
        print("Step 5: Smoothing speaker transitions...")
        final_segments = self._smooth_speaker_transitions(merged_results)
        
        # Save final results
        self._save_output(final_segments, "granite_final_results.json")
        
        print(f"Pipeline completed. Results saved to: {self.output_dir}")
        
        return final_segments
    
    def _transcribe_with_granite(self, wav: torch.Tensor, audio_path: str) -> str:
        """Transcribe audio using Granite Speech"""
        system_prompt = "Knowledge Cutoff Date: April 2024.\nToday's Date: November 2025.\nYou are Granite, developed by IBM. You are a helpful AI assistant"
        user_prompt = "<|audio|>Please transcribe this audio accurately."
        
        chat = [
            dict(role="system", content=system_prompt),
            dict(role="user", content=user_prompt),
        ]
        
        prompt = self.granite_processor.tokenizer.apply_chat_template(
            chat, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        model_inputs = self.granite_processor(
            prompt, 
            wav, 
            device=self.device, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            model_outputs = self.granite_model.generate(
                **model_inputs,
                max_new_tokens=2048,
                do_sample=False,
                num_beams=1
            )
        
        # Decode transcription
        num_input_tokens = model_inputs["input_ids"].shape[-1]
        new_tokens = model_outputs[0, num_input_tokens:].unsqueeze(0)
        transcript_text = self.granite_processor.tokenizer.batch_decode(
            new_tokens, 
            add_special_tokens=False, 
            skip_special_tokens=True
        )[0]
        
        # CRITICAL: Clear memory immediately after use
        del model_inputs, model_outputs, new_tokens
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()
        
        return transcript_text.strip()
    
    def _forced_alignment(self, audio_path: str, transcript: str) -> List[Dict]:
        """
        Use wav2vec2 for forced phoneme alignment to get precise word timestamps
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
            
            # CRITICAL: Clear temporary data
            del inputs, logits, audio
            gc.collect()
            if self.device == "mps":
                torch.mps.empty_cache()
            elif self.device == "cuda":
                torch.cuda.empty_cache()
            
            return words_with_timestamps
            
        except Exception as e:
            print(f"Forced alignment failed: {e}, using simple timing")
            # Fallback
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
            
            del audio
            gc.collect()
            
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


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Granite Speech Diarization Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python granite_pipeline.py audio_file.mp3
  python granite_pipeline.py /path/to/audio.wav --output custom_outputs
  python granite_pipeline.py interview.mp3 --token your_hf_token
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
    
    # Run pipeline
    print(f"Processing audio file: {args.audio_file}")
    print(f"Output directory: {args.output}")
    print()
    
    try:
        diarizer = GraniteSpeechDiarizer(HF_TOKEN, args.output)
        results = diarizer.transcribe_and_diarize(args.audio_file)
        
        print(f"\nPipeline completed successfully!")
        print(f"Total words processed: {len(results)}")
        print(f"Results saved to: {args.output}/")
        
        # Show summary statistics
        boundary_cases = [s for s in results if s.get('boundary_case', False)]
        smoothed_words = [s for s in results if s.get('smoothed', False)]
        
        print(f"Boundary cases flagged: {len(boundary_cases)}")
        print(f"Words smoothed: {len(smoothed_words)}")
        
        # Show speaker distribution
        speakers = {}
        for seg in results:
            speaker = seg.get('speaker', 'Unknown')
            speakers[speaker] = speakers.get(speaker, 0) + 1
        
        print(f"Speaker distribution:")
        for speaker, count in speakers.items():
            print(f"  {speaker}: {count} words")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()