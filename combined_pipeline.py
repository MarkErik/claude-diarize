"""
Combined Granite-Whisper Pipeline for Maximum Accuracy Diarization

Hybrid approach leveraging:
- Granite Speech 3.3-8B for superior transcription accuracy
- Whisper large-v3 for precise word-level timing
- Pyannote community-1 for speaker diarization
- Enhanced merging with confidence weighting and context awareness

This pipeline combines the best of both models:
- Granite provides accurate transcription
- Whisper provides precise word timing
- Pyannote provides speaker attribution
"""

import torch
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import os
import gc
from dotenv import load_dotenv
import re
from difflib import SequenceMatcher

# Load environment variables from .env file
load_dotenv()


class CombinedGraniteWhisperDiarizer:
    """
    Hybrid pipeline combining Granite transcription with Whisper timing
    for maximum accuracy diarization.
    
    Features:
    - Granite Speech 3.3-8B for superior transcription
    - Whisper large-v3 for precise word-level timing
    - Pyannote.audio community-1 for speaker diarization
    - Enhanced word-to-timing mapping with fuzzy matching
    - Multi-strategy speaker assignment with confidence weighting
    - Context-aware smoothing for speaker transitions
    """
    
    def __init__(
        self,
        hf_token: str,
        output_dir: str = "outputs",
        # Granite parameters
        granite_max_new_tokens: int = 200,
        granite_num_beams: int = 4,
        granite_temperature: float = 1.0,
        # Whisper parameters (for timing only)
        whisper_beam_size: int = 6,
        whisper_vad_min_silence_ms: int = 400,
        # Diarization parameters
        num_speakers: int = 2,
        # Merging parameters
        min_confidence_threshold: float = 0.5,
        context_window_size: int = 3,
    ):
        self.hf_token = hf_token
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Store parameters
        self.granite_max_new_tokens = granite_max_new_tokens
        self.granite_num_beams = granite_num_beams
        self.granite_temperature = granite_temperature
        self.whisper_beam_size = whisper_beam_size
        self.whisper_vad_min_silence_ms = whisper_vad_min_silence_ms
        self.num_speakers = num_speakers
        self.min_confidence_threshold = min_confidence_threshold
        self.context_window_size = context_window_size
        
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
        
        # Initialize models
        self.setup_models()
        
        # Context tracking for smoothing
        self._previous_speaker = None
        self._speaker_history = []
    
    def setup_models(self):
        """Initialize all pipeline components"""
        print("Setting up hybrid Granite-Whisper pipeline...")
        
        # 1. Granite Speech for transcription
        print("Loading Granite Speech 3.3-8B...")
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        
        model_id = "ibm-granite/granite-speech-3.3-8b"
        self.granite_processor = AutoProcessor.from_pretrained(model_id)
        
        try:
            self.granite_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                dtype=torch.float16 if self.device == "mps" else torch.float32
            ).to(self.device)
            
            if self.device == "mps":
                self.granite_model.gradient_checkpointing_enable()
                
        except Exception as e:
            print(f"Failed to load Granite model on {self.device}: {e}")
            print("Falling back to CPU...")
            self.device = "cpu"
            self.granite_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id
            ).to(self.device)
        
        # 2. Whisper for timing (transcription only, not used for text)
        print("Loading Whisper large-v3 for timing...")
        from faster_whisper import WhisperModel
        
        if self.device == "cuda":
            device_str = "cuda"
            compute_type = "float16"
        else:
            device_str = "cpu"
            compute_type = "float32"

        self.whisper_model = WhisperModel(
            "large-v3",
            device=device_str,
            compute_type=compute_type,
        )
        
        # 3. Pyannote for diarization
        print("Loading pyannote community-1 diarization pipeline...")
        from pyannote.audio import Pipeline
        
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=self.hf_token
        )
        
        # Move to appropriate device
        if torch.backends.mps.is_available():
            try:
                self.diarization_pipeline.to(torch.device("mps"))
            except:
                print("Note: pyannote may not fully support MPS, using CPU for diarization")
        elif torch.cuda.is_available():
            self.diarization_pipeline.to(torch.device("cuda"))
        
        print("✓ All models loaded successfully")
    
    def transcribe_and_diarize(self, audio_path: str) -> List[Dict]:
        """
        Complete hybrid pipeline:
        1. Transcribe with Granite for superior accuracy
        2. Get precise timing from Whisper
        3. Generate speaker segments with pyannote
        4. Enhanced merging with confidence weighting
        
        Returns:
            List of dicts with: {word, start, end, speaker, confidence, timing_confidence, match_confidence}
        """
        
        # Step 0: Preprocess audio for consistency
        print("Step 0: Preprocessing audio...")
        preprocessed_audio_path = self._preprocess_audio_for_diarization(audio_path)
        
        # Step 1: Get superior transcription from Granite
        print("Step 1: Transcribing with Granite Speech...")
        granite_transcript = self._transcribe_with_granite(preprocessed_audio_path)
        print(f"Granite transcript: {granite_transcript[:200]}...")
        
        # Save raw Granite transcription
        self._save_output(granite_transcript, "combined_granite_transcription.json")
        
        # Step 2: Get precise timing from Whisper
        print("Step 2: Getting word timing from Whisper...")
        words_with_timing = self._get_word_timing_from_whisper(
            preprocessed_audio_path, granite_transcript
        )
        
        # Save timing results
        self._save_output(words_with_timing, "combined_whisper_timing.json")
        
        # Step 3: Get speaker diarization
        print("Step 3: Running speaker diarization...")
        speaker_segments = self._get_speaker_segments(preprocessed_audio_path)
        
        # Save speaker segments
        self._save_output(speaker_segments, "combined_speaker_segments.json")
        
        # Step 4: Enhanced merging with confidence weighting
        print("Step 4: Enhanced merging with confidence weighting...")
        final_segments = self._merge_with_enhanced_timing(
            words_with_timing, speaker_segments
        )
        
        # Save final results
        self._save_output(final_segments, "combined_final_results.json")
        
        print(f"✓ Pipeline completed successfully!")
        print(f"  - Total words processed: {len(final_segments)}")
        print(f"  - High confidence words (>0.8): {len([s for s in final_segments if s['confidence'] > 0.8])}")
        print(f"  - Boundary cases flagged: {len([s for s in final_segments if s.get('boundary_case', False)])}")
        
        return final_segments
    
    def _transcribe_with_granite(self, audio_path: str) -> str:
        """Transcribe audio using Granite Speech for superior accuracy"""
        import torchaudio
        
        # Load and preprocess audio
        wav, sr = torchaudio.load(audio_path, normalize=True)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != 16000:
            import torchaudio.transforms as T
            resampler = T.Resample(sr, 16000)
            wav = resampler(wav)
        
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
        )
        
        # Ensure all inputs are on the correct device
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}
        
        with torch.no_grad():
            try:
                model_outputs = self.granite_model.generate(
                    **model_inputs,
                    max_new_tokens=self.granite_max_new_tokens,
                    num_beams=self.granite_num_beams,
                    do_sample=False,
                    min_length=1,
                    top_p=1.0,
                    repetition_penalty=3.0,
                    length_penalty=1.0,
                    temperature=self.granite_temperature,
                    bos_token_id=self.granite_processor.tokenizer.bos_token_id,
                    eos_token_id=self.granite_processor.tokenizer.eos_token_id,
                    pad_token_id=self.granite_processor.tokenizer.pad_token_id,
                )
            except RuntimeError as e:
                if "MPS" in str(e) or "metal" in str(e).lower():
                    print(f"MPS error encountered: {e}")
                    print("Falling back to CPU for transcription...")
                    self.granite_model.to("cpu")
                    model_inputs = {k: v.to("cpu") for k, v in model_inputs.items()}
                    
                    model_outputs = self.granite_model.generate(
                        **model_inputs,
                        max_new_tokens=self.granite_max_new_tokens,
                        num_beams=self.granite_num_beams,
                        do_sample=False,
                        min_length=1,
                        top_p=1.0,
                        repetition_penalty=3.0,
                        length_penalty=1.0,
                        temperature=self.granite_temperature,
                        bos_token_id=self.granite_processor.tokenizer.bos_token_id,
                        eos_token_id=self.granite_processor.tokenizer.eos_token_id,
                        pad_token_id=self.granite_processor.tokenizer.pad_token_id,
                    )
                    
                    self.granite_model.to(self.device)
                else:
                    raise e
        
        # Decode transcription
        num_input_tokens = model_inputs["input_ids"].shape[-1]
        new_tokens = model_outputs[0, num_input_tokens:].unsqueeze(0)
        transcript_text = self.granite_processor.tokenizer.batch_decode(
            new_tokens, 
            add_special_tokens=False, 
            skip_special_tokens=True
        )[0]
        
        return transcript_text.strip()
    
    def _get_word_timing_from_whisper(self, audio_path: str, granite_transcript: str) -> List[Dict]:
        """
        Use Whisper for precise word timing while using Granite's transcript
        """
        # Get timing from Whisper
        segments, info = self.whisper_model.transcribe(
            audio_path,
            word_timestamps=True,
            language="en",
            beam_size=self.whisper_beam_size,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=self.whisper_vad_min_silence_ms),
        )
        
        # Extract words with timing from Whisper
        whisper_words = []
        for segment in segments:
            if hasattr(segment, 'words'):
                for word in segment.words:
                    whisper_words.append({
                        'word': word.word.strip().lower(),
                        'start': word.start,
                        'end': word.end,
                        'confidence': getattr(word, 'probability', 1.0)
                    })
        
        # Create word list from Granite's superior transcript
        granite_words = self._normalize_text(granite_transcript).split()
        
        # Map Whisper timing to Granite words
        return self._map_timing_to_words(granite_words, whisper_words)
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for better matching"""
        # Remove extra whitespace, normalize punctuation
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\!\?\-]', '', text)
        return text.strip()
    
    def _word_similarity(self, word1: str, word2: str) -> float:
        """Calculate similarity between two words"""
        # Exact match
        if word1 == word2:
            return 1.0
        
        # Fuzzy similarity
        fuzzy_score = SequenceMatcher(None, word1, word2).ratio()
        
        # Partial match (word1 contains word2 or vice versa)
        if word1 in word2 or word2 in word1:
            partial_score = 0.8
        else:
            partial_score = 0.0
        
        return max(fuzzy_score, partial_score)
    
    def _map_timing_to_words(self, granite_words: List[str], whisper_words: List[Dict]) -> List[Dict]:
        """
        Intelligently map Whisper timing to Granite's superior transcript
        """
        result = []
        whisper_idx = 0
        
        for granite_word in granite_words:
            # Find best matching word in Whisper's timing
            best_match = None
            best_score = 0
            
            # Search within a reasonable window
            search_window = min(5, len(whisper_words) - whisper_idx)
            
            for i in range(search_window):
                if whisper_idx + i >= len(whisper_words):
                    break
                    
                whisper_word = whisper_words[whisper_idx + i]
                similarity = self._word_similarity(granite_word.lower(), whisper_word['word'].lower())
                
                if similarity > best_score:
                    best_match = whisper_word
                    best_score = similarity
                
                # If we find a good match, use it
                if similarity > 0.8:
                    break
            
            if best_match and best_score > 0.3:  # Minimum threshold
                result.append({
                    'word': granite_word,
                    'start': best_match['start'],
                    'end': best_match['end'],
                    'confidence': best_match['confidence'] * best_score,
                    'timing_confidence': best_match['confidence'],
                    'match_confidence': best_score,
                    'source_word': best_match['word']
                })
                
                # Skip ahead in Whisper words
                while whisper_idx < len(whisper_words) and best_match['word'] != whisper_words[whisper_idx]['word']:
                    whisper_idx += 1
                if whisper_idx < len(whisper_words):
                    whisper_idx += 1
            else:
                # Fallback: will be calculated later
                result.append({
                    'word': granite_word,
                    'start': 0.0,
                    'end': 0.0,
                    'confidence': 0.1,  # Low confidence for fallback
                    'timing_confidence': 0.0,
                    'match_confidence': 0.0,
                    'source_word': None
                })
        
        # Calculate fallback timing for unmatched words
        return self._calculate_fallback_timing(result)
    
    def _calculate_fallback_timing(self, words: List[Dict]) -> List[Dict]:
        """
        Calculate timing for words that couldn't be matched
        """
        if not words:
            return words
        
        # Get total duration from first and last matched words
        matched_words = [w for w in words if w['timing_confidence'] > 0]
        
        if len(matched_words) < 2:
            # Simple uniform distribution
            duration = 60.0  # Default assumption
            for i, word in enumerate(words):
                word['start'] = (i / len(words)) * duration
                word['end'] = ((i + 1) / len(words)) * duration
        else:
            # Use matched words as anchors
            total_duration = matched_words[-1]['end'] - matched_words[0]['start']
            
            current_time = matched_words[0]['start']
            word_idx = 0
            
            for i, word in enumerate(words):
                if word['timing_confidence'] > 0:
                    # This is a matched word, use its timing
                    current_time = word['start']
                else:
                    # This is an unmatched word, distribute evenly
                    if i < len(words) - 1 and words[i + 1]['timing_confidence'] > 0:
                        # Next word is matched, interpolate
                        next_time = words[i + 1]['start']
                        duration = (next_time - current_time) / 2
                        word['start'] = current_time + duration / 2
                        word['end'] = current_time + duration * 1.5
                        current_time = word['end']
                    else:
                        # Use average word duration
                        avg_duration = total_duration / len(words)
                        word['start'] = current_time
                        word['end'] = current_time + avg_duration
                        current_time = word['end']
        
        return words
    
    def _get_speaker_segments(self, audio_path: str) -> List[Dict]:
        """Get speaker segments from pyannote"""
        diarization = self.diarization_pipeline(
            audio_path,
            num_speakers=self.num_speakers
        )
        
        # Handle different pyannote versions
        if hasattr(diarization, 'speaker_diarization'):
            annotation = diarization.speaker_diarization
        else:
            annotation = diarization
        
        speaker_segments = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            speaker_segments.append({
                'start': turn.start,
                'end': turn.end,
                'speaker': speaker
            })
        
        return speaker_segments
    
    def _merge_with_enhanced_timing(self, words: List[Dict], speaker_segments: List[Dict]) -> List[Dict]:
        """
        Enhanced merging with confidence weighting and context awareness
        """
        final_segments = []
        
        # Reset context tracking
        self._speaker_history = []
        
        for word_data in words:
            word_start = word_data['start']
            word_end = word_data['end']
            word_mid = (word_start + word_end) / 2
            
            # Enhanced speaker assignment with multiple strategies
            assigned_speaker, confidence = self._assign_speaker_with_confidence(
                word_start, word_end, word_mid, speaker_segments
            )
            
            # Enhanced confidence calculation
            timing_confidence = word_data.get('timing_confidence', 1.0)
            match_confidence = word_data.get('match_confidence', 1.0)
            
            # Weighted combination of confidence sources
            timing_weight = 0.4
            match_weight = 0.3
            overlap_weight = 0.3
            
            overall_confidence = (
                timing_confidence * timing_weight +
                match_confidence * match_weight +
                confidence * overlap_weight
            )
            
            # Update context tracking
            self._update_speaker_context(assigned_speaker)
            
            final_segments.append({
                'word': word_data['word'],
                'start': word_start,
                'end': word_end,
                'speaker': assigned_speaker,
                'confidence': overall_confidence,
                'timing_confidence': timing_confidence,
                'match_confidence': match_confidence,
                'overlap_confidence': confidence,
                'boundary_case': confidence < self.min_confidence_threshold,
                'source_word': word_data.get('source_word'),
                'context_smoothed': len(self._speaker_history) > 1 and self._speaker_history[-2] == assigned_speaker
            })
        
        return final_segments
    
    def _assign_speaker_with_confidence(self, word_start: float, word_end: float, 
                                      word_mid: float, speaker_segments: List[Dict]) -> Tuple[str, float]:
        """
        Advanced speaker assignment with multiple confidence strategies
        """
        # Strategy 1: Maximum overlap
        max_overlap = 0
        best_speaker = None
        overlaps = {}
        
        for spk_seg in speaker_segments:
            overlap_start = max(word_start, spk_seg['start'])
            overlap_end = min(word_end, spk_seg['end'])
            overlap_duration = max(0, overlap_end - overlap_start)
            
            overlaps[spk_seg['speaker']] = overlap_duration
            
            if overlap_duration > max_overlap:
                max_overlap = overlap_duration
                best_speaker = spk_seg['speaker']
        
        # Calculate confidence based on overlap ratio
        word_duration = word_end - word_start
        overlap_confidence = max_overlap / word_duration if word_duration > 0 else 0
        
        # Strategy 2: Midpoint fallback for ambiguous cases
        if len(overlaps) > 1 and max_overlap - min(overlaps.values()) < 0.1 * word_duration:
            for spk_seg in speaker_segments:
                if spk_seg['start'] <= word_mid <= spk_seg['end']:
                    best_speaker = spk_seg['speaker']
                    break
        
        # Strategy 3: Context-aware smoothing
        context_confidence = self._calculate_context_confidence(best_speaker)
        
        # Combine confidence scores
        if context_confidence > 0:
            overall_confidence = (overlap_confidence + context_confidence) / 2
        else:
            overall_confidence = overlap_confidence
        
        return best_speaker, overall_confidence
    
    def _calculate_context_confidence(self, speaker: str) -> float:
        """Calculate confidence based on speaker context"""
        if not self._speaker_history:
            return 0.0
        
        # Check if speaker is consistent with recent history
        recent_speakers = self._speaker_history[-self.context_window_size:]
        same_speaker_count = sum(1 for spk in recent_speakers if spk == speaker)
        
        # Higher confidence if speaker is consistent
        consistency_ratio = same_speaker_count / len(recent_speakers)
        return consistency_ratio * 0.8  # Max 0.8 confidence from context
    
    def _update_speaker_context(self, speaker: str):
        """Update speaker history for context awareness"""
        self._speaker_history.append(speaker)
        
        # Keep only recent history
        if len(self._speaker_history) > self.context_window_size * 2:
            self._speaker_history = self._speaker_history[-self.context_window_size * 2:]
    
    def _preprocess_audio_for_diarization(self, audio_path: str) -> str:
        """Preprocess audio file for consistency across pipeline steps"""
        try:
            import librosa
            import soundfile as sf
            
            # Load audio and resample to 16kHz
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            # Create preprocessed output path
            preprocessed_path = self.output_dir / "combined-results" / "preprocessed_audio.wav"
            preprocessed_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save with consistent format
            sf.write(str(preprocessed_path), audio, 16000, subtype='PCM_16')
            
            print(f"  Preprocessed audio saved: {preprocessed_path}")
            return str(preprocessed_path)
            
        except Exception as e:
            print(f"Warning: Audio preprocessing failed: {e}")
            print("  Falling back to original audio file")
            return audio_path
    
    def _save_output(self, data: any, filename: str):
        """Save output data to JSON file"""
        output_path = self.output_dir / filename
        
        # Convert Path objects to strings for JSON serialization
        if isinstance(data, list) and data and isinstance(data[0], dict):
            json_data = data
        else:
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
        description="Combined Granite-Whisper Diarization Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python combined_pipeline.py audio_file.mp3
  python combined_pipeline.py /path/to/audio.wav --output custom_results
  python combined_pipeline.py interview.mp3 --token your_hf_token
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
    
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=2,
        help="Number of speakers to detect (default: 2)"
    )
    
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence threshold for boundary cases (default: 0.5)"
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
        pipeline = CombinedGraniteWhisperDiarizer(
            HF_TOKEN, 
            args.output,
            num_speakers=args.num_speakers,
            min_confidence_threshold=args.min_confidence
        )
        results = pipeline.transcribe_and_diarize(args.audio_file)
        
        # Save formatted outputs
        pipeline.format_output(results, "combined_final")
        
        print(f"\n✓ Combined pipeline completed successfully!")
        print(f"  - Total words processed: {len(results)}")
        print(f"  - High confidence words (>0.8): {len([s for s in results if s['confidence'] > 0.8])}")
        print(f"  - Boundary cases flagged: {len([s for s in results if s.get('boundary_case', False)])}")
        print(f"  - Context smoothed words: {len([s for s in results if s.get('context_smoothed', False)])}")
        print(f"  - Outputs saved to: {args.output}/")
        
    except Exception as e:
        print(f"✗ Combined pipeline failed: {e}")
        import traceback
        traceback.print_exc()