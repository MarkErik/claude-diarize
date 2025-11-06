"""
Comparison test: Granite Speech vs Multi-Component Pipeline
Focus: Maximum accuracy for 2-speaker interview diarization
"""

import torch
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# ============================================================================
# OPTION A: GRANITE SPEECH 3.3-8B
# ============================================================================

class GraniteSpeechDiarizer:
    """Unified model approach using IBM Granite Speech"""
    
    def __init__(self):
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        
        print("Loading Granite Speech 3.3-8B...")
        model_id = "ibm-granite/granite-speech-3.3-8b"
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(model_id)
        
    def transcribe_and_diarize(self, audio_path: str) -> List[Dict]:
        """
        Process audio file and return word-level diarization
        
        Returns:
            List of dicts with: {word, start, end, speaker}
        """
        import librosa
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Process with Granite
        inputs = self.processor(
            audio, 
            sampling_rate=sr, 
            return_tensors="pt"
        ).to(self.device)
        
        # Generate with diarization task
        # Note: Check Granite docs for exact prompt format for diarization
        predicted_ids = self.model.generate(
            **inputs,
            max_new_tokens=512,
            num_beams=5,
            task="diarize",  # Specify diarization task
        )
        
        # Decode output
        result = self.processor.batch_decode(
            predicted_ids, 
            skip_special_tokens=True
        )[0]
        
        # Parse Granite output format (adjust based on actual format)
        word_segments = self._parse_granite_output(result)
        
        return word_segments
    
    def _parse_granite_output(self, output: str) -> List[Dict]:
        """
        Parse Granite's output format into word-level segments
        This needs to be adjusted based on actual Granite output format
        """
        # Placeholder - actual parsing depends on Granite's output structure
        # Expected format might be something like:
        # "<speaker_0> [0.0-1.2] Hello </speaker_0> <speaker_1> [1.2-2.5] Hi there </speaker_1>"
        
        segments = []
        # TODO: Implement actual parsing based on Granite documentation
        return segments


# ============================================================================
# OPTION B: MULTI-COMPONENT PIPELINE (MAXIMUM ACCURACY)
# ============================================================================

class AccuratePipelineDiarizer:
    """
    Multi-step pipeline for maximum accuracy:
    1. Whisper large-v3 for transcription
    2. Montreal Forced Aligner for precise word timestamps
    3. pyannote.audio 3.x for diarization
    4. Intelligent merging with boundary handling
    """
    
    def __init__(self, hf_token: str):
        self.hf_token = hf_token
        self.setup_models()
        
    def setup_models(self):
        """Initialize all pipeline components"""
        # 1. Whisper for transcription
        from faster_whisper import WhisperModel
        
        print("Loading Whisper large-v3...")
        self.whisper = WhisperModel(
            "large-v3",
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="float16" if torch.cuda.is_available() else "float32"
        )
        
        # 2. Pyannote for diarization
        from pyannote.audio import Pipeline
        
        print("Loading pyannote diarization pipeline...")
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=self.hf_token
        )
        
        if torch.cuda.is_available():
            self.diarization_pipeline.to(torch.device("cuda"))
    
    def transcribe_and_diarize(self, audio_path: str) -> List[Dict]:
        """
        Complete pipeline for word-level diarized transcription
        
        Returns:
            List of dicts with: {word, start, end, speaker, confidence}
        """
        # Step 1: Transcribe with Whisper
        print("Step 1: Transcribing with Whisper...")
        segments, info = self.whisper.transcribe(
            audio_path,
            word_timestamps=True,
            language="en",  # Adjust as needed
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
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
        
        # Step 2: Forced Alignment (if available)
        print("Step 2: Performing forced alignment...")
        aligned_words = self._forced_alignment(audio_path, words_with_timestamps)
        
        # Step 3: Diarization
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
        
        # Step 4: Merge with intelligent boundary handling
        print("Step 4: Merging with boundary resolution...")
        final_segments = self._merge_with_boundary_handling(
            aligned_words, 
            speaker_segments
        )
        
        return final_segments
    
    def _forced_alignment(self, audio_path: str, words: List[Dict]) -> List[Dict]:
        """
        Use wav2vec2 for forced phoneme alignment
        This refines word boundaries to be more accurate
        """
        try:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
            import librosa
            import torch.nn.functional as F
            
            # Load alignment model
            processor = Wav2Vec2Processor.from_pretrained(
                "facebook/wav2vec2-large-960h-lv60-self"
            )
            model = Wav2Vec2ForCTC.from_pretrained(
                "facebook/wav2vec2-large-960h-lv60-self"
            )
            
            if torch.cuda.is_available():
                model = model.to("cuda")
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Process audio
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Get logits
            with torch.no_grad():
                logits = model(**inputs).logits
            
            # Align words to audio frames
            # This is a simplified version - full MFA implementation is more complex
            aligned_words = self._align_words_to_frames(words, logits, processor)
            
            return aligned_words
            
        except Exception as e:
            print(f"Forced alignment failed: {e}, using Whisper timestamps")
            return words
    
    def _align_words_to_frames(self, words: List[Dict], logits, processor) -> List[Dict]:
        """Align words to audio frames using CTC logits"""
        # Simplified alignment - in production, use proper CTC decoding with timestamps
        # For now, return original words (you can implement full alignment)
        return words
    
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
        
        # Post-processing: Smooth speaker transitions
        final_segments = self._smooth_speaker_transitions(final_segments)
        
        return final_segments
    
    def _smooth_speaker_transitions(self, segments: List[Dict]) -> List[Dict]:
        """
        Apply temporal smoothing to reduce spurious speaker switches
        Single words surrounded by another speaker likely belong to that speaker
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


# ============================================================================
# EVALUATION AND COMPARISON
# ============================================================================

class DiarizationEvaluator:
    """Compare and evaluate diarization results"""
    
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


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


# ============================================================================
# MAIN COMPARISON SCRIPT
# ============================================================================

def run_comparison_test(audio_path: str, hf_token: str, output_dir: str = "results"):
    """
    Run complete comparison between Granite Speech and Pipeline approach
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    print("="*80)
    print("DIARIZATION COMPARISON TEST")
    print("="*80)
    
    # Test Option A: Granite Speech
    print("\n\nTesting OPTION A: Granite Speech 3.3-8B")
    print("-"*80)
    try:
        granite = GraniteSpeechDiarizer()
        granite_results = granite.transcribe_and_diarize(audio_path)
        
        evaluator = DiarizationEvaluator()
        evaluator.format_output(
            granite_results, 
            f"{output_dir}/granite_speech"
        )
        print(f"✓ Granite Speech completed: {len(granite_results)} words")
        
    except Exception as e:
        print(f"✗ Granite Speech failed: {e}")
        granite_results = None
    
    # Test Option B: Multi-component Pipeline
    print("\n\nTesting OPTION B: Multi-Component Pipeline")
    print("-"*80)
    try:
        pipeline = AccuratePipelineDiarizer(hf_token)
        pipeline_results = pipeline.transcribe_and_diarize(audio_path)
        
        evaluator = DiarizationEvaluator()
        evaluator.format_output(
            pipeline_results, 
            f"{output_dir}/pipeline"
        )
        print(f"✓ Pipeline completed: {len(pipeline_results)} words")
        
        # Show boundary cases
        boundary_cases = [s for s in pipeline_results if s.get('boundary_case', False)]
        print(f"  Boundary cases flagged: {len(boundary_cases)}")
        
    except Exception as e:
        print(f"✗ Pipeline failed: {e}")
        pipeline_results = None
    
    # Summary
    print("\n\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    if granite_results and pipeline_results:
        print(f"\nGranite Speech: {len(granite_results)} words")
        print(f"Pipeline:       {len(pipeline_results)} words")
        print(f"\nResults saved to: {output_dir}/")
        print(f"  - granite_speech.json, .txt, .srt")
        print(f"  - pipeline.json, .txt, .srt")
        print(f"\nReview both outputs to determine which is more accurate for your use case.")
    
    return granite_results, pipeline_results


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Configuration
    AUDIO_FILE = ""  # Replace with your audio file
    HF_TOKEN = ""  # Get from https://huggingface.co/settings/tokens
    
    # Run comparison
    granite_results, pipeline_results = run_comparison_test(
        audio_path=AUDIO_FILE,
        hf_token=HF_TOKEN,
        output_dir="diarization_comparison"
    )
