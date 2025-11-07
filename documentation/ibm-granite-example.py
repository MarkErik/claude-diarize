import torch
import torchaudio
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from huggingface_hub import hf_hub_download

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "ibm-granite/granite-speech-3.3-8b"
speech_granite_processor = AutoProcessor.from_pretrained(
    model_name)
tokenizer = speech_granite_processor.tokenizer
speech_granite = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_name).to(device)

# prepare speech and text prompt, using the appropriate prompt template

audio_path = hf_hub_download(repo_id=model_name, filename='10226_10111_000000.wav')
wav, sr = torchaudio.load(audio_path, normalize=True)
assert wav.shape[0] == 1 and sr == 16000 # mono, 16khz

# create text prompt
chat = [
    {
        "role": "system",
        "content": "Knowledge Cutoff Date: April 2024.\nToday's Date: April 9, 2024.\nYou are Granite, developed by IBM. You are a helpful AI assistant",
    },
    {
        "role": "user",
        "content": "<|audio|>can you transcribe the speech into a written format?",
    }
]

text = tokenizer.apply_chat_template(
    chat, tokenize=False, add_generation_prompt=True
)

# compute audio embeddings
model_inputs = speech_granite_processor(
    text,
    wav,
    device=device, # Computation device; returned tensors are put on CPU
    return_tensors="pt",
).to(device)

model_outputs = speech_granite.generate(
    **model_inputs,
    max_new_tokens=200,
    num_beams=4,
    do_sample=False,
    min_length=1,
    top_p=1.0,
    repetition_penalty=3.0,
    length_penalty=1.0,
    temperature=1.0,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)

# Transformers includes the input IDs in the response.
num_input_tokens = model_inputs["input_ids"].shape[-1]
new_tokens = torch.unsqueeze(model_outputs[0, num_input_tokens:], dim=0)

output_text = tokenizer.batch_decode(
    new_tokens, add_special_tokens=False, skip_special_tokens=True
)
print(f"STT output = {output_text[0].upper()}")
