#!/usr/bin/env python3
"""Local benchmark for F5-TTS TRT-LLM (no HF dataset needed).

Constructs test samples from local reference audios (repo examples) with
varying prompt lengths and target text lengths, then measures RTF for the
TRT-LLM backend.

Usage:
    python tests/benchmark_local.py --model-path ckpts/F5TTS_v1_Base/model_1250000.safetensors \
        --vocab-file ckpts/F5TTS_v1_Base/vocab.txt \
        --tllm-model-dir ckpts/F5TTS_v1_Base/trtllm_engine \
        --vocoder-trt-engine-path ckpts/vocos_vocoder.plan
"""
import argparse
import importlib
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
import torchaudio

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_HERE)                    # tests/ (for model_repo_f5_tts)
sys.path.append(os.path.dirname(_HERE))   # triton_trtllm/ (for benchmark.py)
sys.path.append(f"{_HERE}/../../../../")  # F5-TTS/src (for f5_tts package)

from f5_tts.eval.utils_eval import padded_mel_batch
from f5_tts.model.modules import get_vocos_mel_spectrogram
from f5_tts.model.utils import convert_char_to_pinyin, get_tokenizer, list_str_to_idx

F5TTS = importlib.import_module("model_repo_f5_tts.f5_tts.1.f5_tts_trtllm").F5TTS
from benchmark import VocosTensorRT  # noqa: E402

TARGET_SR = 24000
TARGET_RMS = 0.1

# (audio_path, ref_text, target_text)
SAMPLES = [
    # English: basic_ref_en.wav ~11s, various prompt cuts and target lengths
    ("src/f5_tts/infer/examples/basic/basic_ref_en.wav",
     "Some call me nature, others call me mother nature.",
     "I don't really care what you call me."),
    ("src/f5_tts/infer/examples/basic/basic_ref_en.wav",
     "Some call me nature, others call me mother nature.",
     "I've been a silent spectator, watching species evolve, empires rise and fall."),
    ("src/f5_tts/infer/examples/basic/basic_ref_en.wav",
     "Some call me nature, others call me mother nature.",
     "But always remember, I am mighty and enduring. Nature is not a place to visit, it is home."),
    ("src/f5_tts/infer/examples/basic/basic_ref_en.wav",
     "Some call me nature, others call me mother nature.",
     "The mountains are calling and I must go. Every sunrise brings new hope for a better day."),
    # Chinese: basic_ref_zh.wav, various prompt cuts and target lengths
    ("src/f5_tts/infer/examples/basic/basic_ref_zh.wav",
     "秋天到了，满山的红叶如火如霞，美不胜收。",
     "清晨的阳光洒在山谷里，薄雾慢慢散开。"),
    ("src/f5_tts/infer/examples/basic/basic_ref_zh.wav",
     "秋天到了，满山的红叶如火如霞，美不胜收。",
     "溪水潺潺流过石间，鸟儿在枝头欢快地歌唱，仿佛整个世界都在苏醒。"),
    ("src/f5_tts/infer/examples/basic/basic_ref_zh.wav",
     "秋天到了，满山的红叶如火如霞，美不胜收。",
     "远方的山峰在夕阳下镀上了一层金色，云朵像是被点燃的棉花，缓缓飘向天边。"),
    ("src/f5_tts/infer/examples/basic/basic_ref_zh.wav",
     "秋天到了，满山的红叶如火如霞，美不胜收。",
     "在这金秋时节，层林尽染，叠翠流金，大自然用最绚烂的色彩描绘出一幅动人的画卷。"),
]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=str)
    parser.add_argument("--vocab-file", required=True, type=str)
    parser.add_argument("--tllm-model-dir", required=True, type=str)
    parser.add_argument("--vocoder-trt-engine-path", default=None, type=str)
    parser.add_argument("--output-dir", default="./tests/benchmark_local", type=str)
    parser.add_argument("--max-prompt-sec", default=None, type=float, help="cap prompt length (s)")
    parser.add_argument("--batch-size", default=1, type=int)
    return parser.parse_args()


def prepare_sample(audio_path, ref_text, target_text, max_prompt_sec=None):
    wav, sr = torchaudio.load(audio_path)
    if max_prompt_sec is not None:
        wav = wav[:, : int(max_prompt_sec * sr)]
    rms = torch.sqrt(torch.mean(torch.square(wav)))
    if rms < TARGET_RMS:
        wav = wav * TARGET_RMS / rms
    if sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
        wav = resampler(wav)
    ref_mel = get_vocos_mel_spectrogram(wav).squeeze(0)  # [100, T]
    ref_mel_len = ref_mel.shape[-1]
    est_len = int(ref_mel_len * (1 + len(target_text.encode("utf-8")) / max(len(ref_text.encode("utf-8")), 1)))
    return ref_mel, ref_mel_len, est_len, ref_text + target_text


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(0)

    vocab_char_map, vocab_size = get_tokenizer(args.vocab_file, "custom")
    with open(os.path.join(args.tllm_model_dir, "config.json")) as f:
        tllm_model_config = json.load(f)

    model = F5TTS(
        tllm_model_config,
        debug_mode=False,
        tllm_model_dir=args.tllm_model_dir,
        model_path=args.model_path,
        vocab_size=vocab_size,
    )
    vocoder = VocosTensorRT(engine_path=args.vocoder_trt_engine_path)

    # warmup
    for _ in range(2):
        ref_mel = torch.randn(100, 300)
        text = torch.randint(1, 2000, (1, 30))
        model.sample(text, ref_mel.unsqueeze(0).transpose(1, 2), torch.tensor([100]), [300])

    rows = []
    for audio_path, ref_text, target_text in SAMPLES:
        ref_mel, ref_mel_len, est_len, full_text = prepare_sample(
            audio_path, ref_text, target_text, args.max_prompt_sec
        )
        pinyin = convert_char_to_pinyin([full_text], polyphone=True)
        text_idx = list_str_to_idx(pinyin, vocab_char_map)

        mel_batch = padded_mel_batch([ref_mel])  # [1, T, 100]
        cond = F.pad(mel_batch, (0, 0, 0, est_len - mel_batch.shape[1], 0, 0))

        t0 = time.time()
        denoised, _ = model.sample(text_idx, cond, torch.tensor([ref_mel_len]), [est_len])
        gen = denoised[0, ref_mel_len:est_len, :].unsqueeze(0)
        gen_mel = gen.permute(0, 2, 1).to(torch.float32).contiguous()  # CPU 连续化（GPU contiguous 需 kernel）
        wave = vocoder.decode(gen_mel.cuda()).cpu()
        elapsed = time.time() - t0
        dur = wave.shape[1] / TARGET_SR
        rows.append((audio_path, ref_mel_len, est_len, elapsed, dur))
        print(f"[{audio_path.split('/')[-1]}] ref_mel={ref_mel_len} est={est_len} | "
              f"latency={elapsed*1000:.0f} ms | audio={dur:.2f}s | RTF={elapsed/dur:.4f}")

    total_time = sum(r[3] for r in rows)
    total_dur = sum(r[4] for r in rows)
    rtf = total_time / total_dur
    print(f"\n=== RESULT: {len(rows)} samples | total latency {total_time:.2f}s | "
          f"audio {total_dur:.2f}s | RTF={rtf:.4f} ===")

    with open(os.path.join(args.output_dir, "rtf.txt"), "w") as f:
        f.write(f"RTF: {rtf:.4f}\n")
        f.write(f"total_duration: {total_dur:.3f} seconds\n")
        f.write(f"total decoding time: {total_time:.3f} seconds\n")
        f.write(f"samples: {len(rows)}\n")
        for r in rows:
            f.write(f"  {r[0]} ref_mel={r[1]} est={r[2]} latency={r[3]*1000:.0f}ms RTF={r[3]/r[4]:.4f}\n")


if __name__ == "__main__":
    main()
