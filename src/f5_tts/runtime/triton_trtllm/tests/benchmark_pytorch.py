#!/usr/bin/env python3
"""PyTorch-native benchmark for F5-TTS (GPU, torch>=2.8 for RTX 5080 sm_120).

Runs the same local samples as tests/benchmark_local.py but through the
original PyTorch DiT + Vocos path, for fair TRT-vs-PyTorch comparison.

Usage:
    python tests/benchmark_pytorch.py --model-path ckpts/F5TTS_v1_Base/model_1250000.safetensors \
        --vocab-file ckpts/F5TTS_v1_Base/vocab.txt
"""
import argparse
import json
import os
import sys
import time

import torch
import torchaudio

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{_HERE}/../../../../")  # F5-TTS/src (for f5_tts package)

from f5_tts.eval.utils_eval import padded_mel_batch
from f5_tts.infer.utils_infer import load_model
from f5_tts.model import DiT
from f5_tts.model.modules import get_vocos_mel_spectrogram
from f5_tts.model.utils import convert_char_to_pinyin, get_tokenizer, list_str_to_idx

TARGET_SR = 24000
TARGET_RMS = 0.1

SAMPLES = [
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
    parser.add_argument("--tllm-config", default="ckpts/F5TTS_v1_Base/trtllm_ckpt/config.json", type=str)
    parser.add_argument("--output-dir", default="./tests/benchmark_local", type=str)
    parser.add_argument("--steps", default=32, type=int, help="NFE steps")
    return parser.parse_args()


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(0)
    device = torch.device("cuda")

    vocab_char_map, vocab_size = get_tokenizer(args.vocab_file, "custom")

    with open(args.tllm_config) as f:
        cfg = json.load(f)["pretrained_config"]
    pt_model_config = dict(
        dim=cfg["hidden_size"],
        depth=cfg["num_hidden_layers"],
        heads=cfg["num_attention_heads"],
        ff_mult=cfg["ff_mult"],
        text_dim=cfg["text_dim"],
        text_mask_padding=cfg["text_mask_padding"],
        conv_layers=cfg["conv_layers"],
        pe_attn_head=cfg["pe_attn_head"],
    )
    model = load_model(DiT, pt_model_config, args.model_path).to(device)

    from vocos import Vocos
    vocos_dir = "ckpts/vocos"
    vocoder = Vocos.from_hparams(f"{vocos_dir}/config.yaml")
    state_dict = torch.load(f"{vocos_dir}/pytorch_model.bin", map_location="cpu", weights_only=True)
    vocoder.load_state_dict(state_dict)
    vocoder = vocoder.eval().to(device)

    rows = []
    for audio_path, ref_text, target_text in SAMPLES:
        wav, sr = torchaudio.load(audio_path)
        rms = torch.sqrt(torch.mean(torch.square(wav)))
        if rms < TARGET_RMS:
            wav = wav * TARGET_RMS / rms
        if sr != TARGET_SR:
            resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
            wav = resampler(wav)
        ref_mel = get_vocos_mel_spectrogram(wav.to(device)).squeeze(0)  # [100, T] on GPU
        ref_mel_len = ref_mel.shape[-1]
        est_len = int(ref_mel_len * (1 + len(target_text.encode("utf-8")) / max(len(ref_text.encode("utf-8")), 1)))

        pinyin = convert_char_to_pinyin([ref_text + target_text], polyphone=True)
        text_idx = list_str_to_idx(pinyin, vocab_char_map).to(device)

        mel_batch = padded_mel_batch([ref_mel.cpu()]).to(device)  # [1, T, 100]
        t0 = time.time()
        with torch.inference_mode():
            generated, _ = model.sample(
                cond=mel_batch,
                text=text_idx,
                duration=torch.tensor([est_len], device=device),
                lens=torch.tensor([ref_mel_len], device=device),
                steps=args.steps,
                cfg_strength=2.0,
                sway_sampling_coef=-1,
            )
        gen = generated[0, ref_mel_len:est_len, :].unsqueeze(0).permute(0, 2, 1).to(torch.float32)
        wave = vocoder.decode(gen)
        elapsed = time.time() - t0
        dur = wave.shape[-1] / TARGET_SR
        rows.append((audio_path, ref_mel_len, est_len, elapsed, dur))
        print(f"[{audio_path.split('/')[-1]}] ref_mel={ref_mel_len} est={est_len} | "
              f"latency={elapsed*1000:.0f} ms | audio={dur:.2f}s | RTF={elapsed/dur:.4f}")

    total_time = sum(r[3] for r in rows)
    total_dur = sum(r[4] for r in rows)
    rtf = total_time / total_dur
    print(f"\n=== RESULT: {len(rows)} samples | total latency {total_time:.2f}s | "
          f"audio {total_dur:.2f}s | RTF={rtf:.4f} ===")

    with open(os.path.join(args.output_dir, "rtf_pytorch.txt"), "w") as f:
        f.write(f"RTF: {rtf:.4f}\n")
        f.write(f"total_duration: {total_dur:.3f} seconds\n")
        f.write(f"total decoding time: {total_time:.3f} seconds\n")
        f.write(f"samples: {len(rows)}\n")
        for r in rows:
            f.write(f"  {r[0]} ref_mel={r[1]} est={r[2]} latency={r[3]*1000:.0f}ms RTF={r[3]/r[4]:.4f}\n")


if __name__ == "__main__":
    main()
