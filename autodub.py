#!/usr/bin/env python3
"""Batch-dub English videos into Indonesian, fully local and open source.

Pipeline per video: ffmpeg extract audio -> faster-whisper (ASR + timestamps)
-> NLLB-200 (translation) -> Piper TTS (Indonesian voice, natural pace, no
speed-up) -> pad the video with freeze-frame pauses wherever a segment's
Indonesian speech runs longer than its original English slot -> reassemble
audio track on the new (padded) timeline -> mux back onto the video.

Usage:
    .venv/bin/python autodub.py --source ./videos_in --output ./videos_out
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import urllib.request
import wave
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)  # progress prints visible immediately even when redirected to a log file

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".ts"}

PIPER_VOICE_BASE_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
    "id/id_ID/news_tts/medium/"
)
PIPER_VOICE_NAME = "id_ID-news_tts-medium"

NLLB_MODEL_NAME = "facebook/nllb-200-distilled-600M"
NLLB_SRC_LANG = "eng_Latn"
NLLB_TGT_LANG = "ind_Latn"

SAMPLE_RATE = 22050  # Piper's id_ID-news_tts-medium output rate

FREEZE_MIN_GAP = 0.12  # ignore overruns shorter than this to avoid jittery micro-pauses


def tmp_for(path: Path) -> Path:
    """A same-directory scratch path for atomic writes: write here, then
    Path.replace() into the real name. A file only ever exists at the real
    name once it's fully written -- a killed/crashed write can never leave a
    truncated file behind that a resume would mistake for "already done"."""
    return path.with_name(path.stem + ".part" + path.suffix)


def ensure_piper_voice(models_dir: Path) -> tuple[Path, Path]:
    models_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = models_dir / f"{PIPER_VOICE_NAME}.onnx"
    config_path = models_dir / f"{PIPER_VOICE_NAME}.onnx.json"
    for path in (onnx_path, config_path):
        if not path.exists():
            url = PIPER_VOICE_BASE_URL + path.name
            print(f"[setup] downloading {path.name} ...")
            urllib.request.urlretrieve(url, path)
    return onnx_path, config_path


def extract_audio(video_path: Path, wav_path: Path) -> None:
    tmp_path = tmp_for(wav_path)
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-ac", "1", "-ar", "16000", "-vn", str(tmp_path),
        ],
        check=True, capture_output=True,
    )
    tmp_path.replace(wav_path)


def transcribe(wav_path: Path, whisper_model, ) -> list[dict]:
    segments_iter, _info = whisper_model.transcribe(
        str(wav_path), language="en", vad_filter=True,
    )
    segments = [
        {"start": seg.start, "end": seg.end, "text": seg.text.strip()}
        for seg in segments_iter if seg.text.strip()
    ]
    return segments


def translate_segments(segments: list[dict], tokenizer, model, on_batch=None) -> list[dict]:
    import torch

    tokenizer.src_lang = NLLB_SRC_LANG
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(NLLB_TGT_LANG)

    batch_size = 8
    for i in range(0, len(segments), batch_size):
        batch_segs = segments[i:i + batch_size]
        inputs = tokenizer([seg["text"] for seg in batch_segs], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            generated = model.generate(
                **inputs, forced_bos_token_id=forced_bos_token_id, max_new_tokens=256,
            )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for seg, translated in zip(batch_segs, decoded):
            seg["text_id"] = translated.strip()
        if on_batch:
            on_batch()
    return segments


def _piper_executable() -> str:
    # Resolve relative to the running interpreter so this works whether or
    # not the venv has been "activated" (PATH may not include .venv/bin).
    candidate = Path(sys.executable).parent / "piper"
    return str(candidate) if candidate.exists() else "piper"


def synthesize_segment(text: str, onnx_path: Path, config_path: Path, out_wav: Path) -> None:
    subprocess.run(
        [
            _piper_executable(),
            "--model", str(onnx_path),
            "--config", str(config_path),
            "--output_file", str(out_wav),
        ],
        input=text.encode("utf-8"),
        check=True, capture_output=True,
    )


def wav_duration_seconds(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def _atempo_factors(ratio: float) -> list[float]:
    # ffmpeg's atempo filter only accepts [0.5, 2.0] per instance; chain
    # multiple instances to reach ratios outside that band.
    factors = []
    remaining = ratio
    for _ in range(3):
        if 0.5 <= remaining <= 2.0:
            factors.append(remaining)
            remaining = 1.0
            break
        step = 2.0 if remaining > 1.0 else 0.5
        factors.append(step)
        remaining /= step
    if remaining != 1.0:
        factors.append(remaining)
    return factors


def fit_duration(in_wav: Path, target_seconds: float, out_wav: Path) -> float:
    """Time-fit synthesized speech to its target slot, returning the final duration.

    Speech is never sped up: rushing the audio to squeeze into a short slot is
    what made words unintelligible before. If the slot has spare time the
    audio is slowed down to fill it naturally; if the audio still needs more
    time than the slot has, it is left at its natural pace and the leftover
    is padded into the video afterwards as a freeze-frame pause.
    """
    actual = wav_duration_seconds(in_wav)
    tmp_path = tmp_for(out_wav)
    if actual <= 0 or target_seconds <= 0 or actual >= target_seconds:
        tmp_path.write_bytes(in_wav.read_bytes())
        tmp_path.replace(out_wav)
        return actual

    ratio = max(0.5, actual / target_seconds)
    filter_str = ",".join(f"atempo={f:.4f}" for f in _atempo_factors(ratio))
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(in_wav), "-filter:a", filter_str, str(tmp_path)],
        check=True, capture_output=True,
    )
    tmp_path.replace(out_wav)
    return wav_duration_seconds(out_wav)


def assemble_track(segments: list[dict], total_duration: float, temp_dir: Path) -> Path:
    import numpy as np
    from pydub import AudioSegment

    buffer = np.zeros(int(total_duration * SAMPLE_RATE) + SAMPLE_RATE, dtype=np.int32)
    for idx, seg in enumerate(segments):
        fitted_path = temp_dir / f"seg_{idx:04d}_fitted.wav"
        if not fitted_path.exists():
            continue
        audio = AudioSegment.from_wav(fitted_path)
        samples = np.array(audio.get_array_of_samples(), dtype=np.int32)
        if audio.channels > 1:
            samples = samples.reshape((-1, audio.channels)).mean(axis=1).astype(np.int32)

        start = int(seg["new_start"] * SAMPLE_RATE)
        end = start + len(samples)
        if end > len(buffer):
            buffer = np.concatenate([buffer, np.zeros(end - len(buffer), dtype=np.int32)])
        buffer[start:end] += samples

    clipped = np.clip(buffer, -32768, 32767).astype(np.int16)
    track = AudioSegment(clipped.tobytes(), frame_rate=SAMPLE_RATE, sample_width=2, channels=1)

    out_path = temp_dir / "dubbed_track.wav"
    tmp_path = tmp_for(out_path)
    track.export(tmp_path, format="wav")
    tmp_path.replace(out_path)
    return out_path


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def build_freeze_padded_video(video_path: Path, freeze_points: list[tuple[float, float]], out_path: Path) -> None:
    """Re-encode the video, holding the last frame at each (time, duration) point.

    Used to buy extra time for Indonesian speech that runs longer than its
    original English slot, instead of speeding the speech up.

    Implemented as a raw-frame relay (decode -> duplicate frames in Python ->
    encode), not an ffmpeg trim/tpad/concat filtergraph: that filtergraph fans
    every decoded frame out to every trim node, so its cost scales with
    frames * points -- with thousands of freeze points (routine here, since
    Indonesian speech often overruns its English slot) it effectively never
    finishes. Duplicating frames in a linear pass is O(frames + extra frames)
    no matter how many freeze points there are.
    """
    tmp_path = tmp_for(out_path)
    duration = probe_duration(video_path)
    points = sorted((t, d) for t, d in freeze_points if t < duration - 0.01)

    if not points:
        subprocess.run(["ffmpeg", "-y", "-i", str(video_path), "-c", "copy", str(tmp_path)], check=True, capture_output=True)
        tmp_path.replace(out_path)
        return

    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate",
            "-of", "csv=p=0", str(video_path),
        ],
        check=True, capture_output=True, text=True,
    )
    # Some sources are actually mpegts containers wearing a .mp4 extension;
    # ffprobe can then repeat the stream line once per PMT occurrence, so
    # only the first line is real.
    first_line = probe.stdout.strip().splitlines()[0]
    width_s, height_s, rate_s = first_line.split(",")
    width, height = int(width_s), int(height_s)
    num, den = rate_s.split("/")
    fps = float(num) / float(den)
    frame_size = width * height * 3 // 2  # yuv420p: full-res Y + quarter-res U and V

    decode = subprocess.Popen(
        ["ffmpeg", "-y", "-i", str(video_path), "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    encode = subprocess.Popen(
        [
            "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p",
            "-s", f"{width}x{height}", "-r", f"{fps:.6f}", "-i", "-",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18", str(tmp_path),
        ],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )

    point_idx = 0
    frame_idx = 0
    try:
        while True:
            frame = decode.stdout.read(frame_size)
            if len(frame) < frame_size:
                break
            encode.stdin.write(frame)
            t = frame_idx / fps
            while point_idx < len(points) and t >= points[point_idx][0]:
                for _ in range(round(points[point_idx][1] * fps)):
                    encode.stdin.write(frame)
                point_idx += 1
            frame_idx += 1
    finally:
        encode.stdin.close()
        decode.stdout.close()
    decode.wait()
    encode.wait()
    if decode.returncode != 0 or encode.returncode != 0:
        raise RuntimeError(f"freeze-pad pipe failed (decode rc={decode.returncode}, encode rc={encode.returncode})")

    tmp_path.replace(out_path)


def mux_video(video_path: Path, dubbed_audio: Path, out_path: Path) -> None:
    tmp_path = tmp_for(out_path)
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(video_path), "-i", str(dubbed_audio),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest", str(tmp_path),
        ],
        check=True, capture_output=True,
    )
    tmp_path.replace(out_path)


def load_state(state_path: Path) -> dict:
    if state_path.exists():
        return json.loads(state_path.read_text())
    return {"segments": None, "cumulative_shift": 0.0, "freeze_points": []}


def save_state(state_path: Path, state: dict) -> None:
    tmp_path = tmp_for(state_path)
    tmp_path.write_text(json.dumps(state))
    tmp_path.replace(state_path)


def process_video(video_path: Path, out_path: Path, models: dict, keep_temp: bool) -> None:
    # Persistent (non-tmpfs) work dir, deterministic per output file, so a
    # killed/crashed run can resume from its last completed step/segment
    # instead of redoing everything from scratch.
    work_dir = out_path.parent / ".autodub_work" / out_path.stem
    work_dir.mkdir(parents=True, exist_ok=True)
    state_path = work_dir / "state.json"
    state = load_state(state_path)

    wav_path = work_dir / "audio.wav"
    print(f"  [1/6] extracting audio ...")
    if wav_path.exists():
        print(f"        sudah ada, dilewati")
    else:
        extract_audio(video_path, wav_path)

    print(f"  [2/6] transcribing (English) ...")
    if state["segments"] is None:
        segments = transcribe(wav_path, models["whisper"])
        for seg in segments:
            seg["text_id"] = None
            seg["new_start"] = None
        state["segments"] = segments
        save_state(state_path, state)
    else:
        segments = state["segments"]
    print(f"        {len(segments)} segments detected")

    print(f"  [3/6] translating to Indonesian ...")
    pending = [seg for seg in segments if seg["text_id"] is None]
    if pending:
        translate_segments(
            pending, models["nllb_tokenizer"], models["nllb_model"],
            on_batch=lambda: save_state(state_path, state),
        )
    else:
        print(f"        sudah ada, dilewati")

    print(f"  [4/6] synthesizing Indonesian speech (natural pace) ...")
    cumulative_shift = state["cumulative_shift"]
    freeze_points = state["freeze_points"]
    done = sum(1 for seg in segments if seg["new_start"] is not None)
    if done:
        print(f"        melanjutkan dari segmen {done}/{len(segments)}")
    for idx, seg in enumerate(segments):
        raw_wav = work_dir / f"seg_{idx:04d}_raw.wav"
        fitted_wav = work_dir / f"seg_{idx:04d}_fitted.wav"
        if seg["new_start"] is not None and fitted_wav.exists():
            continue

        synthesize_segment(seg["text_id"], models["piper_onnx"], models["piper_config"], raw_wav)
        fitted_duration = fit_duration(raw_wav, seg["end"] - seg["start"], fitted_wav)

        seg["new_start"] = seg["start"] + cumulative_shift
        extra = fitted_duration - (seg["end"] - seg["start"])
        if extra > FREEZE_MIN_GAP:
            freeze_points.append([seg["end"], extra])
            cumulative_shift += extra

        state["cumulative_shift"] = cumulative_shift
        state["freeze_points"] = freeze_points
        save_state(state_path, state)

    total_duration = wav_duration_seconds(wav_path) + cumulative_shift
    print(f"  [5/6] assembling dubbed audio track ...")
    dubbed_audio = work_dir / "dubbed_track.wav"
    if dubbed_audio.exists():
        print(f"        sudah ada, dilewati")
    else:
        assemble_track(segments, total_duration, work_dir)

    print(f"  [6/6] muxing into final video ...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if freeze_points:
        print(f"        pausing video at {len(freeze_points)} spot(s) so speech has room to finish (+{cumulative_shift:.1f}s total)")
        padded_video = work_dir / "padded_video.mp4"
        if not padded_video.exists():
            build_freeze_padded_video(video_path, [tuple(p) for p in freeze_points], padded_video)
        mux_video(padded_video, dubbed_audio, out_path)
    else:
        mux_video(video_path, dubbed_audio, out_path)

    if not keep_temp:
        shutil.rmtree(work_dir, ignore_errors=True)
    else:
        print(f"  (temp files kept at {work_dir})")


def load_models(whisper_model_size: str, models_dir: Path) -> dict:
    print(f"[setup] loading faster-whisper ({whisper_model_size}, CPU, int8) ...")
    from faster_whisper import WhisperModel
    whisper_model = WhisperModel(whisper_model_size, device="cpu", compute_type="int8")

    print(f"[setup] loading translation model ({NLLB_MODEL_NAME}) ...")
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    nllb_tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL_NAME)
    nllb_model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_NAME)

    print(f"[setup] preparing Piper voice ({PIPER_VOICE_NAME}) ...")
    piper_onnx, piper_config = ensure_piper_voice(models_dir)

    return {
        "whisper": whisper_model,
        "nllb_tokenizer": nllb_tokenizer,
        "nllb_model": nllb_model,
        "piper_onnx": piper_onnx,
        "piper_config": piper_config,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="Folder berisi video sumber (Bahasa Inggris)")
    parser.add_argument("--output", required=True, help="Folder tujuan untuk video hasil dubbing")
    parser.add_argument("--whisper-model", default="medium", help="Ukuran model faster-whisper (default: medium)")
    parser.add_argument("--suffix", default="_ID", help="Sufiks nama file output (default: _ID)")
    parser.add_argument("--keep-temp", action="store_true", help="Simpan file sementara untuk debugging")
    args = parser.parse_args()

    source_dir = Path(args.source).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    models_dir = Path(__file__).parent / "models" / "piper"

    if not source_dir.is_dir():
        sys.exit(f"Folder source tidak ditemukan: {source_dir}")

    def video_sort_key(path: Path) -> tuple:
        # Sort by the leading number in the title (e.g. "2. Foo" before
        # "10. Bar"), not plain alphabetical order.
        match = re.match(r"\s*(\d+)", path.stem)
        number = int(match.group(1)) if match else float("inf")
        return (number, path.name)

    videos = sorted(
        (p for p in source_dir.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS),
        key=video_sort_key,
    )
    if not videos:
        sys.exit(f"Tidak ada file video ({', '.join(sorted(VIDEO_EXTS))}) di {source_dir}")

    print(f"Ditemukan {len(videos)} video di {source_dir}")
    models = load_models(args.whisper_model, models_dir)

    ok, failed, skipped = 0, 0, 0
    for i, video_path in enumerate(videos, 1):
        rel = video_path.relative_to(source_dir)
        out_path = (output_dir / rel).with_name(rel.stem + args.suffix + ".mp4")

        print(f"\n[{i}/{len(videos)}] {rel}")
        if out_path.exists():
            print(f"  sudah ada, dilewati -> {out_path}")
            skipped += 1
            continue

        try:
            process_video(video_path, out_path, models, args.keep_temp)
            print(f"  selesai -> {out_path}")
            ok += 1
        except Exception as exc:
            print(f"  GAGAL: {exc}", file=sys.stderr)
            failed += 1

    print(f"\nSelesai. Berhasil: {ok}, dilewati: {skipped}, gagal: {failed}")


if __name__ == "__main__":
    main()
