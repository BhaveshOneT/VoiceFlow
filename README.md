# VoiceFlow

Local dictation app for macOS (Apple Silicon).  
Hold `Right Cmd`, speak, release, and VoiceFlow inserts cleaned text into the focused app.

## What It Does

- Real-time speech-to-text with local Whisper (`mlx-whisper`)
- Smart post-processing with local LLM (`Qwen2.5-3B-4bit`)
- Two transcription modes:
  - `Programmer Mode` (code-aware terms, file tagging)
  - `Normal Mode` (clean dictation without code-specific tagging)
- Optional auto mode switching by active app (`Terminal`, `Codex`, `Claude`, editors)
- Built-in cleanup for filler words and self-corrections (`"no no", "sorry", "I mean"`)
- File mention tagging, for example:
  - `"update function.py file"` -> `@function.py`
  - `"modify text refiner file"` -> `@text_refiner`
- Language modes:
  - `Auto (English + German)`
  - `English`
  - `German`

## Install (Team / Non-Technical)

1. Download `VoiceFlow.dmg` from Releases.
2. Open the DMG.
3. Drag `VoiceFlow.app` into `Applications`.
4. Launch `VoiceFlow` from `Applications`.
5. Grant permissions when prompted:
   - Microphone
   - Accessibility

After that, hold `Right Cmd` and dictate in any text field.

## Build From Source

```bash
git clone https://github.com/BhaveshOneT/VoiceFlow.git
cd VoiceFlow
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
./scripts/build.sh
```

Build outputs:

- `dist/VoiceFlow.app`
- `dist/VoiceFlow.dmg`

`build.sh` creates a DMG even without `create-dmg` by falling back to `hdiutil`.

### Signed + Notarized Release Build (recommended)

```bash
export VOICEFLOW_CODESIGN_IDENTITY="Developer ID Application: YOUR NAME (TEAMID)"
export VOICEFLOW_NOTARY_PROFILE="voiceflow-notary-profile"
./scripts/build.sh
```

If env vars are not set, build still works and skips signing/notarization.

## Runtime Notes

- Hotkey becomes available as soon as the speech model is ready.
- LLM refiner loads in background and does not block recording startup.
- Long recordings are transcribed in overlap-safe chunks with tail-pass recovery.
- First launch may download model files to `~/.cache/huggingface/`.
- VoiceFlow logs stage timings for post-record latency:
  - `capture_stop_ms`
  - `Pipeline timings (ms): total, stt, clean, refine, finalize`
  - `End-to-end post-record timings (ms): pipeline, paste, total`

## Security Notes

- Transcript text is redacted from logs by default.
- To temporarily include raw transcript logs for debugging only:
  - `VOICEFLOW_LOG_TRANSCRIPTS=1`
- `scripts/download_models.py` uses pinned model revisions and verifies the Silero VAD checksum.

## Troubleshooting

### Right Cmd does nothing

- Enable Accessibility:
  - `System Settings -> Privacy & Security -> Accessibility`
- Quit and relaunch VoiceFlow.

### No text inserted

- Keep VoiceFlow in Accessibility list and enabled.
- Some apps block synthetic paste in secure fields.

### Models keep downloading

- Ensure `~/.cache/huggingface/` is writable.
- Ensure enough free disk space.

### macOS blocks app (unverified developer)

If Gatekeeper blocks launch on an unsigned build, run:

```bash
xattr -dr com.apple.quarantine /Applications/VoiceFlow.app
open /Applications/VoiceFlow.app
```

This works the same on macOS devices as long as VoiceFlow is installed in
`/Applications`.

### Logs

Log file:

`~/Library/Application Support/VoiceFlow/logs/voiceflow.log`

Quick latency summary:

```bash
grep -E "Pipeline timings|End-to-end post-record timings|capture_stop_ms" \
  ~/Library/Application\ Support/VoiceFlow/logs/voiceflow.log | tail -n 40
./scripts/latency_report.py
```

## CI Quality Gates

GitHub Actions runs quality gates on push/PR to `main`:

- `ruff` lint checks
- unit tests
- `bandit` security scan
- `pip-audit` dependency vulnerability scan

## Requirements

- macOS 13+
- Apple Silicon (`M1`, `M2`, `M3`, `M4`)
