# VoiceFlow

Local AI dictation for macOS. Hold a key, speak, release — your words appear as text in any app. All processing happens on-device using Apple Silicon ML acceleration.

## Install

### Option A: Pre-built App (non-technical users)

1. Download `VoiceFlow.dmg` from the latest release
2. Open the DMG and drag **VoiceFlow** to your Applications folder
3. Launch VoiceFlow from Applications
4. Grant **Microphone** and **Accessibility** permissions when prompted
5. On first launch, models will download automatically (~2 GB one-time download)

### Option B: From Source (developers)

```bash
git clone <repo-url> && cd wisprflow-clone
python -m venv .venv && source .venv/bin/activate
pip install -e .
voiceflow
```

### Building the .app

```bash
pip install pyinstaller
./scripts/build.sh
# Output: dist/VoiceFlow.app
```

Install `create-dmg` (`brew install create-dmg`) to auto-generate a DMG installer with drag-to-Applications.

## Usage

1. **Hold Right Cmd** (default) to start recording — a floating pill appears at the bottom of your screen
2. **Speak** your text naturally
3. **Release** to transcribe and insert text at your cursor

The overlay shows:
- **Recording** (pulsing red dot) — capturing audio
- **Processing...** — transcribing and refining
- Hidden — idle, ready for next dictation

### Accuracy Modes

Click the **VF** menu bar icon to switch modes:

| Mode | Speed | Quality | Uses |
|------|-------|---------|------|
| Fast | ~1s | Good | Quick notes, messages |
| Standard | ~2s | Better | Code comments, docs |
| Max Accuracy | ~3s | Best | Precise technical dictation |

### Self-Correction

VoiceFlow's LLM understands natural self-corrections. No need to start over:

| You say | VoiceFlow types |
|---------|----------------|
| "change it to red, sorry, blue" | "change it to blue" |
| "call the function foo, actually bar" | "call the function bar" |
| "set font size to 12, no wait, 14 pixels" | "set the font size to 14 pixels" |
| "delete the file, scratch that, just rename it" | "just rename it" |

Supported correction phrases: "sorry", "I mean", "actually", "no wait", "scratch that", "never mind that", "let me rephrase", "correction".

### Custom Vocabulary

Edit `~/.voiceflow/dictionary.json` to add domain-specific terms:

```json
{
  "pie torch": "PyTorch",
  "cuda": "CUDA",
  "j query": "jQuery"
}
```

## Troubleshooting

**"Accessibility Permission Required" notification**
Open System Settings > Privacy & Security > Accessibility > enable VoiceFlow. Restart the app.

**No text appears after dictation**
Ensure the target app accepts keyboard input. Some apps require Accessibility permission for VoiceFlow to insert text.

**Models downloading on every launch**
Models are cached in `~/.cache/huggingface/`. Ensure this directory is writable and has ~2 GB free space.

**Build fails with missing `mlx_whisper` assets**
The `VoiceFlow.spec` includes `mlx_whisper/assets` as data files. Ensure `mlx-whisper` is installed: `pip install mlx-whisper`.

## Requirements

- macOS 13.0+
- Apple Silicon (M1/M2/M3/M4)
- ~2 GB disk space for models (downloaded on first launch)
