# VoiceFlow — macOS AI Dictation & Meeting Transcription

PySide6 + AppKit hybrid desktop app. MLX-accelerated Whisper on Apple Silicon. Local-first (no cloud required).
macOS 13+ only. Entry point: `python -m app.main` → `VoiceFlowApplication`.

## Architecture Index

```
app/main.py                  | VoiceFlowApplication: PySide6 orchestrator, ~900 LOC, tray+window+overlay
app/config.py                | AppConfig dataclass → ~/Library/Application Support/VoiceFlow/config.json
app/dictionary.py            | Custom vocabulary for transcription hints
app/core/signals.py          | AppSignals(QObject): Qt signals for ALL backend→UI communication
app/core/model_manager.py    | Model download/warm-up lifecycle
app/core/audio_device_manager.py | System audio device enumeration
app/audio/capture.py         | AudioCapture: sounddevice InputStream → numpy buffer
app/audio/vad.py             | Silero ONNX voice activity detection
app/audio/meeting_recorder.py| Long-form WAV recording for meetings
app/audio/file_importer.py   | Import existing audio files for transcription
app/transcription/whisper_engine.py | mlx_whisper.transcribe() wrapper, warm-up, language support
app/transcription/text_cleaner.py   | File tagging (@file.py), filler removal, self-correction, symbol tags
app/transcription/text_refiner.py   | Optional LLM post-processing (Qwen2.5-3B via mlx-lm)
app/transcription/diarization.py    | Optional speaker diarization (pyannote.audio, NOT installed by default)
app/transcription/meeting_transcriber.py | Chunked Whisper transcription with progress signals
app/transcription/meeting_summarizer.py  | Meeting summary via local LLM or OpenAI API
app/input/hotkey.py          | NSEvent global monitor, push-to-talk + toggle modes, 50ms debounce
app/input/text_inserter.py   | Clipboard save → set text → CGEvent Cmd+V → restore clipboard
app/storage/database.py      | SQLite WAL mode, thread-safe writes, ~/...VoiceFlow/meetings.db
app/storage/models.py        | MeetingStatus enum, Meeting/Speaker/TranscriptSegment/MeetingSummary dataclasses
app/storage/audio_store.py   | WAV file management for meeting recordings
app/ui/overlay.py            | AppKit NSPanel (NOT Qt!) — recording pill at bottom-center of screen
app/ui/recording_messages.py | Overlay status message strings
app/ui_qt/main_window.py     | QMainWindow with stacked pages
app/ui_qt/pages/             | dashboard, meetings, meeting_detail, recording, settings, onboarding
app/ui_qt/widgets/           | device_selector, model_status, recording_controls, search_bar, transcript_view, waveform
app/ui_qt/styles/            | theme.py + stylesheets.py (QSS)
```

## Two Separate Pipelines

**Dictation** (push-to-talk / toggle hotkey):
```
hotkey press → AudioCapture(sounddevice) → WhisperEngine(mlx_whisper) → TextCleaner → TextRefiner(optional LLM) → TextInserter(clipboard+Cmd+V)
```

**Meeting** (from RecordingPage UI):
```
RecordingPage → MeetingRecorder(WAV) → MeetingTranscriber(chunked Whisper + optional diarization) → MeetingSummarizer(LLM/OpenAI) → SQLite DB
```

## Critical Patterns

- **Qt signals only** (`core/signals.py`) for backend→UI. Never call UI methods from background threads directly.
- **`AppHelper.callAfter`** for background→AppKit overlay. NSPanel methods must run on AppKit main thread.
- **`_MainThreadInvoker`** (QObject) for background→Qt main thread dispatching.
- **Single-instance lock**: PID file at `~/Library/Application Support/VoiceFlow/.lock`. Always check for stale processes.
- **`NSApp.setActivationPolicy_(0)`** must be called BEFORE `QApplication()` for proper Dock/window behavior.
- **Config persistence**: `AppConfig.save()` writes JSON. Fields include `transcription_mode`, `recording_mode`, `hotkey`, `whisper_model`, `llm_model`, `language`, meeting settings.

## Gotchas — Don't Do This

1. **Multiple VoiceFlow processes** cause text duplication, GPU contention, clipboard races. Always `ps aux | grep VoiceFlow` before debugging. Kill `/Applications/VoiceFlow.app` during dev.
2. **`pkill -f`** doesn't reliably kill all Python subprocesses — verify with `ps aux`.
3. **pyannote.audio is runtime-optional** (not in default deps). `SpeakerDiarizer` is always `Optional` — guard all diarization code.
4. **NSEvent hotkey handler** has 50ms debounce (`_DEBOUNCE_S = 0.05`) for Qt's duplicate `NSFlagsChanged` events. Don't remove it.
5. **`_processing` lock** must stay `True` through the entire paste operation, not just transcription. Releasing early causes race conditions.
6. **WhisperEngine prompt echo** — the engine sometimes outputs the `initial_prompt` verbatim. The transcription pipeline has blocklist detection for this.
7. **TextCleaner file tagging** requires a nearby action cue (update/modify/file/path/etc.) to avoid false-positive `@` tags on normal words.
8. **LLM refinement** is gated to short-to-medium dictations only. The refiner rejects: prompt leakage, answer-like responses, >2x word expansion, >45% novel keywords.
9. **SQLite WAL mode** is required — enables concurrent reads during live transcription writes. WAL checkpoint on close.
10. **PyInstaller** must `collect_all("mlx")`, `collect_all("mlx_lm")`, `collect_all("mlx_whisper")` and bundle `mlx_whisper/assets/*.json` tokenizer files.
11. **`LSUIElement` must be `False`** (regular app with Dock icon, not background agent).
12. **Info.plist** must have `NSMicrophoneUsageDescription` + `NSAppleEventsUsageDescription` or macOS will silently deny permissions.

## Build & Run

```bash
# Dev
python -m app.main

# Build .app bundle
bash scripts/build.sh          # → dist/VoiceFlow.app

# Install
cp -R dist/VoiceFlow.app /Applications/

# Optional: code signing
VOICEFLOW_CODESIGN_IDENTITY="Developer ID" bash scripts/build.sh

# Optional: notarization
VOICEFLOW_NOTARY_PROFILE="profile-name" bash scripts/build.sh
```

**Paths:**
- Config: `~/Library/Application Support/VoiceFlow/config.json`
- Database: `~/Library/Application Support/VoiceFlow/meetings.db`
- PID lock: `~/Library/Application Support/VoiceFlow/.lock`
- Audio store: `~/Library/Application Support/VoiceFlow/recordings/`
- Spec: `VoiceFlow.spec` (PyInstaller config)

## Key Dependencies

**Core:** PySide6, mlx-whisper, mlx-lm, sounddevice, numpy, pyobjc-framework-{Cocoa,Quartz,ApplicationServices}, onnxruntime, huggingface-hub, soundfile
**Optional:** pyannote.audio + torch (speaker diarization), OpenAI API key (cloud transcription/summarization)
**Build:** PyInstaller, create-dmg (optional)
