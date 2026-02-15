#!/usr/bin/env bash
#
# VoiceFlow — One-command installer for macOS
#
# Usage:
#   ./scripts/install.sh            # Full install (all models)
#   ./scripts/install.sh --light    # Skip secondary Parakeet model (~650MB smaller)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC}   $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERR]${NC}  $*" >&2; }

# ── Pre-flight checks ───────────────────────────────────────────────────────

echo ""
echo "=================================================="
echo "  VoiceFlow Installer"
echo "=================================================="
echo ""

# Check macOS
if [[ "$(uname)" != "Darwin" ]]; then
    error "VoiceFlow only runs on macOS."
    exit 1
fi

# Check Apple Silicon
ARCH="$(uname -m)"
if [[ "$ARCH" != "arm64" ]]; then
    warn "VoiceFlow is optimized for Apple Silicon (M1/M2/M3/M4)."
    warn "Running on $ARCH — MLX models may not work."
fi

# Check Python 3.10+
if ! command -v python3 &>/dev/null; then
    error "Python 3 not found. Install it: brew install python@3.12"
    exit 1
fi

PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
if [[ "$PY_MAJOR" -lt 3 || "$PY_MINOR" -lt 10 ]]; then
    error "Python 3.10+ required (found $PY_VERSION). Install: brew install python@3.12"
    exit 1
fi
success "Python $PY_VERSION found"

# ── Virtual environment ──────────────────────────────────────────────────────

VENV_DIR="$PROJECT_DIR/.venv"

if [[ -d "$VENV_DIR" ]]; then
    info "Virtual environment already exists at $VENV_DIR"
else
    info "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    success "Virtual environment created"
fi

# Activate venv
source "$VENV_DIR/bin/activate"
success "Virtual environment activated"

# ── Install dependencies ─────────────────────────────────────────────────────

info "Installing Python dependencies..."
pip install --upgrade pip --quiet
pip install -e "$PROJECT_DIR" --quiet
success "Dependencies installed"

# ── Download models ──────────────────────────────────────────────────────────

DOWNLOAD_FLAGS=""
if [[ "${1:-}" == "--light" ]]; then
    DOWNLOAD_FLAGS="--skip-max-accuracy"
    info "Light mode: skipping secondary Parakeet model"
fi

info "Downloading AI models (this may take a while on first run)..."
python "$SCRIPT_DIR/download_models.py" $DOWNLOAD_FLAGS
success "All models downloaded"

# ── Microphone permission hint ───────────────────────────────────────────────

echo ""
echo "=================================================="
echo "  Setup Complete!"
echo "=================================================="
echo ""
info "Before running VoiceFlow, grant these macOS permissions:"
echo ""
echo "  1. Microphone Access"
echo "     System Settings > Privacy & Security > Microphone"
echo ""
echo "  2. Accessibility Access"
echo "     System Settings > Privacy & Security > Accessibility"
echo ""
echo "  (You'll be prompted on first launch)"
echo ""
echo "  To start VoiceFlow:"
echo "    source .venv/bin/activate"
echo "    python -m app.main"
echo ""
success "VoiceFlow is ready!"
