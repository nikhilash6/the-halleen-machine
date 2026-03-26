# The Halleen Machine

**Version 0.9.5 Beta**  
Workflow management system for AI video generation using ComfyUI.

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Organize AI video projects into structured timelines. Create reusable asset libraries for poses, characters, locations, and styles. Batch operations for generation, upscaling, and exports.

---

## Requirements

- Python 3.11+ 
- ComfyUI installed and running
- Custom ComfyUI nodes installed, see COMFYUI_INSTALL_GUIDE.md

## If you do not have ComfyUI running, refer instead to COMFYUI_INSTALL_GUIDE.md which covers installing both ComfyUI and The Halleen Machine


---

## Basic Installation

```bash
# Clone repository
git clone https://github.com/mikehalleen/the-halleen-machine.git
cd the-halleen-machine

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure
python setup.py
# or
cp config.toml.example config.toml
# Then edit config.toml with your ComfyUI paths

# Setup may recommend additional model dowloads and will give instructions

# Refer to COMFYUI_INSTALL_GUIDE.md Phase 2 and 3 for dependencies


# Launch
python app.py --listen 0.0.0.0 --port 7860
```

---


## Learning More

**Tutorials and guides:**  (coming soon)
https://www.youtube.com/halleen

---

## Beta Status

- Minor bugs (coming in v1.0)
- Pre-launch (before samples and how-to videos have been published)

---

## License

GNU Affero General Public License v3.0 (AGPL-3.0)

- ✅ Use, modify, and distribute freely
- ⚠️ Must disclose source code of modified versions
- ⚠️ Network use counts as distribution (must share source)

See [LICENSE](LICENSE) for full terms.
See [MODELS.md](MODELS) for model attribution.
