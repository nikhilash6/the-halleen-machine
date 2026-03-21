# The Halleen Machine

**Version 0.9.2 Beta**  
Workflow management system for AI video generation using ComfyUI.

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Organize AI video projects into structured timelines. Create reusable asset libraries for poses, characters, locations, and styles. Batch operations for generation, upscaling, and exports.

---

## Requirements

- Python 3.11+ 
- ComfyUI installed and running


---

## Installation

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
cp config.toml.example config.toml
# Edit config.toml with your ComfyUI paths

# Launch
python app.py --listen 0.0.0.0 --port 7860
```

---

## Configuration

Edit `config.toml` with your paths:

```toml
[comfyui]
install_path = "C:/ComfyUI"
api_base = "http://127.0.0.1:8188"
output_root = "C:/ComfyUI/output"

[paths]
models = "C:/ComfyUI/models/checkpoints"
loras = "C:/ComfyUI/models/loras"
workspace = "./samples"
```

See `config.toml.example` for all options.

---

## Learning More

**Tutorials and guides:**  
https://www.youtube.com/halleen

---

## Beta Status

- ✅ Windows tested and supported
- ❓ Linux minimally tested
- Minor bugs, no detailed installation guide (coming in v1.0)

---

## License

GNU Affero General Public License v3.0 (AGPL-3.0)

- ✅ Use, modify, and distribute freely
- ⚠️ Must disclose source code of modified versions
- ⚠️ Network use counts as distribution (must share source)

See [LICENSE](LICENSE) for full terms.
