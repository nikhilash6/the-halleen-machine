#!/usr/bin/env python3
"""
setup.py - Interactive configuration wizard for The Halleen Machine

Run this once after installation to generate config.toml with your paths.
Alternatively, copy config.toml.example to config.toml and edit manually.

Usage:
    python setup.py              # Interactive mode
    python setup.py --defaults   # Use auto-detected paths, no prompts
    python setup.py --check      # Validate existing config.toml
"""

import sys
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------
IS_WINDOWS = sys.platform == "win32"


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if IS_WINDOWS else 'clear')


def print_header(step: int = None, total_steps: int = 5, step_title: str = None, model_num: int = None, total_models: int = 6):
    """Print consistent wizard header with step progress."""
    print()
    print("=" * 60)
    print("  The Halleen Machine - Setup Wizard")
    print("=" * 60)
    print()
    if step is not None and step_title:
        if model_num is not None:
            # Model sub-step
            print("-" * 60)
            print(f"STEP {step} of {total_steps}: {step_title} ({model_num} of {total_models})")
            print("-" * 60)
        else:
            print("-" * 60)
            print(f"STEP {step} of {total_steps}: {step_title}")
            print("-" * 60)


# ---------------------------------------------------------------------------
# Common ComfyUI installation paths to check
# ---------------------------------------------------------------------------
COMFY_SEARCH_PATHS = [
    # Linux/RunPod
    "/workspace/ComfyUI",
    "/home/ComfyUI",
    str(Path.home() / "ComfyUI"),
    # Windows
    "D:/ComfyUI",
    "C:/ComfyUI",
    str(Path.home() / "ComfyUI"),
    # Relative (portable installs)
    "../ComfyUI",
    "./ComfyUI",
]

# ---------------------------------------------------------------------------
# Model definitions: key -> (default_filename, subfolder, description)
# ---------------------------------------------------------------------------
MODEL_DEFINITIONS = {
    "default_project_model": (
        "sdXL_v10VAEFix.safetensors",
        "checkpoints",
        "Default checkpoint for new projects. A general-purpose SDXL model."
    ),
    "pose_model_fast": (
        "sdXL_v10VAEFix.safetensors",
        "checkpoints",
        "Fast model for basic pose generation. Speed over detail."
    ),
    "pose_model_enhanced": (
        "obsessionIllustrious_v21.safetensors",
        "checkpoints",
        "Expressive model for detailed pose generation. More stylized output."
    ),
    "controlnet_model": (
        "diffusion_pytorch_model_promax.safetensors",
        "controlnet",
        "ControlNet model for pose-based keyframe generation."
    ),
    "inpainting_model": (
        "juggernautxl-inpainting.safetensors",
        "checkpoints",
        "Inpainting model used in the 2CHAR heal process."
    ),
    "upscale_model": (
        "4x_NMKD-Siax_200k.pth",
        "upscale_models",
        "4x upscaling model for enhancing output resolution."
    ),
    # interpolation_model deferred - hardcoded in node
}

# ---------------------------------------------------------------------------
# Video generation models (not in config, just need to exist on disk)
# filename -> subfolder
# ---------------------------------------------------------------------------
VIDEO_MODELS = {
    "umt5_xxl_fp8_e4m3fn_scaled.safetensors": "text_encoders",
    "wan_2.1_vae.safetensors": "vae",
    "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors": "diffusion_models",
    "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors": "diffusion_models",
    "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors": "loras",
    "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors": "loras",
}


def find_comfyui() -> Path | None:
    """Auto-detect ComfyUI installation by looking for main.py."""
    for path_str in COMFY_SEARCH_PATHS:
        path = Path(path_str).resolve()
        if (path / "main.py").exists():
            return path
    return None


def validate_comfyui_path(path: Path) -> tuple[bool, str]:
    """Validate a ComfyUI installation path."""
    if not path.exists():
        return False, f"Path does not exist: {path}"
    if not (path / "main.py").exists():
        return False, f"main.py not found in {path} - is this a ComfyUI installation?"
    return True, "OK"


def derive_paths(comfy_root: Path) -> dict:
    """Derive standard subpaths from ComfyUI root."""
    return {
        "install_path": str(comfy_root),
        "output_root": str(comfy_root / "output"),
        "models_root": str(comfy_root / "models" / "checkpoints"),
        "loras_root": str(comfy_root / "models" / "loras"),
        "controlnet_root": str(comfy_root / "models" / "controlnet"),
        "upscale_root": str(comfy_root / "models" / "upscale_models"),
    }


def check_video_models(comfy_root: Path) -> list[str]:
    """
    Check which video models are missing.
    
    Returns:
        List of missing video model filenames
    """
    missing = []
    models_base = comfy_root / "models"
    
    for filename, subfolder in VIDEO_MODELS.items():
        model_path = models_base / subfolder / filename
        if not model_path.exists():
            missing.append(filename)
    
    return missing


def find_restart_script(comfy_root: Path) -> str | None:
    """Try to auto-detect a restart script in the ComfyUI folder."""
    candidates = [
        "run_nvidia_gpu.bat",
        "run_cpu.bat", 
        "start.bat",
        "restart.bat",
        "start.ps1",
        "run.sh",
        "start.sh",
        "restart.sh",
    ]
    for name in candidates:
        script_path = comfy_root / name
        if script_path.exists():
            return str(script_path)
    return None


def prompt_yes_no(question: str, default: bool = True) -> bool:
    """Prompt for yes/no with default."""
    suffix = " [Y/n]: " if default else " [y/N]: "
    while True:
        answer = input(question + suffix).strip().lower()
        if not answer:
            return default
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("Please answer 'y' or 'n'")


def prompt_choice(question: str, options: list[str], default: str = None) -> str:
    """Prompt for a choice from options."""
    print(question)
    for opt in options:
        print(f"  {opt}")
    
    valid = [o[1] if o.startswith("[") else o[0].lower() for o in options]
    
    while True:
        answer = input("Choice: ").strip().upper()
        if not answer and default:
            return default
        if answer in [v.upper() for v in valid]:
            return answer
        print(f"Please enter one of: {', '.join(valid)}")


def prompt_path(prompt: str, default: str = "", must_exist: bool = False) -> str:
    """Prompt for a path with optional default."""
    if default:
        full_prompt = f"{prompt} [{default}]: "
    else:
        full_prompt = f"{prompt}: "
    
    while True:
        answer = input(full_prompt).strip()
        if not answer and default:
            answer = default
        if not answer:
            print("Path cannot be empty")
            continue
        
        # Don't validate relative paths like ./samples
        if answer.startswith("./"):
            return answer
            
        path = Path(answer).resolve()
        if must_exist and not path.exists():
            print(f"Path does not exist: {path}")
            if not prompt_yes_no("Use it anyway?", default=False):
                continue
        
        return str(path)


def scan_models(folder: Path, extensions: list[str] = None) -> list[str]:
    """Scan a folder for model files."""
    if extensions is None:
        extensions = [".safetensors", ".ckpt", ".pth", ".pt"]
    
    if not folder.exists():
        return []
    
    models = []
    for f in folder.iterdir():
        if f.is_file() and f.suffix.lower() in extensions:
            models.append(f.name)
    
    return sorted(models)


def prompt_model_selection(models: list[str], model_key: str) -> str | None:
    """Show numbered list of models and get selection."""
    print()
    print(f"Available models ({len(models)}):")
    print("-" * 40)
    for i, name in enumerate(models, 1):
        print(f"  {i:3}. {name}")
    print("-" * 40)
    
    while True:
        answer = input("Enter number (or 'c' to cancel): ").strip().lower()
        if answer == 'c':
            return None
        try:
            idx = int(answer)
            if 1 <= idx <= len(models):
                return models[idx - 1]
            print(f"Please enter a number between 1 and {len(models)}")
        except ValueError:
            print("Please enter a number or 'c' to cancel")


def configure_model(
    model_key: str,
    comfy_root: Path,
    paths: dict,
    model_num: int,
    total_models: int
) -> tuple[str, bool]:
    """
    Configure a single model.
    
    Returns: (filename_or_empty, needs_download)
    """
    default_filename, subfolder, description = MODEL_DEFINITIONS[model_key]
    
    # Determine the folder to scan
    if subfolder == "checkpoints":
        model_folder = Path(paths["models_root"])
    elif subfolder == "controlnet":
        model_folder = Path(paths["controlnet_root"])
    elif subfolder == "upscale_models":
        model_folder = Path(paths["upscale_root"])
    else:
        model_folder = comfy_root / "models" / subfolder
    
    # Pretty print the key
    display_name = model_key.replace("_", " ").title()
    
    # Clear screen for each model
    clear_screen()
    print_header(step=5, step_title="Model Configuration", model_num=model_num, total_models=total_models)
    print()
    print("=" * 60)
    print(f"  {display_name}")
    print("=" * 60)
    print(f"  {description}")
    print(f"  Default: {default_filename}")
    print(f"  Location: {model_folder}/")
    print()
    
    # Check if default exists
    default_path = model_folder / default_filename
    if default_path.exists():
        print(f"  ✓ Default model found on disk")
        if prompt_yes_no(f"  Use {default_filename}?"):
            return default_filename, False
        # They want something else, show list
        models = scan_models(model_folder)
        if models:
            selected = prompt_model_selection(models, model_key)
            if selected:
                return selected, False
        return "", False
    
    # Default not found
    print(f"  ✗ Default model not found in {model_folder}/")
    print()
    
    # Check what models exist
    models = scan_models(model_folder)
    
    choice = prompt_choice(
        "  Options:",
        [
            "[D] Download later (command provided at end of setup)",
            "[C] Choose from existing models on disk" + (f" ({len(models)} found)" if models else " (none found)"),
            "[S] Skip (leave unconfigured)",
        ]
    )
    
    if choice == "D":
        return "", True  # needs_download = True
    elif choice == "C":
        if not models:
            print("  No models found in this folder.")
            return "", False
        selected = prompt_model_selection(models, model_key)
        if selected:
            return selected, False
        return "", False
    else:  # S
        return "", False


def generate_config_toml(
    paths: dict,
    workspace: str,
    restart_script: str,
    models: dict,
    api_base: str = "http://127.0.0.1:8188"
) -> str:
    """Generate config.toml content."""
    # Normalize path separators for TOML (always forward slashes)
    def norm(p: str) -> str:
        return p.replace("\\", "/") if p else ""
    
    return f'''# The Halleen Machine - Configuration
# Generated by setup.py

[comfyui]
# Path to your ComfyUI installation
install_path = "{norm(paths['install_path'])}"

# ComfyUI API endpoint
api_base = "{api_base}"

# Timeout for ComfyUI API calls in seconds
timeout_seconds = 300

# Where ComfyUI saves generated images
output_root = "{norm(paths['output_root'])}"

[paths]
# Path to model checkpoints
models = "{norm(paths['models_root'])}"

# Path to LoRA files
loras = "{norm(paths['loras_root'])}"

# Path to ControlNet models
controlnet = "{norm(paths['controlnet_root'])}"

# Path to upscale models
upscale_models = "{norm(paths['upscale_root'])}"

# Workspace for project files
workspace = "{norm(workspace)}"

[backups]
# Number of project backups to keep
retention = 50

# Minimum seconds between auto-backups
throttle_seconds = 300

[models]
# Default checkpoint for new projects
default_project_model = "{models.get('default_project_model', '')}"

# Fast model for basic pose generation
pose_model_fast = "{models.get('pose_model_fast', '')}"

# Expressive model for detailed pose generation
pose_model_enhanced = "{models.get('pose_model_enhanced', '')}"

# ControlNet model for pose-based keyframe generation
controlnet_model = "{models.get('controlnet_model', '')}"

# Inpainting model for 2CHAR heal process
inpainting_model = "{models.get('inpainting_model', '')}"

# 4x upscaling model
upscale_model = "{models.get('upscale_model', '')}"

# Interpolation model (configured in ComfyUI node)
interpolation_model = "rife47.pth"

[advanced]
# Script to restart ComfyUI (optional, enables restart button)
restart_script = "{norm(restart_script)}"
'''


def validate_config(config_path: Path) -> tuple[bool, list[str]]:
    """Validate an existing config.toml file."""
    errors = []
    warnings = []
    
    if not config_path.exists():
        return False, ["config.toml not found"]
    
    # Try to parse it
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        try:
            import tomli as tomllib
        except ImportError:
            return False, ["tomli not installed (required for Python < 3.11): pip install tomli"]
    
    try:
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
    except Exception as e:
        return False, [f"Failed to parse config.toml: {e}"]
    
    # Check required sections
    if "comfyui" not in data:
        errors.append("Missing [comfyui] section")
    else:
        comfy = data["comfyui"]
        if not comfy.get("output_root"):
            errors.append("[comfyui] output_root is empty or missing")
        elif not Path(comfy["output_root"]).exists():
            warnings.append(f"[comfyui] output_root does not exist: {comfy['output_root']}")
    
    if "paths" not in data:
        errors.append("Missing [paths] section")
    else:
        paths = data["paths"]
        if not paths.get("models"):
            errors.append("[paths] models is empty or missing")
        elif not Path(paths["models"]).exists():
            warnings.append(f"[paths] models does not exist: {paths['models']}")
    
    # Check models section
    if "models" in data:
        models = data["models"]
        empty_models = [k for k, v in models.items() if not v and k != "interpolation_model"]
        if empty_models:
            warnings.append(f"Unconfigured models: {', '.join(empty_models)}")
    
    # Print warnings even if valid
    for w in warnings:
        print(f"  ⚠️  {w}")
    
    if errors:
        return False, errors
    return True, []


def run_interactive():
    """Run the interactive setup wizard."""
    clear_screen()
    print()
    print("=" * 60)
    print("  The Halleen Machine - Setup Wizard")
    print("=" * 60)
    print()
    
    config_path = Path("config.toml")
    
    # Check for existing config
    if config_path.exists():
        print(f"Found existing config.toml")
        valid, errors = validate_config(config_path)
        if valid:
            print("  ✓ Configuration is valid!")
            if not prompt_yes_no("Overwrite with new configuration?", default=False):
                print("\nKeeping existing configuration.")
                return 0
        else:
            print("  ✗ Configuration has errors:")
            for e in errors:
                print(f"     - {e}")
            if not prompt_yes_no("Generate new configuration?", default=True):
                print("\nExiting. Fix errors manually or re-run setup.")
                return 1
    
    # =========================================================================
    # STEP 1: ComfyUI Installation
    # =========================================================================
    clear_screen()
    print_header(step=1, step_title="ComfyUI Installation")
    
    detected = find_comfyui()
    if detected:
        print(f"✓ Found ComfyUI at: {detected}")
        if prompt_yes_no("Use this installation?"):
            comfy_root = detected
        else:
            comfy_root = Path(prompt_path("Enter ComfyUI installation path", must_exist=True))
    else:
        print("✗ Could not auto-detect ComfyUI installation.")
        print("  Searched:", ", ".join(COMFY_SEARCH_PATHS[:4]), "...")
        comfy_root = Path(prompt_path("Enter ComfyUI installation path", must_exist=True))
    
    # Validate the chosen path
    valid, msg = validate_comfyui_path(comfy_root)
    if not valid:
        print(f"⚠️  {msg}")
        if not prompt_yes_no("Continue anyway?", default=False):
            return 1
    
    # Derive paths
    paths = derive_paths(comfy_root)
    
    print()
    print("Derived paths:")
    print(f"  Output:      {paths['output_root']}")
    print(f"  Checkpoints: {paths['models_root']}")
    print(f"  LoRAs:       {paths['loras_root']}")
    print(f"  ControlNet:  {paths['controlnet_root']}")
    print(f"  Upscale:     {paths['upscale_root']}")
    print()
    
    if not prompt_yes_no("Accept these paths?"):
        print()
        print("-" * 60)
        print("NOTE: Custom model paths require matching ComfyUI config.")
        print("Ensure your extra_model_paths.yaml points to these locations.")
        print("-" * 60)
        print()
        paths["output_root"] = prompt_path("ComfyUI output directory", paths["output_root"])
        paths["models_root"] = prompt_path("Checkpoints directory", paths["models_root"])
        paths["loras_root"] = prompt_path("LoRAs directory", paths["loras_root"])
        paths["controlnet_root"] = prompt_path("ControlNet directory", paths["controlnet_root"])
        paths["upscale_root"] = prompt_path("Upscale models directory", paths["upscale_root"])
    
    # =========================================================================
    # STEP 2: Workspace
    # =========================================================================
    clear_screen()
    print_header(step=2, step_title="Workspace")
    print("Where should project files be saved?")
    print("Use './samples' to keep them with the app, or an absolute path.")
    print()
    workspace = prompt_path("Workspace directory", default="./samples")
    
    # =========================================================================
    # STEP 3: ComfyUI Restart Script (Optional)
    # =========================================================================
    clear_screen()
    print_header(step=3, step_title="ComfyUI Restart Script (Optional)")
    print("This enables the 'Restart ComfyUI' button in the app.")
    print("Useful when running ComfyUI on a network device (e.g., mobile access).")
    print()
    
    detected_script = find_restart_script(comfy_root)
    restart_script = ""
    
    if detected_script:
        print(f"✓ Found restart script: {detected_script}")
        if prompt_yes_no("Use this script?"):
            restart_script = detected_script
    else:
        print("No restart script auto-detected.")
    
    if not restart_script:
        if prompt_yes_no("Configure a restart script?", default=False):
            restart_script = prompt_path("Path to restart script", must_exist=True)
        else:
            print("Skipping - restart button will be disabled.")
    
    # =========================================================================
    # STEP 4: ComfyUI API
    # =========================================================================
    clear_screen()
    print_header(step=4, step_title="ComfyUI API")
    api_base = "http://127.0.0.1:8188"
    print(f"ComfyUI API URL: {api_base}")
    if not prompt_yes_no("Use default API URL?"):
        api_base = input("Enter ComfyUI API URL: ").strip() or api_base
    
    # =========================================================================
    # STEP 5: Model Configuration
    # =========================================================================
    clear_screen()
    print_header(step=5, step_title="Model Configuration")
    print("Configure which models to use for each purpose.")
    print("If you don't have the default models, you can download them later")
    print("or select from models already on your system.")
    print()
    print("Note: Leaving models unconfigured will limit some features.")
    
    configured_models = {}
    models_to_download = []
    
    model_keys = list(MODEL_DEFINITIONS.keys())
    total_models = len(model_keys)
    
    for model_num, model_key in enumerate(model_keys, 1):
        filename, needs_download = configure_model(model_key, comfy_root, paths, model_num, total_models)
        configured_models[model_key] = filename
        
        if needs_download:
            default_filename, subfolder, _ = MODEL_DEFINITIONS[model_key]
            models_to_download.append((model_key, default_filename, subfolder))
    
    # =========================================================================
    # Generate and write config
    # =========================================================================
    config_content = generate_config_toml(
        paths=paths,
        workspace=workspace,
        restart_script=restart_script,
        models=configured_models,
        api_base=api_base
    )
    
    clear_screen()
    print()
    print("=" * 60)
    print("  The Halleen Machine - Setup Wizard")
    print("=" * 60)
    print()
    print("-" * 60)
    print("Configuration Summary")
    print("-" * 60)
    print()
    print(config_content)
    print("=" * 60)
    
    if prompt_yes_no("Write this to config.toml?"):
        # Backup existing if present
        if config_path.exists():
            backup_path = config_path.with_suffix(".toml.bak")
            config_path.replace(backup_path)  # replace() overwrites on Windows
            print(f"  Backed up existing config to {backup_path}")
        
        config_path.write_text(config_content)
        print(f"\n✓ Configuration saved to {config_path}")
    else:
        print("\nConfiguration not saved.")
        return 1
    
    # =========================================================================
    # Download instructions (if any)
    # =========================================================================
    
    # Check for missing video models
    missing_video_models = check_video_models(comfy_root)
    
    if models_to_download or missing_video_models:
        clear_screen()
        print()
        print("=" * 60)
        print("  The Halleen Machine - Setup Wizard")
        print("=" * 60)
        print()
        print("-" * 60)
        print("Models to Download")
        print("-" * 60)
        print()
        
        if models_to_download:
            print("CONFIG MODELS (marked for download):")
            print()
            
            for model_key, filename, subfolder in models_to_download:
                display_name = model_key.replace("_", " ").title()
                print(f"  • {display_name}")
                print(f"    File: {filename}")
                print(f"    Destination: models/{subfolder}/")
                print()
            
            print("These fields are left blank in config.toml until files exist.")
            print()
        
        if missing_video_models:
            print("VIDEO GENERATION MODELS (required for i2v workflows):")
            print()
            
            for filename in missing_video_models:
                subfolder = VIDEO_MODELS[filename]
                print(f"  • {filename}")
                print(f"    Destination: models/{subfolder}/")
            print()
        
        # Build download command
        # Group by subfolder for cleaner command, dedupe filenames
        checkpoints = list(dict.fromkeys(f for k, f, s in models_to_download if s == "checkpoints"))
        controlnet = list(dict.fromkeys(f for k, f, s in models_to_download if s == "controlnet"))
        upscale = list(dict.fromkeys(f for k, f, s in models_to_download if s == "upscale_models"))
        
        cmd_parts = ["python download_models.py"]
        if checkpoints:
            cmd_parts.append(f"--checkpoints {' '.join(checkpoints)}")
        if controlnet:
            cmd_parts.append(f"--controlnet {' '.join(controlnet)}")
        if upscale:
            cmd_parts.append(f"--upscale {' '.join(upscale)}")
        if missing_video_models:
            cmd_parts.append("--video")
        
        print("To download, run:")
        print()
        print(f"  {' '.join(cmd_parts)}")
        print()
        print("Or download manually. See MODELS.md for download links and more info.")
        
        # Only need to re-run setup if config models were deferred
        if models_to_download:
            print()
            print("After downloading, re-run:")
            print("  python setup.py")
        print()
    
    print()
    print("Setup complete! Run: python app.py")
    return 0


def run_defaults():
    """Run with auto-detected defaults, no prompts."""
    print("Running setup with defaults...")
    
    detected = find_comfyui()
    if not detected:
        print("✗ Could not auto-detect ComfyUI installation.")
        print("  Run 'python setup.py' for interactive mode.")
        return 1
    
    print(f"✓ Found ComfyUI at: {detected}")
    
    paths = derive_paths(detected)
    restart_script = find_restart_script(detected) or ""
    
    # Check for default models
    configured_models = {}
    for model_key, (default_filename, subfolder, _) in MODEL_DEFINITIONS.items():
        if subfolder == "checkpoints":
            model_folder = Path(paths["models_root"])
        elif subfolder == "controlnet":
            model_folder = Path(paths["controlnet_root"])
        elif subfolder == "upscale_models":
            model_folder = Path(paths["upscale_root"])
        else:
            model_folder = detected / "models" / subfolder
        
        if (model_folder / default_filename).exists():
            configured_models[model_key] = default_filename
        else:
            configured_models[model_key] = ""
    
    config_content = generate_config_toml(
        paths=paths,
        workspace="./samples",
        restart_script=restart_script,
        models=configured_models
    )
    
    config_path = Path("config.toml")
    if config_path.exists():
        backup_path = config_path.with_suffix(".toml.bak")
        config_path.replace(backup_path)  # replace() overwrites on Windows
        print(f"  Backed up existing config to {backup_path}")
    
    config_path.write_text(config_content)
    print(f"✓ Configuration saved to {config_path}")
    
    # Report unconfigured models
    unconfigured = [k for k, v in configured_models.items() if not v]
    missing_video = check_video_models(detected)
    
    if unconfigured or missing_video:
        print()
        
        if unconfigured:
            print("⚠️  Some config models were not found and left unconfigured:")
            for k in unconfigured:
                print(f"   - {k}")
        
        if missing_video:
            print()
            print("⚠️  Some video generation models are missing:")
            for f in missing_video:
                print(f"   - {f}")
        
        print()
        print("Run 'python setup.py' to configure interactively, or:")
        
        # Build download command
        cmd_parts = ["python download_models.py"]
        if unconfigured:
            cmd_parts.append("--all")
        if missing_video:
            cmd_parts.append("--video")
        print(f"  {' '.join(cmd_parts)}")
    
    return 0


def run_check():
    """Validate existing config.toml."""
    print("Validating config.toml...")
    print()
    
    config_path = Path("config.toml")
    valid, errors = validate_config(config_path)
    
    if valid:
        print("✓ Configuration is valid!")
        return 0
    else:
        print("✗ Configuration errors:")
        for e in errors:
            print(f"   - {e}")
        return 1


def main():
    args = sys.argv[1:]
    
    if "--help" in args or "-h" in args:
        print(__doc__)
        return 0
    
    if "--defaults" in args:
        return run_defaults()
    
    if "--check" in args:
        return run_check()
    
    return run_interactive()


if __name__ == "__main__":
    sys.exit(main())
