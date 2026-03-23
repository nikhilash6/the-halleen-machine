#!/bin/bash

# ==============================================================================
# 🛠️ THE HALLEEN MACHINE/COMFYUI RUNPOD INSTALLATION BIBLE (MARCH 2026)
# ==============================================================================
# This guide provides a clean, virtual-environment-based setup for RunPod 
# Future updates to ComfyUI may require further changes.
# Start a pod with attached storage, assumed to be /workspace
# Open ports 8188,7860
# ==============================================================================

# --- PHASE 0: COMFYUI INSTALLATION - SKIP if you already have ComfyUI running  ---
# Update and install critical OS-level dependencies (Fixes CV2, GL, and FFmpeg issues)
cd workspace
apt update && apt install -y \
    python3-venv \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    build-essential \
    git \
    wget

# Clone the official ComfyUI repository
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# Create and activate the virtual environment to keep the system clean
python3 -m venv venv
source venv/bin/activate

# Upgrade pip and install the Core Engine (Torch + Base Requirements)
pip install --upgrade pip
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Install high-level management and hardware monitoring tools
# 'nvidia-ml-py' silences deprecated pynvml warnings in the console
pip install opencv-python-headless GitPython nvidia-ml-py

# install one model
# Navigate to the checkpoints folder and download the SDXL test model
cd models/checkpoints
# Using -O to ensure the filename ends in .safetensors (required for UI visibility)
wget -O sdXL_v10VAEFix.safetensors https://madels-for-machine.s3.us-west-2.amazonaws.com/sdXL_v10VAEFix.safetensors
cd ../..


# Launch with --highvram for RTX 4090 performance 
# The server will be accessible via RunPod's HTTP Port 8188
python3 main.py --listen 0.0.0.0 --port 8188 --highvram


## this should get you to a purple bottle, any problems, seek help with your LLM of choice and resources on YouTube, getting ComfyUI running can be complicated, but there is an amazing helpful community for you to rely on





# --- PHASE 1: INSTALL THE HALLEEN MACHINE  - Do not start if you do not have ComfyUI set up ---

cd /workspace

# Clone repository
git clone https://github.com/mikehalleen/the-halleen-machine.git
cd the-halleen-machine

# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# run setup.py to configure The Halleen Machine to use your ComfyUI instances and scan for models needed
python setup.py

# the setup script will give you a download command to run for your environment, or you can use this one to download all the default models
python download_models.py --all

# optional: start the machine to see what works, most failures will be due to needed custom nodes that need to be installed next
python app.py --listen 0.0.0.0 --port 7860





# --- PHASE 2: UPDATE COMFYUI WITH CUSTOM NODES ---

# deactivate current venv
deactivate

cd /workspace/ComfyUI
source venv/bin/activate

cd custom_nodes

# Clone the Manager and essential node packs
# Note: --recursive is used for Impact-Pack to ensure submodules are included
git clone https://github.com/ltdrdata/ComfyUI-Manager.git
git clone https://github.com/Fannovel16/comfyui_controlnet_aux
git clone https://github.com/rgthree/rgthree-comfy
git clone https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes
git clone https://github.com/yolain/ComfyUI-Easy-Use
git clone https://github.com/crystian/ComfyUI-Crystools
git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts
git clone --recursive https://github.com/ltdrdata/ComfyUI-Impact-Pack
git clone https://github.com/kijai/ComfyUI-KJNodes
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite
git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation


# 2CHAR specific -- optional if you don't want my 2CHAR workflow
git clone https://github.com/chflame163/ComfyUI_LayerStyle
git clone https://github.com/jags111/efficiency-nodes-comfyui
git clone https://github.com/cubiq/ComfyUI_essentials
git clone https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch
git clone https://github.com/melMass/comfy_mtb
git clone https://github.com/WASasquatch/was-node-suite-comfyui




# BATCH INSTALL: Automatically find and install requirements for ALL cloned nodes
cd /workspace/ComfyUI
cp custom_nodes/ComfyUI-Manager/requirements.txt ./manager_requirements.txt
./venv/bin/python3 -m pip install -r ./manager_requirements.txt
find custom_nodes -name "requirements.txt" -exec ./venv/bin/pip install -r {} \;


# Launch with --highvram for RTX 4090 performance and --enable-manager for the UI extension
# The server will be accessible via RunPod's HTTP Port 8188

# optional, make sure you're in the right place
deactivate
cd /workspace/ComfyUI
source venv/bin/activate

# restart comfy with the custom nodes in place
python3 main.py --listen 0.0.0.0 --port 8188 --highvram --enable-manager




# --- PHASE 3: TEST WORKFLOWS IN COMFY---

cp samples/workflows_openincomfy/THM*.json /workspace/ComfyUI/user/default/workflows/

# Now go to Comfy UI and open the workflows that start with "THM", they should be in your default folder.
# One by one, open them and resolve any errors, missing nodes or models, vram configruation etc. Iterate until you can run the default prompts to get to a finished result.  Key values will be overridden by The Halleen Machine.  This is only to prove the nodes and models needed are in place.  
# Editing these workflows will not change the ones run by The Halleen Machine, this step is for validation only. 



# --- PHASE 4: RUN THE HALLEEN MACHINE ---
# optional, make sure you're in the right place
deactivate
cd /workspace/the-halleen-machine
source venv/bin/activate

# start the app if needed
python app.py --listen 0.0.0.0 --port 7860

# find the url in the RunPod panel running on port 7860



