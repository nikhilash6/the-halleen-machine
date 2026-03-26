# Registry for High/Low noise LoRA pairs. 
# If a prompt LoRA matches a trigger, the specific high/low files are injected into their respective paths.
# Not necessary to edit this file if your high and low loras match the naming pattern ending in "_high_noise" and "_low_noise"
LORA_REGISTRY = [
{ "triggers": ["WAN22-LORA-flood-high", ""], "high": "WAN22-LORA-flood-high", "low": "" },
{ "triggers": ["WAN22-LORA-hbz-hydrogen-bomb-mushroom-cloud-hbz-22", ""], "high": "WAN22-LORA-hbz-hydrogen-bomb-mushroom-cloud-hbz-22", "low": "" },
{ "triggers": ["Wan22-LORA-high-waist-dance-V5", ""], "high": "Wan22-LORA-high-waist-dance-V5", "low": "" },
]
