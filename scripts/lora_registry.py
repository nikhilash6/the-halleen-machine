# Registry for High/Low noise LoRA pairs. 
# If a prompt LoRA matches a trigger, the specific high/low files are injected into their respective paths.
LORA_REGISTRY = [
{ "triggers": ["WAN22-LORA-flood-high", ""], "high": "WAN22-LORA-flood-high", "low": "" },
{ "triggers": ["WAN22-LORA-hbz-hydrogen-bomb-mushroom-cloud-hbz-22", ""], "high": "WAN22-LORA-hbz-hydrogen-bomb-mushroom-cloud-hbz-22", "low": "" },
{ "triggers": ["Wan22-LORA-high-waist-dance-V5", ""], "high": "Wan22-LORA-high-waist-dance-V5", "low": "" },
{ "triggers": ["", "Wan22-PlantGrowth-low-byFractalDesign"], "high": "", "low": "Wan22-PlantGrowth-low-byFractalDesign" },
{ "triggers": ["", "wan2.2_14b_i2v_counterside_run_000000750_low_noise"], "high": "", "low": "wan2.2_14b_i2v_counterside_run_000000750_low_noise" },
]
