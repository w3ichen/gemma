"""
Github: https://github.com/Blaizzy/mlx-vlm
Models: https://huggingface.co/collections/mlx-community/gemma-3n-685d6c8d02d7486c7e77a7dc
"""

from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
import time

# Load multi-modal model
model_path = "mlx-community/gemma-3n-E2B-it-4bit"
model, processor = load(model_path)
config = model.config

# Prepare inputs
image = ["/Users/weichen/Downloads/cats.jpg"]
audio = ["/Users/weichen/Downloads/ask.wav"]
prompt = "Tell me what hear"

# Start timer
start_time = time.time()

# Apply chat template
print("Applying chat template")
formatted_prompt = apply_chat_template(
    processor, config, prompt,
    # num_images=len(image),
    num_audios=len(audio)
)

# Generate output
print("Generating output")
output = generate(model, 
                  processor, 
                  formatted_prompt, 
                #   image, 
                  audio=audio, 
                  verbose=False,
                  num_audios=len(audio)
)
print(output)

# End timer
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")