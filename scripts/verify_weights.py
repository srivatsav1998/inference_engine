import argparse
import json
import os
import numpy as np
from transformers import GPT2Model

def main():
    parser = argparse.ArgumentParser(description="Verify converted weights against HuggingFace PyTorch tensors.")
    default_bin_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "bin"))
    parser.add_argument("--bin-dir", type=str, default=default_bin_dir, help="Directory containing model_weights.bin and model_offsets.json")
    args = parser.parse_args()

    json_path = os.path.join(args.bin_dir, "model_offsets.json")
    bin_path = os.path.join(args.bin_dir, "model_weights.bin")

    if not os.path.exists(json_path) or not os.path.exists(bin_path):
        print(f"Error: Could not find weights in {args.bin_dir}")
        return

    with open(json_path, "r") as f:
        metadata = json.load(f)

    print("Loading HuggingFace model for ground-truth comparison...")
    model = GPT2Model.from_pretrained('gpt2')
    state_dict = model.state_dict()

    layers_to_test = [
        "wte.weight",            # Very first tensor
        "h.5.mlp.c_proj.weight",  # Somewhere in the middle
        "ln_f.bias"              # Very last tensor
    ]

    all_passed = True
    with open(bin_path, "rb") as f:
        for layer_name in layers_to_test:
            layer_meta = metadata[layer_name]
            offset = layer_meta["byte_offset"]
            shape = tuple(layer_meta["shape"])

            num_elements = np.prod(shape)
            f.seek(offset)
            loaded_weights = np.fromfile(f, dtype=np.float32, count=num_elements).reshape(shape)
            original_weights = state_dict[layer_name].numpy()

            is_shape_match = (loaded_weights.shape == original_weights.shape)
            is_exact_match = np.array_equal(loaded_weights, original_weights)

            print(f"Layer: {layer_name}")
            print(f"  Shape matches: {is_shape_match}")
            print(f"  Data exactly matches: {is_exact_match}")

            if not (is_shape_match and is_exact_match):
                all_passed = False

    if all_passed:
        print("\nAll tested layers matched bit-for-bit!")
    else:
        print("\nVerification FAILED on some layers.")

if __name__ == "__main__":
    main()
