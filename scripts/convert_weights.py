import argparse
import json
import os
import numpy as np
from transformers import GPT2Model

def main():
    parser = argparse.ArgumentParser(description="Extract GPT-2 weights to binary format for the C++ engine.")
    default_bin_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "bin"))
    parser.add_argument("--out-dir", type=str, default=default_bin_dir, help="Directory to save weights and config")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    bin_file_path = os.path.join(args.out_dir, "model_weights.bin")
    json_file_path = os.path.join(args.out_dir, "model_offsets.json")

    print("Downloading/loading GPT-2 weights from HuggingFace...")
    model = GPT2Model.from_pretrained('gpt2')

    weights_dict = model.state_dict()
    byte_offset_index = {}
    current_byte_offset = 0

    print(f"Writing binary weights to: {bin_file_path}")
    with open(bin_file_path, "wb") as f:
        for layer_name, weights in weights_dict.items():
            print(f"  {layer_name} | shape: {list(weights.shape)}")

            # Detach from autograd, ensure CPU, and force float32
            np_weights = weights.detach().cpu().numpy().astype(np.float32)

            byte_offset_index[layer_name] = {
                "byte_offset": current_byte_offset,
                "shape": list(np_weights.shape)
            }

            raw_bytes = np_weights.tobytes()
            f.write(raw_bytes)
            current_byte_offset += len(raw_bytes)

    print(f"Writing metadata JSON to: {json_file_path}")
    with open(json_file_path, "w") as f:
        json.dump(byte_offset_index, f, indent=4)

    print(f"\nExport complete. Total file size: {current_byte_offset / (1024**2):.2f} MB")

if __name__ == "__main__":
    main()
