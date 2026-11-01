import argparse
import json
import os
import numpy as np

def build_gpt2_schema():
    """Generates the tensor layout and byte offsets for GPT-2 124M."""
    schema = {}
    current_byte_offset = 0

    def add_tensor(name, shape):
        nonlocal current_byte_offset
        num_elements = int(np.prod(shape))
        byte_size = num_elements * 4  # float32 = 4 bytes
        schema[name] = {
            "byte_offset": current_byte_offset,
            "shape": list(shape)
        }
        current_byte_offset += byte_size

    # Token and positional embeddings
    add_tensor("wte.weight", [50257, 768])
    add_tensor("wpe.weight", [1024, 768])

    # 12 Transformer Blocks
    for i in range(12):
        prefix = f"h.{i}."
        add_tensor(f"{prefix}ln_1.weight", [768])
        add_tensor(f"{prefix}ln_1.bias", [768])
        add_tensor(f"{prefix}attn.c_attn.weight", [768, 2304])
        add_tensor(f"{prefix}attn.c_attn.bias", [2304])
        add_tensor(f"{prefix}attn.c_proj.weight", [768, 768])
        add_tensor(f"{prefix}attn.c_proj.bias", [768])
        add_tensor(f"{prefix}ln_2.weight", [768])
        add_tensor(f"{prefix}ln_2.bias", [768])
        add_tensor(f"{prefix}mlp.c_fc.weight", [768, 3072])
        add_tensor(f"{prefix}mlp.c_fc.bias", [3072])
        add_tensor(f"{prefix}mlp.c_proj.weight", [3072, 768])
        add_tensor(f"{prefix}mlp.c_proj.bias", [768])

    # Final LayerNorm
    add_tensor("ln_f.weight", [768])
    add_tensor("ln_f.bias", [768])

    return schema, current_byte_offset

def generate_synthetic_weights(out_dir, seed=42):
    """
    Generates deterministic, non-zero synthetic weights for the GPT-2 124M architecture.
    Runs in < 1 second with ZERO network requests.
    """
    os.makedirs(out_dir, exist_ok=True)
    schema, total_bytes = build_gpt2_schema()
    total_elements = total_bytes // 4

    print(f"Generating deterministic synthetic weights ({total_bytes / (1024*1024):.2f} MB)...")
    rng = np.random.default_rng(seed)
    buffer = np.zeros(total_elements, dtype=np.float32)

    # Initialize weights with standard transformer scale
    for name, meta in schema.items():
        start = meta["byte_offset"] // 4
        count = int(np.prod(meta["shape"]))

        if "ln" in name and "weight" in name:
            # LayerNorm weights initialized to 1.0
            buffer[start : start + count] = 1.0
        elif "bias" in name:
            # Biases initialized to 0.0
            buffer[start : start + count] = 0.0
        else:
            # Linear & embedding projections initialized with small standard normal
            buffer[start : start + count] = rng.normal(0.0, 0.02, count).astype(np.float32)

    bin_path = os.path.join(out_dir, "model_weights.bin")
    json_path = os.path.join(out_dir, "model_offsets.json")

    print(f"Writing binary weights to: {bin_path}")
    buffer.tofile(bin_path)

    print(f"Writing schema JSON to: {json_path}")
    with open(json_path, "w") as f:
        json.dump(schema, f, indent=4)

    print("Synthetic model generation complete!")

def main():
    parser = argparse.ArgumentParser(description="Generate deterministic synthetic GPT-2 weights for fast CI testing.")
    parser.add_argument("--out-dir", type=str, required=True, help="Directory to output synthetic model weights and offsets")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for deterministic weights")
    args = parser.parse_args()

    generate_synthetic_weights(args.out_dir, seed=args.seed)

if __name__ == "__main__":
    main()
