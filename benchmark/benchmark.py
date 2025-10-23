import os

benchmarking_script = "benchmarking_script.py"

model_descriptions = {
    "small": {
        "vocab_size": 10000,
        "d_model": 768,
        "num_layers": 12,
        "num_heads": 12,
        "d_ff": 3072,
        "rope_theta": 10000.0,
        "context_length": 512,
        "batch_size": 4,
        "num_warmup_iters": 5,
        "num_iters": 10,
    },
    "medium": {
        "vocab_size": 10000,
        "d_model": 1024,
        "num_layers": 24,
        "num_heads": 16,
        "d_ff": 4096,
        "rope_theta": 10000.0,
        "context_length": 512,
        "batch_size": 4,
        "num_warmup_iters": 5,
        "num_iters": 10,
    },
    "large": {
        "vocab_size": 10000,
        "d_model": 1280,
        "num_layers": 36,
        "num_heads": 20,
        "d_ff": 5120,
        "rope_theta": 10000.0,
        "context_length": 512,
        "batch_size": 4,
        "num_warmup_iters": 5,
        "num_iters": 10,
    },
    "xl": {
        "vocab_size": 10000,
        "d_model": 1600,
        "num_layers": 48,
        "num_heads": 25,
        "d_ff": 6400,
        "rope_theta": 10000.0,
        "context_length": 512,
        "batch_size": 4,
        "num_warmup_iters": 5,
        "num_iters": 10,
    },
    "2.7B": {
        "vocab_size": 10000,
        "d_model": 2560,
        "num_layers": 32,
        "num_heads": 32,
        "d_ff": 10240,
        "rope_theta": 10000.0,
        "context_length": 512,
        "batch_size": 4,
        "num_warmup_iters": 5,
        "num_iters": 10,
    }
}

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    benchmarking_script_path = os.path.join(current_dir, benchmarking_script)
    # for description, model_description in model_descriptions.items():
    #     command = f"uv run python {benchmarking_script_path} --description no_mixed_precision_{description} --vocab-size {model_description['vocab_size']} --d-model {model_description['d_model']} --num-layers {model_description['num_layers']} --num-heads {model_description['num_heads']} --d-ff {model_description['d_ff']} --rope-theta {model_description['rope_theta']} --context-length {model_description['context_length']} --batch-size {model_description['batch_size']} --num-warmup-iters {model_description['num_warmup_iters']} --num-iters {model_description['num_iters']}"
    #     print(command)
    #     os.system(command)
    #     command = f"uv run python {benchmarking_script_path} --description mixed_precision_{description} --vocab-size {model_description['vocab_size']} --d-model {model_description['d_model']} --num-layers {model_description['num_layers']} --num-heads {model_description['num_heads']} --d-ff {model_description['d_ff']} --rope-theta {model_description['rope_theta']} --context-length {model_description['context_length']} --batch-size {model_description['batch_size']} --num-warmup-iters {model_description['num_warmup_iters']} --num-iters {model_description['num_iters']} --use-mixed-precision"
    #     print(command)
    #     os.system(command)
    for context_length in [128, 256, 512]:
        description: str = f"2.7B_context_length_{context_length}"
        command = f"uv run python {benchmarking_script_path} --description {description} --vocab-size {model_descriptions['2.7B']['vocab_size']} --d-model {model_descriptions['2.7B']['d_model']} --num-layers {model_descriptions['2.7B']['num_layers']} --num-heads {model_descriptions['2.7B']['num_heads']} --d-ff {model_descriptions['2.7B']['d_ff']} --rope-theta {model_descriptions['2.7B']['rope_theta']} --context-length {context_length} --batch-size {model_descriptions['2.7B']['batch_size']} --num-warmup-iters {model_descriptions['2.7B']['num_warmup_iters']} --num-iters {model_descriptions['2.7B']['num_iters']} --profile-memory"
        print(command)
        os.system(command)
        description: str = f"2.7B_context_length_{context_length}_use_mixed_precision"
        command = f"uv run python {benchmarking_script_path} --description {description} --vocab-size {model_descriptions['2.7B']['vocab_size']} --d-model {model_descriptions['2.7B']['d_model']} --num-layers {model_descriptions['2.7B']['num_layers']} --num-heads {model_descriptions['2.7B']['num_heads']} --d-ff {model_descriptions['2.7B']['d_ff']} --rope-theta {model_descriptions['2.7B']['rope_theta']} --context-length {context_length} --batch-size {model_descriptions['2.7B']['batch_size']} --num-warmup-iters {model_descriptions['2.7B']['num_warmup_iters']} --num-iters {model_descriptions['2.7B']['num_iters']} --use-mixed-precision --profile-memory"
        print(command)
        os.system(command)

if __name__ == "__main__":
    main()
