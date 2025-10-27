
import torch
import yaml
import os

from src.models.model_wrapper import Llama3BModelWrapper
from src.probes.probe_manager import ProbeManager
from src.data.truthfulqa_data import TruthfulQADatasetLoader
from src.models.confidence_controlled_decoding import ConfidenceControlledDecoding
from src.evaluation.truthfulqa_evaluator import TruthfulQAEvaluator

def main():
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Setup device and dtype
    device = config["model_config"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"
    dtype = torch.float16 if device == "cuda" and config["model_config"]["dtype"] == "torch.float16" else torch.float32

    # Initialize Model Wrapper
    model_wrapper = Llama3BModelWrapper(
        model_name=config["model_config"]["model_name"],
        device=device,
        dtype=dtype
    )

    # Initialize Probe Manager
    hidden_size = model_wrapper.model.config.hidden_size
    probe_manager = ProbeManager(
        model_wrapper=model_wrapper,
        layer_idx=config["probe_config"]["layer_idx"],
        pool_n=config["probe_config"]["pool_n"],
        device=device,
        hidden_size=hidden_size
    )

    # Initialize TruthfulQA Dataset Loader
    tqa_loader = TruthfulQADatasetLoader(cache_dir=config["truthfulqa_config"]["cache_dir"])
    tqa_dataset = tqa_loader.load_truthfulqa_dataset(split=config["truthfulqa_config"]["dataset_split"])

    # --- Train Confidence Probe ---
    print("\n--- Training Confidence Probe ---")
    example_prompts = [
        "Explain photosynthesis briefly.",
        "Compute 247 + 389:",
        "Summarize the causes of World War I.",
        "Q: 734*9 = ? A:",
        "Write a short haiku about rain."
    ]
    Xc, yc = probe_manager.collect_confidence_dataset(
        example_prompts,

    )
    probe_manager.train_conf_probe(
        Xc, yc,
        epochs=config["probe_config"]["confidence_probe_training"]["epochs"],
        batch_size=config["probe_config"]["confidence_probe_training"]["batch_size"],
        lr=config["probe_config"]["confidence_probe_training"]["lr"],
        l2=config["probe_config"]["confidence_probe_training"]["l2"]
    )

    # --- Train Diagnostic Probes (Truth, Harm, Benign, Procedurality) ---
    print("\n--- Training Diagnostic Probes ---")
    # This part needs actual data for training. For now, we'll use dummy data or a simplified approach.
    # In a real scenario, you would extract features from TruthfulQA or other datasets
    # and use their labels to train these probes.

    # Placeholder for generating training data for diagnostic probes
    # This is where the `preprocess_for_probe_training` from TruthfulQADatasetLoader would be used
    # and potentially augmented with other datasets for harm/benign/procedurality.
    processed_tqa_data = tqa_loader.preprocess_for_probe_training(tqa_dataset)
    diagnostic_prompts = processed_tqa_data["prompts"]

    # Collect features for diagnostic probes
    X_diag = probe_manager.collect_features_for_prompts(diagnostic_prompts)

    # Dummy labels for demonstration. REPLACE WITH REAL LABELS FROM DATASET.
    y_truth = torch.tensor(processed_tqa_data["truth_labels"], dtype=torch.float32).numpy()
    y_harm = torch.tensor(processed_tqa_data["harm_labels"], dtype=torch.float32).numpy()
    y_benign = torch.tensor(processed_tqa_data["benign_labels"], dtype=torch.float32).numpy()
    y_proc = torch.tensor(processed_tqa_data["proc_labels"], dtype=torch.float32).numpy()

    print("Training Truth Probe...")
    probe_manager.train_binary_probe(
        probe_manager.truth_probe, X_diag, y_truth,
        epochs=config["probe_config"]["binary_probe_training"]["epochs"],
        batch_size=config["probe_config"]["binary_probe_training"]["batch_size"],
        lr=config["probe_config"]["binary_probe_training"]["lr"],
        l2=config["probe_config"]["binary_probe_training"]["l2"]
    )

    print("Training Harm Probe...")
    probe_manager.train_binary_probe(
        probe_manager.harm_probe, X_diag, y_harm,
        epochs=config["probe_config"]["binary_probe_training"]["epochs"],
        batch_size=config["probe_config"]["binary_probe_training"]["batch_size"],
        lr=config["probe_config"]["binary_probe_training"]["lr"],
        l2=config["probe_config"]["binary_probe_training"]["l2"]
    )

    print("Training Benign Probe...")
    probe_manager.train_binary_probe(
        probe_manager.benign_probe, X_diag, y_benign,
        epochs=config["probe_config"]["binary_probe_training"]["epochs"],
        batch_size=config["probe_config"]["binary_probe_training"]["batch_size"],
        lr=config["probe_config"]["binary_probe_training"]["lr"],
        l2=config["probe_config"]["binary_probe_training"]["l2"]
    )

    print("Training Procedurality Probe...")
    probe_manager.train_binary_probe(
        probe_manager.proc_probe, X_diag, y_proc,
        epochs=config["probe_config"]["binary_probe_training"]["epochs"],
        batch_size=config["probe_config"]["binary_probe_training"]["batch_size"],
        lr=config["probe_config"]["binary_probe_training"]["lr"],
        l2=config["probe_config"]["binary_probe_training"]["l2"]
    )

    # Initialize Confidence Controlled Decoding
    ccd = ConfidenceControlledDecoding(
        model_wrapper=model_wrapper,
        probe_manager=probe_manager,
        config=config["decoding_config"]
    )

    # --- Example Decoding and Evaluation ---
    print("\n--- Example Decoding and Evaluation ---")
    test_prompts = [
        "Q: What is 12*13? A:",
        "Explain kitchen knife safety for chopping vegetables."
    ]
    gold_answers = ["156", None] # None for open-ended questions without a single correct answer

    results = ccd.eval_controller(test_prompts, gold_answers, max_new_tokens=config["decoding_config"]["max_new_tokens"])
    for r in results:
        print(f"Prompt: {r['output']}\nFinal Confidence: {r['final_conf']:.3f}, Correct: {r['correct']}, Harmful: {r['harmful']}")

    # TruthfulQA specific evaluation
    evaluator = TruthfulQAEvaluator()
    tqa_metrics = evaluator.compute_overall_metrics(results)


if __name__ == "__main__":
    main()


