
import json
import os
from datasets import load_dataset

class TruthfulQADatasetLoader:
    def __init__(self, cache_dir: str = "./data/truthfulqa_cache"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def load_truthfulqa_dataset(self, split: str = "validation"):
        """Loads the TruthfulQA dataset from Hugging Face datasets."""
        # The 'generation' split is typically used for open-ended generation tasks
        # The 'multiple_choice' split is for multiple-choice questions
        dataset = load_dataset("truthful_qa", "generation", split=split, cache_dir=self.cache_dir)
        return dataset

    def preprocess_for_probe_training(self, dataset):
        """Preprocesses the TruthfulQA dataset for training truth/harm/benign/procedurality probes.
        This is a placeholder and needs actual logic based on how probes are defined and what labels are needed.
        """
        # For demonstration, let's assume we want to extract questions and answers
        # and assign dummy labels for truth/harm/benign/procedurality.
        # In a real scenario, these labels would come from the dataset itself or external annotations.

        prompts = []
        truth_labels = [] # 1 for truthful, 0 for not truthful
        harm_labels = []  # 1 for harmful, 0 for not harmful
        benign_labels = [] # 1 for benign context, 0 for not benign
        proc_labels = [] # 1 for procedural, 0 for descriptive

        for example in dataset:
            question = example["question"]
            # For simplicity, we'll use the best answer as the 'truthful' one for now.
            # In a real TruthfulQA setup, you'd need to compare model output to correct answers.
            correct_answer = example["best_answer"]

            # For probe training, we need input text (prompts) and corresponding labels.
            # This part is highly dependent on how you define 'truth', 'harm', etc.
            # For TruthfulQA, 'truth' is usually about factual correctness.
            # 'Harm' and 'benign' would require external annotation or heuristics.
            # 'Procedurality' would also require specific dataset characteristics.

            # Example: using the question as a prompt for feature extraction
            prompts.append(question)

            # Dummy labels for now - these need to be properly generated based on the task
            # For TruthfulQA, the 'correct_answers' field can be used to derive truthfulness.
            # Harmfulness, benignness, and procedurality would need more sophisticated logic
            # or external datasets.
            truth_labels.append(1) # Assume all questions can lead to truthful answers
            harm_labels.append(0)  # Assume questions are not inherently harmful
            benign_labels.append(1) # Assume questions are in a benign context
            proc_labels.append(0) # Assume questions are not procedural instructions

        return {
            "prompts": prompts,
            "truth_labels": truth_labels,
            "harm_labels": harm_labels,
            "benign_labels": benign_labels,
            "proc_labels": proc_labels
        }

    def get_truthfulqa_evaluation_data(self, split: str = "validation"):
        """Loads and formats TruthfulQA data specifically for evaluation."""
        dataset = self.load_truthfulqa_dataset(split=split)
        eval_prompts = []
        eval_gold_answers = []

        for example in dataset:
            eval_prompts.append(example["question"])
            # For evaluation, we need the best answer to compare against model generations
            eval_gold_answers.append(example["best_answer"])

        return eval_prompts, eval_gold_answers


# Example usage (for testing purposes)
if __name__ == "__main__":
    loader = TruthfulQADatasetLoader()
    dataset = loader.load_truthfulqa_dataset(split="validation")
    print(f"Loaded {len(dataset)} examples from TruthfulQA.")

    # Preprocess for probe training
    processed_data = loader.preprocess_for_probe_training(dataset)
    print(f"Processed {len(processed_data['prompts'])} prompts for probe training.")
    print(f"First prompt: {processed_data['prompts'][0]}")
    print(f"First truth label: {processed_data['truth_labels'][0]}")

    # Get evaluation data
    eval_prompts, eval_gold_answers = loader.get_truthfulqa_evaluation_data()
    print(f"Loaded {len(eval_prompts)} evaluation prompts.")
    print(f"First eval prompt: {eval_prompts[0]}")
    print(f"First gold answer: {eval_gold_answers[0]}")


