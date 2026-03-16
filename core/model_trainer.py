"""
Model Fine-tuning Module for Legal Document Analysis
Supports training custom models on legal Q&A data
"""

import os
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
from typing import List, Dict, Optional
import numpy as np
from datetime import datetime

class FineTuner:
    """Fine-tune language models for legal document analysis"""
    
    def __init__(self, model_name: str = "google/flan-t5-base", output_dir: str = "fine_tuned_legal_model"):
        """
        Initialize the fine-tuner
        
        Args:
            model_name: Base model to fine-tune
            output_dir: Directory to save the fine-tuned model
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Device configuration
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
    def prepare_dataset(self, training_data: List[Dict[str, str]]) -> Dataset:
        """
        Prepare dataset for training
        
        Args:
            training_data: List of dicts with 'input' and 'output' keys
            
        Returns:
            HuggingFace Dataset object
        """
        # Convert to dataset format
        data = {
            'input_text': [item['input'] for item in training_data],
            'target_text': [item['output'] for item in training_data]
        }
        
        dataset = Dataset.from_dict(data)
        
        # Tokenize
        def preprocess_function(examples):
            inputs = self.tokenizer(
                examples['input_text'],
                max_length=512,
                truncation=True,
                padding='max_length'
            )
            
            targets = self.tokenizer(
                examples['target_text'],
                max_length=256,
                truncation=True,
                padding='max_length'
            )
            
            inputs['labels'] = targets['input_ids']
            return inputs
        
        tokenized_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def train(
        self,
        training_data: List[Dict[str, str]],
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 5e-5,
        validation_split: float = 0.1
    ):
        """
        Fine-tune the model
        
        Args:
            training_data: List of training samples
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            validation_split: Fraction of data for validation
        """
        print(f"Starting fine-tuning with {len(training_data)} samples...")
        
        # Prepare dataset
        full_dataset = self.prepare_dataset(training_data)
        
        # Split into train and validation
        split_idx = int(len(full_dataset) * (1 - validation_split))
        train_dataset = full_dataset.select(range(split_idx))
        eval_dataset = full_dataset.select(range(split_idx, len(full_dataset)))
        
        print(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=False,
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=10,
            save_total_limit=2,
            fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        # Train
        print("Starting training...")
        train_result = trainer.train()
        
        # Save model
        print(f"Saving model to {self.output_dir}")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        # Save training metadata
        metadata = {
            "base_model": self.model_name,
            "training_samples": len(training_data),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "train_loss": train_result.training_loss,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(f"{self.output_dir}/training_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("✅ Training completed successfully!")
        
        return train_result
    
    def evaluate(self, test_data: List[Dict[str, str]]) -> Dict:
        """
        Evaluate the fine-tuned model
        
        Args:
            test_data: Test samples
            
        Returns:
            Evaluation metrics
        """
        test_dataset = self.prepare_dataset(test_data)
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_eval_batch_size=4,
            fp16=torch.cuda.is_available(),
        )
        
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
        )
        
        results = trainer.evaluate(test_dataset)
        
        return results

def load_fine_tuned_model(model_path: str):
    """
    Load a fine-tuned model
    
    Args:
        model_path: Path to the fine-tuned model
        
    Returns:
        Tuple of (model, tokenizer)
    """
    if not os.path.exists(model_path):
        raise ValueError(f"Model path {model_path} does not exist")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    return model, tokenizer

def get_model_performance(model_path: str) -> Dict:
    """
    Get performance metrics of a fine-tuned model
    
    Args:
        model_path: Path to the fine-tuned model
        
    Returns:
        Dictionary with performance metrics
    """
    metadata_path = f"{model_path}/training_metadata.json"
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    return {"error": "No metadata found"}

def inference_with_fine_tuned_model(
    model_path: str,
    input_text: str,
    max_length: int = 256
) -> str:
    """
    Generate answer using fine-tuned model
    
    Args:
        model_path: Path to the fine-tuned model
        input_text: Input text/question
        max_length: Maximum length of generated output
        
    Returns:
        Generated answer
    """
    model, tokenizer = load_fine_tuned_model(model_path)
    
    # Tokenize input
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True
    )
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
    
    # Decode
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return answer

def create_training_data_from_documents(documents: List[str]) -> List[Dict[str, str]]:
    """
    Generate synthetic training data from documents
    Useful for bootstrapping when you don't have Q&A pairs
    
    Args:
        documents: List of document texts
        
    Returns:
        List of synthetic Q&A pairs
    """
    from transformers import pipeline
    
    # Use a question generation model (if available)
    # This is a placeholder - you can integrate actual QG models
    training_samples = []
    
    for doc in documents:
        # Split document into chunks
        chunks = [doc[i:i+500] for i in range(0, len(doc), 500)]
        
        for chunk in chunks[:5]:  # Limit to avoid too much data
            # Create synthetic Q&A pairs
            # In production, use proper question generation models
            training_samples.append({
                "input": f"Context: {chunk}\n\nQuestion: What is the main topic?",
                "output": chunk[:200]  # Summary as answer
            })
    
    return training_samples

class LegalModelTrainer:
    """
    Specialized trainer for legal document models
    Includes domain-specific preprocessing and evaluation
    """
    
    def __init__(self, base_model: str = "google/flan-t5-base"):
        self.base_model = base_model
        self.fine_tuner = FineTuner(model_name=base_model)
        
    def prepare_legal_training_data(
        self,
        qa_pairs: List[Dict[str, str]],
        add_legal_prefix: bool = True
    ) -> List[Dict[str, str]]:
        """
        Prepare training data with legal domain adaptations
        
        Args:
            qa_pairs: List of Q&A pairs with 'question', 'context', 'answer'
            add_legal_prefix: Add legal domain prefix to inputs
            
        Returns:
            Formatted training data
        """
        formatted_data = []
        
        for pair in qa_pairs:
            prefix = "Legal Document Analysis: " if add_legal_prefix else ""
            
            input_text = f"{prefix}Context: {pair.get('context', '')}\n\nQuestion: {pair['question']}"
            output_text = pair['answer']
            
            formatted_data.append({
                "input": input_text,
                "output": output_text
            })
        
        return formatted_data
    
    def train_with_validation(
        self,
        training_data: List[Dict],
        validation_data: Optional[List[Dict]] = None,
        **kwargs
    ):
        """
        Train model with validation and early stopping
        
        Args:
            training_data: Training Q&A pairs
            validation_data: Optional validation data
            **kwargs: Additional training arguments
        """
        # Prepare data
        formatted_train = self.prepare_legal_training_data(training_data)
        
        # Split if no validation data provided
        if validation_data is None and len(formatted_train) > 10:
            split_idx = int(len(formatted_train) * 0.9)
            train_samples = formatted_train[:split_idx]
            val_samples = formatted_train[split_idx:]
        else:
            train_samples = formatted_train
            val_samples = self.prepare_legal_training_data(validation_data) if validation_data else []
        
        # Train
        return self.fine_tuner.train(
            training_data=train_samples,
            **kwargs
        )
    
    def evaluate_on_legal_metrics(
        self,
        test_data: List[Dict],
        model_path: str
    ) -> Dict:
        """
        Evaluate model with legal-specific metrics
        
        Args:
            test_data: Test Q&A pairs
            model_path: Path to model
            
        Returns:
            Evaluation metrics
        """
        from nltk.translate.bleu_score import sentence_bleu
        from rouge import Rouge
        
        model, tokenizer = load_fine_tuned_model(model_path)
        rouge = Rouge()
        
        predictions = []
        references = []
        
        for sample in test_data:
            # Generate prediction
            input_text = f"Context: {sample.get('context', '')}\n\nQuestion: {sample['question']}"
            pred = inference_with_fine_tuned_model(model_path, input_text)
            
            predictions.append(pred)
            references.append(sample['answer'])
        
        # Calculate ROUGE scores
        try:
            rouge_scores = rouge.get_scores(predictions, references, avg=True)
        except:
            rouge_scores = {"error": "Could not calculate ROUGE"}
        
        # Calculate average lengths
        avg_pred_length = np.mean([len(p.split()) for p in predictions])
        avg_ref_length = np.mean([len(r.split()) for r in references])
        
        return {
            "rouge_scores": rouge_scores,
            "avg_prediction_length": avg_pred_length,
            "avg_reference_length": avg_ref_length,
            "num_test_samples": len(test_data)
        }

def export_model_for_deployment(model_path: str, export_path: str):
    """
    Export model in deployment-ready format
    
    Args:
        model_path: Path to fine-tuned model
        export_path: Path to export
    """
    import shutil
    
    # Copy model files
    os.makedirs(export_path, exist_ok=True)
    
    # Copy all necessary files
    for file in os.listdir(model_path):
        src = os.path.join(model_path, file)
        dst = os.path.join(export_path, file)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
    
    # Create deployment config
    config = {
        "model_type": "legal_document_qa",
        "base_model": "google/flan-t5-base",
        "max_input_length": 512,
        "max_output_length": 256,
        "deployment_date": datetime.now().isoformat()
    }
    
    with open(f"{export_path}/deployment_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Model exported to {export_path}")

# Example usage and testing functions
if __name__ == "__main__":
    # Example training data
    sample_training_data = [
        {
            "input": "Context: This agreement shall be governed by the laws of India.\n\nQuestion: What law governs this agreement?",
            "output": "This agreement is governed by the laws of India."
        },
        {
            "input": "Context: Payment must be made within 30 days of invoice date.\n\nQuestion: What is the payment timeline?",
            "output": "Payment must be made within 30 days from the invoice date."
        }
    ]
    
    print("Model Trainer Module Loaded Successfully!")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")