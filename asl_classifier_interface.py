"""
ASL Classification Interface System
==================================

Abstract base class and implementations for testing multiple LLMs with various prompts
on ASL alphabet classification tasks.

Usage:
    # Initialize classifiers
    openai_classifier = OpenAIClassifier(api_key="your_key")
    claude_classifier = ClaudeClassifier(api_key="your_key")
    
    # Run evaluation
    evaluator = ASLEvaluator([openai_classifier, claude_classifier])
    results = evaluator.run_comprehensive_evaluation(df, prompts=['standard', 'detailed'])
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter
import logging
from PromptManager import PromptManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Single classification result"""
    image_id: str
    true_label: str
    predicted_label: str
    confidence: Optional[float] = None
    processing_time: Optional[float] = None
    raw_response: Optional[str] = None
    error: Optional[str] = None


@dataclass
class EvaluationResults:
    """Complete evaluation results for a classifier-prompt combination"""
    classifier_name: str
    prompt_style: str
    results: List[ClassificationResult]
    overall_accuracy: float
    per_class_accuracy: Dict[str, float]
    confusion_matrix: np.ndarray
    total_time: float
    metadata: Dict[str, Any]


class ASLClassifier(ABC):
    """Abstract base class for ASL classifiers"""
    
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.config = kwargs
        self.prompt_manager = PromptManager()
        
    @abstractmethod
    def classify_image(self, image_base64: str, prompt: str) -> Tuple[str, Optional[str], Optional[float]]:
        """
        Classify a single image
        
        Args:
            image_base64: Base64 encoded image
            prompt: Prompt text to use
            
        Returns:
            Tuple of (predicted_letter, raw_response, confidence_score)
        """
        pass
    
    @abstractmethod
    def setup_client(self):
        """Setup the API client for the specific LLM"""
        pass
    
    def extract_letter_from_response(self, response: str) -> str:
        """Extract single letter from LLM response"""
        import re
        
        # Clean the response
        response = response.strip().upper()
        
        # Try to find a single letter
        letters = re.findall(r'[A-Z]', response)
        
        if letters:
            # If response is just a single letter, return it
            if len(response) == 1 and response in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                return response
            # Otherwise, return the first letter found
            return letters[0]
        
        return "?"  # Unknown if no letter found
    
    def classify_dataset(self, 
                        df: pd.DataFrame, 
                        prompt_style: str = 'standard',
                        max_samples: Optional[int] = None) -> List[ClassificationResult]:
        """
        Classify entire dataset
        
        Args:
            df: DataFrame with 'image_base64' and 'label' columns
            prompt_style: Style of prompt to use
            max_samples: Limit number of samples (for testing)
            
        Returns:
            List of ClassificationResult objects
        """
        prompt = self.prompt_manager.get_prompt(prompt_style)
        results = []
        
        # Limit samples if specified
        data_to_process = df.head(max_samples) if max_samples else df
        
        logger.info(f"Starting classification with {self.name} using {prompt_style} prompt")
        logger.info(f"Processing {len(data_to_process)} samples")
        
        for idx, row in data_to_process.iterrows():
            start_time = time.time()
            
            try:
                predicted_letter, raw_response, confidence = self.classify_image(
                    row['image_base64'], prompt
                )
                
                processing_time = time.time() - start_time
                
                result = ClassificationResult(
                    image_id=str(idx),
                    true_label=row['label'],
                    predicted_label=predicted_letter,
                    confidence=confidence,
                    processing_time=processing_time,
                    raw_response=raw_response,
                    error=None
                )
                
            except Exception as e:
                logger.error(f"Error processing sample {idx}: {str(e)}")
                result = ClassificationResult(
                    image_id=str(idx),
                    true_label=row['label'],
                    predicted_label="?",
                    confidence=None,
                    processing_time=time.time() - start_time,
                    raw_response=None,
                    error=str(e)
                )
            
            results.append(result)
            
            # Progress logging
            if (len(results) % 50 == 0) or (len(results) == len(data_to_process)):
                logger.info(f"Processed {len(results)}/{len(data_to_process)} samples")
        
        return results


class OpenAIClassifier(ASLClassifier):
    """OpenAI GPT classifier implementation"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", **kwargs):
        super().__init__("OpenAI-" + model, **kwargs)
        self.api_key = api_key
        self.model = model
        self.client = None
        self.setup_client()
    
    def setup_client(self):
        """Setup OpenAI client"""
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("OpenAI library not installed. Run: pip install openai")
    
    def classify_image(self, image_base64: str, prompt: str) -> Tuple[str, Optional[str], Optional[float]]:
        """Classify image using OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}",
                                    "detail": "low"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=50,
                temperature=0
            )
            
            raw_response = response.choices[0].message.content
            predicted_letter = self.extract_letter_from_response(raw_response)
            
            return predicted_letter, raw_response, None
            
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return "?", None, None


class ClaudeClassifier(ASLClassifier):
    """Anthropic Claude classifier implementation"""
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229", **kwargs):
        super().__init__("Claude-" + model.split('-')[1], **kwargs)
        self.api_key = api_key
        self.model = model
        self.client = None
        self.setup_client()
    
    def setup_client(self):
        """Setup Claude client"""
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("Anthropic library not installed. Run: pip install anthropic")
    
    def classify_image(self, image_base64: str, prompt: str) -> Tuple[str, Optional[str], Optional[float]]:
        """Classify image using Claude API"""
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=50,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_base64,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ],
                    }
                ],
            )
            
            raw_response = message.content[0].text
            predicted_letter = self.extract_letter_from_response(raw_response)
            
            return predicted_letter, raw_response, None
            
        except Exception as e:
            logger.error(f"Claude API error: {str(e)}")
            return "?", None, None


class GeminiClassifier(ASLClassifier):
    """Google Gemini classifier implementation"""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash", **kwargs):
        super().__init__("Gemini-" + model.split('-')[1], **kwargs)
        self.api_key = api_key
        self.model = model
        self.client = None
        self.setup_client()
    
    def setup_client(self):
        """Setup Gemini client"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model)
        except ImportError:
            raise ImportError("Google AI library not installed. Run: pip install google-generativeai")
    
    def classify_image(self, image_base64: str, prompt: str) -> Tuple[str, Optional[str], Optional[float]]:
        """Classify image using Gemini API"""
        try:
            import base64
            from PIL import Image
            import io
            
            # Convert base64 to PIL Image
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
            
            response = self.client.generate_content([prompt, image])
            raw_response = response.text
            predicted_letter = self.extract_letter_from_response(raw_response)
            
            return predicted_letter, raw_response, None
            
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            return "?", None, None


class ASLEvaluator:
    """Main evaluation class that orchestrates multiple classifier and prompt combinations"""
    
    def __init__(self, classifiers: List[ASLClassifier]):
        self.classifiers = classifiers
        self.results_cache = {}
        
    def run_single_evaluation(self, 
                            classifier: ASLClassifier, 
                            df: pd.DataFrame, 
                            prompt_style: str,
                            max_samples: Optional[int] = None) -> EvaluationResults:
        """Run evaluation for single classifier-prompt combination"""
        
        start_time = time.time()
        
        # Get classification results
        classification_results = classifier.classify_dataset(df, prompt_style, max_samples)
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        true_labels = [r.true_label for r in classification_results]
        predicted_labels = [r.predicted_label for r in classification_results]
        
        # Filter out failed predictions for accuracy calculation
        valid_mask = [pred != "?" for pred in predicted_labels]
        if any(valid_mask):
            true_clean = [true_labels[i] for i in range(len(true_labels)) if valid_mask[i]]
            pred_clean = [predicted_labels[i] for i in range(len(predicted_labels)) if valid_mask[i]]
            overall_accuracy = accuracy_score(true_clean, pred_clean)
            
            # Per-class accuracy
            per_class_accuracy = {}
            for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                letter_mask = [label == letter for label in true_clean]
                if any(letter_mask):
                    letter_true = [true_clean[i] for i in range(len(true_clean)) if letter_mask[i]]
                    letter_pred = [pred_clean[i] for i in range(len(pred_clean)) if letter_mask[i]]
                    per_class_accuracy[letter] = accuracy_score(letter_true, letter_pred)
            
            # Confusion matrix
            all_labels = sorted(list(set(true_clean + pred_clean)))
            cm = confusion_matrix(true_clean, pred_clean, labels=all_labels)
        else:
            overall_accuracy = 0.0
            per_class_accuracy = {}
            cm = np.array([[]])
        
        # Metadata
        metadata = {
            'total_samples': len(classification_results),
            'valid_predictions': sum(valid_mask),
            'failed_predictions': len(classification_results) - sum(valid_mask),
            'average_processing_time': np.mean([r.processing_time for r in classification_results if r.processing_time]),
            'error_rate': (len(classification_results) - sum(valid_mask)) / len(classification_results)
        }
        
        return EvaluationResults(
            classifier_name=classifier.name,
            prompt_style=prompt_style,
            results=classification_results,
            overall_accuracy=overall_accuracy,
            per_class_accuracy=per_class_accuracy,
            confusion_matrix=cm,
            total_time=total_time,
            metadata=metadata
        )
    
    def run_comprehensive_evaluation(self, 
                                   df: pd.DataFrame,
                                   prompt_styles: Optional[List[str]] = None,
                                   max_samples: Optional[int] = None,
                                   save_results: bool = True) -> Dict[str, Dict[str, EvaluationResults]]:
        """
        Run comprehensive evaluation across all classifiers and prompt styles
        
        Returns:
            Dict[classifier_name][prompt_style] -> EvaluationResults
        """
        
        if prompt_styles is None:
            prompt_styles = ['standard', 'detailed', 'few_shot']
        
        results = {}
        
        for classifier in self.classifiers:
            results[classifier.name] = {}
            
            for prompt_style in prompt_styles:
                logger.info(f"Evaluating {classifier.name} with {prompt_style} prompt")
                
                eval_result = self.run_single_evaluation(
                    classifier, df, prompt_style, max_samples
                )
                
                results[classifier.name][prompt_style] = eval_result
                
                logger.info(f"Completed: {classifier.name} + {prompt_style} - "
                          f"Accuracy: {eval_result.overall_accuracy:.3f}")
        
        # Save results if requested
        if save_results:
            self.save_results(results)
        
        return results
    
    def save_results(self, results: Dict[str, Dict[str, EvaluationResults]]):
        """Save evaluation results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save summary CSV
        summary_data = []
        for classifier_name, classifier_results in results.items():
            for prompt_style, eval_result in classifier_results.items():
                summary_data.append({
                    'classifier': classifier_name,
                    'prompt_style': prompt_style,
                    'overall_accuracy': eval_result.overall_accuracy,
                    'valid_predictions': eval_result.metadata['valid_predictions'],
                    'failed_predictions': eval_result.metadata['failed_predictions'],
                    'total_time': eval_result.total_time,
                    'avg_processing_time': eval_result.metadata['average_processing_time'],
                    'error_rate': eval_result.metadata['error_rate']
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_filename = f'asl_evaluation_summary_{timestamp}.csv'
        summary_df.to_csv(summary_filename, index=False)
        logger.info(f"Summary saved to: {summary_filename}")
        
        # Save detailed results
        detailed_filename = f'asl_evaluation_detailed_{timestamp}.json'
        detailed_data = {}
        for classifier_name, classifier_results in results.items():
            detailed_data[classifier_name] = {}
            for prompt_style, eval_result in classifier_results.items():
                detailed_data[classifier_name][prompt_style] = {
                    'overall_accuracy': eval_result.overall_accuracy,
                    'per_class_accuracy': eval_result.per_class_accuracy,
                    'metadata': eval_result.metadata,
                    'total_time': eval_result.total_time
                }
        
        with open(detailed_filename, 'w') as f:
            json.dump(detailed_data, f, indent=2)
        logger.info(f"Detailed results saved to: {detailed_filename}")
    
    def create_comparison_plots(self, results: Dict[str, Dict[str, EvaluationResults]]):
        """Create comparison plots across classifiers and prompts"""
        
        # Prepare data for plotting
        plot_data = []
        for classifier_name, classifier_results in results.items():
            for prompt_style, eval_result in classifier_results.items():
                plot_data.append({
                    'classifier': classifier_name,
                    'prompt': prompt_style,
                    'accuracy': eval_result.overall_accuracy,
                    'processing_time': eval_result.total_time,
                    'error_rate': eval_result.metadata['error_rate']
                })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('ASL Classification Comparison Across LLMs and Prompts', fontsize=16)
        
        # 1. Accuracy comparison
        pivot_acc = plot_df.pivot(index='classifier', columns='prompt', values='accuracy')
        sns.heatmap(pivot_acc, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[0,0])
        axes[0,0].set_title('Accuracy by Classifier and Prompt')
        
        # 2. Processing time comparison
        pivot_time = plot_df.pivot(index='classifier', columns='prompt', values='processing_time')
        sns.heatmap(pivot_time, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[0,1])
        axes[0,1].set_title('Total Processing Time (seconds)')
        
        # 3. Bar plot of accuracy
        sns.barplot(data=plot_df, x='classifier', y='accuracy', hue='prompt', ax=axes[1,0])
        axes[1,0].set_title('Accuracy Comparison')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. Error rate comparison
        sns.barplot(data=plot_df, x='classifier', y='error_rate', hue='prompt', ax=axes[1,1])
        axes[1,1].set_title('Error Rate Comparison')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f'asl_comparison_plots_{timestamp}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        logger.info(f"Comparison plots saved to: {plot_filename}")
        
        plt.show()
        
        return plot_df
    
    def get_best_configurations(self, results: Dict[str, Dict[str, EvaluationResults]]) -> pd.DataFrame:
        """Find best performing classifier-prompt combinations"""
        
        performance_data = []
        for classifier_name, classifier_results in results.items():
            for prompt_style, eval_result in classifier_results.items():
                performance_data.append({
                    'classifier': classifier_name,
                    'prompt': prompt_style,
                    'accuracy': eval_result.overall_accuracy,
                    'processing_time': eval_result.total_time,
                    'error_rate': eval_result.metadata['error_rate'],
                    'valid_predictions': eval_result.metadata['valid_predictions']
                })
        
        performance_df = pd.DataFrame(performance_data)
        
        # Sort by accuracy (descending) and processing time (ascending)
        best_configs = performance_df.sort_values(['accuracy', 'processing_time'], ascending=[False, True])
        
        logger.info("\n🏆 TOP PERFORMING CONFIGURATIONS:")
        for idx, row in best_configs.head(5).iterrows():
            logger.info(f"   {idx+1}. {row['classifier']} + {row['prompt']}: "
                       f"Accuracy={row['accuracy']:.3f}, Time={row['processing_time']:.1f}s")
        
        return best_configs


# Example usage and testing utilities
def create_example_usage():
    """Create example usage code"""
    return """
# EXAMPLE USAGE:

import pandas as pd
from asl_classifier_interface import *

# 1. Setup your data
df = pd.read_csv('your_asl_dataset.csv')  # Should have 'image_base64' and 'label' columns

# 2. Initialize classifiers
classifiers = [
    OpenAIClassifier(api_key="your_openai_key", model="gpt-4o-mini"),
    ClaudeClassifier(api_key="your_claude_key", model="claude-3-sonnet-20240229"),
    GeminiClassifier(api_key="your_gemini_key", model="gemini-1.5-flash")
]

# 3. Create evaluator
evaluator = ASLEvaluator(classifiers)

# 4. Run comprehensive evaluation
results = evaluator.run_comprehensive_evaluation(
    df=df,
    prompt_styles=['standard', 'detailed', 'few_shot', 'chain_of_thought'],
    max_samples=50,  # For testing, remove for full dataset
    save_results=True
)

# 5. Create comparison plots
plot_data = evaluator.create_comparison_plots(results)

# 6. Find best configurations
best_configs = evaluator.get_best_configurations(results)

# 7. Get detailed results for best configuration
best_config = best_configs.iloc[0]
best_result = results[best_config['classifier']][best_config['prompt']]
print(f"Best accuracy: {best_result.overall_accuracy:.3f}")
print(f"Per-class accuracy: {best_result.per_class_accuracy}")

# 8. Add custom prompt and test
PromptManager.add_custom_prompt('custom', 'Your custom prompt here...')
custom_result = evaluator.run_single_evaluation(
    classifiers[0], df, 'custom', max_samples=10
)
"""


if __name__ == "__main__":
    print("ASL Classifier Interface System")
    print("=" * 40)
    print("\nAvailable prompt styles:")
    for prompt in PromptManager.list_prompts():
        print(f"  - {prompt}")
    
    print("\nExample usage:")
    print(create_example_usage())
