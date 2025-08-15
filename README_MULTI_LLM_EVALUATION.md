# ASL Multi-LLM Evaluation System

A comprehensive interface system for evaluating multiple Large Language Models (LLMs) on ASL alphabet classification with various prompt strategies.

## 🚀 Quick Start

```python
from asl_classifier_interface import OpenAIClassifier, ClaudeClassifier, ASLEvaluator
import pandas as pd

# 1. Load your data
df = pd.read_csv('predictions.csv')  # Must have 'image_base64' and 'label' columns

# 2. Setup classifiers
classifiers = [
    OpenAIClassifier(api_key="your_key", model="gpt-4o-mini"),
    ClaudeClassifier(api_key="your_key", model="claude-3-sonnet-20240229")
]

# 3. Run evaluation
evaluator = ASLEvaluator(classifiers)
results = evaluator.run_comprehensive_evaluation(df)

# 4. Get best configuration
best_configs = evaluator.get_best_configurations(results)
print(f"Best: {best_configs.iloc[0]['classifier']} with {best_configs.iloc[0]['prompt']} prompt")
```

## 🏗️ Architecture

### Abstract Base Class: `ASLClassifier`

The system uses an abstract base class that defines the interface all LLM implementations must follow:

```python
class ASLClassifier(ABC):
    @abstractmethod
    def classify_image(self, image_base64: str, prompt: str) -> Tuple[str, Optional[str], Optional[float]]:
        """Returns: (predicted_letter, raw_response, confidence)"""
        pass
    
    @abstractmethod
    def setup_client(self):
        """Initialize the LLM API client"""
        pass
```

### Supported LLMs

#### 1. OpenAI GPT Models
```python
openai_classifier = OpenAIClassifier(
    api_key="your_openai_key",
    model="gpt-4o-mini"  # or "gpt-4o", "gpt-4-turbo", etc.
)
```

#### 2. Anthropic Claude Models
```python
claude_classifier = ClaudeClassifier(
    api_key="your_claude_key", 
    model="claude-3-sonnet-20240229"  # or "claude-3-opus-20240229", etc.
)
```

#### 3. Google Gemini Models
```python
gemini_classifier = GeminiClassifier(
    api_key="your_gemini_key",
    model="gemini-1.5-flash"  # or "gemini-1.5-pro", etc.
)
```

## 📝 Prompt Strategies

The system includes 7 built-in prompt strategies:

| Strategy | Description | Best For |
|----------|-------------|----------|
| `standard` | Simple, direct instruction | Baseline comparison |
| `detailed` | Detailed analysis request | Better accuracy |
| `few_shot` | Includes examples | Improved context |
| `chain_of_thought` | Step-by-step reasoning | Complex decisions |
| `expert` | Expert-level instruction | Technical accuracy |
| `simple` | Minimal instruction | Speed/cost optimization |
| `context_aware` | ASL-specific context | Domain expertise |

### Adding Custom Prompts

```python
from asl_classifier_interface import PromptManager

PromptManager.add_custom_prompt('my_prompt', """
Your custom prompt here...
Make sure to end with: Respond with only the letter.
""")

# Use in evaluation
results = evaluator.run_comprehensive_evaluation(
    df, prompt_styles=['my_prompt']
)
```

## 🧪 Running Evaluations

### Single Evaluation
```python
# Test one classifier with one prompt
result = evaluator.run_single_evaluation(
    classifier=openai_classifier,
    df=df,
    prompt_style='standard',
    max_samples=50  # Optional: limit for testing
)
```

### Comprehensive Evaluation
```python
# Test multiple classifiers with multiple prompts
results = evaluator.run_comprehensive_evaluation(
    df=df,
    prompt_styles=['standard', 'detailed', 'few_shot'],
    max_samples=None,  # Use full dataset
    save_results=True  # Save to CSV and JSON
)
```

### Evaluation Results Structure
```python
results = {
    'OpenAI-gpt-4o-mini': {
        'standard': EvaluationResults(...),
        'detailed': EvaluationResults(...),
        # ...
    },
    'Claude-sonnet': {
        'standard': EvaluationResults(...),
        # ...
    }
}
```

## 📊 Analysis and Visualization

### Generate Comparison Plots
```python
plot_data = evaluator.create_comparison_plots(results)
# Creates 4-panel comparison: accuracy heatmap, processing time, bar charts, error rates
```

### Find Best Configurations
```python
best_configs = evaluator.get_best_configurations(results)
print(best_configs.head())  # Top performing combinations
```

### Access Detailed Results
```python
# Get specific result
openai_standard = results['OpenAI-gpt-4o-mini']['standard']

print(f"Accuracy: {openai_standard.overall_accuracy}")
print(f"Per-class accuracy: {openai_standard.per_class_accuracy}")
print(f"Processing time: {openai_standard.total_time}s")
print(f"Metadata: {openai_standard.metadata}")
```

## 🔧 Extending the System

### Adding a New LLM

```python
class MyCustomLLMClassifier(ASLClassifier):
    def __init__(self, api_key, model="default-model", **kwargs):
        super().__init__(f"MyLLM-{model}", **kwargs)
        self.api_key = api_key
        self.model = model
        self.client = None
        self.setup_client()
    
    def setup_client(self):
        # Initialize your LLM's client
        import my_llm_library
        self.client = my_llm_library.Client(api_key=self.api_key)
    
    def classify_image(self, image_base64: str, prompt: str) -> Tuple[str, Optional[str], Optional[float]]:
        try:
            # Make API call to your LLM
            response = self.client.chat_with_image(
                prompt=prompt,
                image=image_base64
            )
            
            # Extract letter from response
            predicted_letter = self.extract_letter_from_response(response.text)
            
            return predicted_letter, response.text, response.confidence
            
        except Exception as e:
            return "?", None, None

# Use your custom LLM
custom_classifier = MyCustomLLMClassifier(api_key="your_key")
evaluator = ASLEvaluator([custom_classifier])
```

## 💾 Output Files

The system automatically saves results in multiple formats:

### Summary CSV
`asl_evaluation_summary_TIMESTAMP.csv` - High-level metrics for all combinations

### Detailed JSON  
`asl_evaluation_detailed_TIMESTAMP.json` - Complete results with per-class accuracy

### Comparison Plots
`asl_comparison_plots_TIMESTAMP.png` - Visual comparison charts

### Comparison Table
`asl_comparison_table_TIMESTAMP.csv` - Detailed comparison spreadsheet

## 📈 Performance Metrics

Each evaluation provides:

- **Overall Accuracy**: Percentage of correct predictions
- **Per-class Accuracy**: Accuracy for each letter A-Z
- **Processing Time**: Total and average per image
- **Success Rate**: Percentage of valid responses (non-errors)
- **Error Rate**: Percentage of failed API calls
- **Confusion Matrix**: Detailed misclassification patterns

## 🛠️ Installation Requirements

```bash
# Core requirements
pip install pandas numpy matplotlib seaborn scikit-learn

# LLM-specific requirements (install as needed)
pip install openai              # For OpenAI classifiers
pip install anthropic          # For Claude classifiers  
pip install google-generativeai # For Gemini classifiers
```

## 🔐 Security Best Practices

1. **API Keys**: Use environment variables or secure key management
2. **Rate Limiting**: Implement delays between requests if needed
3. **Error Handling**: The system handles API failures gracefully
4. **Data Privacy**: Be mindful of sending sensitive data to external APIs

```python
import os

# Secure way to handle API keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY')
```

## 🧪 Example Workflows

### Research Comparison
```python
# Compare all available LLMs with academic rigor
results = evaluator.run_comprehensive_evaluation(
    df=full_dataset,
    prompt_styles=['standard', 'detailed', 'few_shot', 'chain_of_thought'],
    save_results=True
)

# Generate academic-style comparison
best_configs = evaluator.get_best_configurations(results)
plot_data = evaluator.create_comparison_plots(results)
```

### Production Optimization
```python
# Find best accuracy/cost balance
results = evaluator.run_comprehensive_evaluation(
    df=sample_dataset,
    prompt_styles=['standard', 'simple'],  # Focus on efficient prompts
    max_samples=100  # Quick validation
)

# Identify cost-effective solution
best = evaluator.get_best_configurations(results)
production_classifier = results[best.iloc[0]['classifier']][best.iloc[0]['prompt']]
```

### A/B Testing New Prompts
```python
# Test new prompt against baseline
PromptManager.add_custom_prompt('experimental', "Your new prompt...")

results = evaluator.run_comprehensive_evaluation(
    df=test_dataset,
    prompt_styles=['standard', 'experimental'],
    max_samples=200
)

# Compare performance
comparison = evaluator.create_comparison_plots(results)
```

## 📞 Support

For issues or questions:
1. Check the demo notebook: `asl_multi_llm_demo.ipynb`
2. Review the interface code: `asl_classifier_interface.py`
3. Examine output files for detailed debugging information

The system is designed to be extensible and maintainable. Feel free to add new LLMs, prompts, or evaluation metrics as needed!
