# ASL Alphabet LLM Evaluation

## Research Overview
This project evaluates current Large Language Models (LLMs) on their ability to identify American Sign Language (ASL) alphabets. The goal is to assess the current state of LLMs in the area of accessibility, specifically their vision capabilities for sign language recognition.

## Project Structure
```
ASL/
├── asl_llm_evaluation.ipynb   # Main evaluation notebook
├── requirements.txt            # Python dependencies
├── asl_alphabet_dataset/       # Dataset folder (create this)
│   ├── A/                     # Images for letter A
│   ├── B/                     # Images for letter B
│   └── ...                    # Continue for all letters
└── evaluation_results/         # Results output folder (auto-created)
```

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up Dataset
Create the dataset folder structure and add ASL alphabet images:
```bash
mkdir -p asl_alphabet_dataset/{A..Z}
```
Place ASL hand sign images in their respective letter folders.

### 3. Configure API Keys
Set your API keys as environment variables:
```bash
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export GOOGLE_API_KEY="your-google-api-key"
```

Or update them directly in the notebook's CONFIG section.

### 4. Run the Evaluation
Open the Jupyter notebook:
```bash
jupyter notebook asl_llm_evaluation.ipynb
```

Follow the cells in order to:
1. Load your ASL dataset
2. Initialize LLM evaluators
3. Run the evaluation pipeline
4. Analyze results with visualizations

## Features

### Evaluation Capabilities
- **Multi-Model Testing**: Evaluate GPT-4V, Claude 3, Gemini Pro Vision, and more
- **Prompt Engineering**: Compare different prompting strategies
- **Comprehensive Metrics**: Accuracy, confusion matrices, per-class performance
- **Response Time Analysis**: Measure inference speed
- **Error Analysis**: Identify common misclassifications

### Visualization Tools
- Model comparison charts
- Confusion matrices
- Per-class accuracy breakdown
- Response time distributions
- Prompt strategy comparisons

## Dataset Requirements

### Image Format
- Supported formats: JPG, PNG, JPEG
- Clear hand signs against contrasting background
- Consistent lighting recommended
- Various hand positions and orientations for robustness

### Folder Structure
Each letter (A-Z) should have its own folder containing multiple image examples.

## Results

Results are automatically saved to CSV files in the `evaluation_results/` folder with timestamps. Each evaluation run generates:
- Detailed predictions for each image
- Model performance metrics
- Response times
- Raw LLM responses

## Research Applications

This evaluation framework can help:
1. Assess LLM vision capabilities for accessibility
2. Identify areas for improvement in sign language recognition
3. Compare different LLM providers and models
4. Optimize prompting strategies for ASL classification
5. Generate insights for future model development

## Customization

### Adding New Models
Extend the evaluator classes in the notebook to add support for additional LLMs.

### Modifying Prompts
Edit the `create_prompt()` method in the `BaseLLMEvaluator` class to test new prompting strategies.

### Extending to Other Sign Languages
The framework can be adapted for other sign languages by modifying the class labels and dataset structure.

## Citation
If you use this evaluation framework in your research, please cite appropriately and acknowledge the accessibility focus of this work.

## License
This project is for research and educational purposes.
