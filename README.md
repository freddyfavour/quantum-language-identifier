# Quantum-Enhanced Bigram Language Identifier

**A Hybrid Quantum-Classical Approach for Multilingual Text Classification in the AQC Hackathon**

## Project Overview

This repository presents a sophisticated hybrid quantum-classical framework for language identification, specifically tailored for English (`eng`), French (`fra`), and Twi (`twi`). The system leverages classical bigram-based feature extraction with quantum-optimized weighting to achieve high accuracy and efficiency.

The core innovation lies in using a variational quantum circuit to learn optimal feature weights, which are then applied in a classical inference pipeline. This design addresses the hackathon challenge by employing quantum computing to enhance accuracy through non-linear feature combinations, while ensuring scalability for production use.

Key contributions:
- **Feature Engineering**: Combined word and character bigram scoring with additive smoothing for robust handling of unseen patterns.
- **Quantum Optimization**: A parameterized quantum circuit optimizes feature weights via cross-entropy minimization.
- **Hybrid Inference**: Quantum-trained weights enable fast, classical prediction on large corpora.
- **Evaluation Methodology**: Dual testing on held-out dataset and 100,000 synthetically generated sentences for realistic performance assessment.

This work demonstrates the potential of quantum computing in natural language processing tasks, particularly for low-resource languages like Twi.

## Features

- **Multi-Level Bigram Modeling**: Word and character-level features for comprehensive linguistic analysis.
- **Quantum Weight Learning**: 4-qubit variational circuit with rotational and entangling gates to derive optimal feature weights.
- **Efficient Prediction**: Batch processing supports 100k+ sentences with low latency.
- **Robust Smoothing**: Additive smoothing (α=0.1) to handle sparse or unseen bigrams.
- **Realistic Evaluation**: Synthetic sentence generation using Faker for English/French and custom word bank for Twi.
- **Modular Design**: Clean classes for feature extraction, quantum optimization, and hybrid prediction.

## Installation

### Prerequisites
- Python 3.8+
- Google Colab (recommended for quantum simulation)

### Steps
1. Clone the repository:
   ```
   git clone https://github.com/freddyfavour/quantum-language-identifier.git
   cd quantum-language-identifier
   ```

2. Install dependencies:
   ```
   pip install qiskit qiskit-aer scipy pandas numpy faker
   ```

3. Place `bigrams.csv` in your working directory or Google Drive.

## Usage

### In Google Colab
1. Mount Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. Load and run the notebook cells in sequence.

### Prediction Example
```python
extractor = FeatureExtractor(alpha=0.1)
extractor.build_models(df)  # df is the loaded bigrams.csv

# Train quantum optimizer (on your training data)
quantum = QuantumWeightOptimizer(num_qubits=4, layers=4)
quantum.train(X_train, y_train)  # X_train, y_train from prepared data

model = HybridLanguageIdentifier(extractor, quantum.weights)

# Single prediction
print(model.predict("This is an English sentence."))

# Batch prediction
batch = ["Bonjour le monde", "Akwaaba to the world", "Hello world"] * 1000
preds = model.predict_batch(batch)
```

### Training
The system trains the quantum circuit on a sample of the dataset (or generated sentences) to learn feature weights.

## Methodology

### 1. Feature Extraction
- Word bigrams capture sequential patterns.
- Character bigrams handle orthographic nuances (e.g., Twi accents).
- Scores computed as averaged log-probabilities with smoothing.

### 2. Quantum Weight Optimization
- Variational circuit: RX, RY, RZ rotations + CX entanglements.
- Measurement probabilities mapped to 6 feature weights.
- Optimized using COBYLA on cross-entropy loss.

### 3. Hybrid Prediction
- Weighted sum of features using quantum-derived weights.
- Aggregation: Word + character scores per language → argmax.

### 4. Evaluation
- **Dataset Test**: Held-out bigrams from CSV.
- **Generated Test**: 100k synthetic full sentences using Faker (English/French) and Twi word bank.
- Metrics: Accuracy, classification report, confusion matrix.

## Results

- **CSV Test Accuracy**: ~0.95–0.99 (reference metric).
- **Generated Sentences Accuracy**: ~0.85–0.92 (realistic generalization).
- **Training Time**: ~3 minutes for 1M samples.
- **Inference Speed**: Thousands of sentences per second.

Detailed reports and confusion matrices are printed during execution.

## Limitations and Future Work

- Twi sentence generation relies on limited word bank — could integrate more diverse corpora.
- Quantum simulation on classical hardware; potential for real QPU deployment.
- Explore deeper circuits or alternative encodings for further accuracy gains.

## Team

- Salam Musa
- Favour Alfred
- Kashi

## License

MIT License. See [LICENSE](LICENSE) for details.

---

**Submitted to the AQC Hackathon**  
*Date: 5th January, 2025*