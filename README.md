# Bidirectional LSTM Sequence-to-Sequence Model for Text Summarization

## Overview

This project implements a sophisticated text summarization model using a bidirectional LSTM encoder-decoder architecture with attention mechanism. The model is designed to generate concise summaries of longer text documents, specifically trained on Amazon Fine Food Reviews dataset.

## Model Architecture

### Key Components

1. **Bidirectional LSTM Encoder**: Processes input text in both forward and backward directions to capture contextual information
2. **LSTM Decoder with Attention**: Generates summaries word by word while attending to relevant parts of the input
3. **Bahdanau Attention Mechanism**: Allows the decoder to focus on different parts of the input sequence during generation
4. **Pre-trained Word Embeddings**: Uses GloVe 100-dimensional embeddings for better word representations

### Architecture Details

- **Encoder**: Bidirectional LSTM with 128 units in each direction (256 total)
- **Decoder**: LSTM with 256 units, dropout (0.3) and recurrent dropout (0.2)
- **Attention**: Custom Bahdanau attention layer
- **Embedding Dimension**: 100 (using pre-trained GloVe embeddings)
- **Maximum Text Length**: 800 tokens
- **Maximum Summary Length**: 150 tokens

## Dataset

- **Source**: Amazon Fine Food Reviews dataset
- **Preprocessing**: 
  - Removed duplicates and null values
  - Applied text cleaning (contractions, URLs, special characters)
  - Removed stopwords for input text
  - Added special tokens (`<sostok>` and `<eostok>`) to summaries
- **Sample Size**: 1000 reviews (for demonstration purposes)
- **Train/Test Split**: 90/10

## Features

### Text Preprocessing
- Contraction expansion (e.g., "don't" → "do not")
- URL and HTML tag removal
- Special character cleaning
- Stopword removal (for input text only)
- Tokenization and sequence padding

### Model Features
- **Attention Mechanism**: Helps model focus on relevant parts of input text
- **Bidirectional Processing**: Captures both forward and backward context
- **Teacher Forcing**: Uses ground truth tokens during training
- **Early Stopping**: Prevents overfitting with patience=2
- **Model Checkpointing**: Saves best model weights

## File Structure

```
├── bi-lstm-seq2seq.ipynb    # Main notebook with model implementation
├── README.md                # This documentation
├── summary.json             # Saved model architecture
├── summary.weights.h5       # Saved model weights
├── encoder_model.h5         # Encoder model for inference
├── decoder_model.h5         # Decoder model for inference
└── s_tokenizer.pkl          # Summary tokenizer for inference
```

## Installation and Setup

### Dependencies

```python
# Core libraries
numpy
pandas
scikit-learn

# Deep learning
tensorflow
keras

# NLP
nltk

# Visualization
matplotlib

# Utilities
pickle
warnings
re
```

### Data Requirements

1. Amazon Fine Food Reviews dataset (`Reviews.csv`)
2. GloVe pre-trained embeddings (`glove.6B.100d.txt`)

## Usage

### Training the Model

1. **Data Loading and Preprocessing**:
   ```python
   # Load dataset
   data = pd.read_csv("path/to/Reviews.csv")
   
   # Clean and preprocess text
   clean_texts = [clean_text(text) for text in data['Text']]
   clean_summaries = [clean_text(summary, remove_stopwords=False) for summary in data['Summary']]
   ```

2. **Model Training**:
   ```python
   # Compile model
   model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop')
   
   # Train with early stopping
   model.fit([train_x, train_y[:, :-1]], train_y.reshape(...), 
             epochs=5, callbacks=[early_stop], batch_size=128)
   ```

### Inference

```python
# Generate summary for new text
summary = generate_summary(input_sequence)
print(f"Generated Summary: {summary}")
```

## Model Performance

### Training Configuration
- **Optimizer**: RMSprop
- **Loss Function**: Sparse Categorical Crossentropy
- **Batch Size**: 128
- **Epochs**: 5 (with early stopping)
- **Validation Split**: 10%

### Vocabulary Statistics
- **Text Vocabulary**: Dynamically determined based on word frequency (threshold=4)
- **Summary Vocabulary**: Separate vocabulary for target sequences
- **Rare Word Filtering**: Words appearing less than 4 times are excluded

## Key Functions

### Text Preprocessing
```python
clean_text(text, remove_stopwords=True)
```
- Expands contractions
- Removes URLs, HTML tags, special characters
- Optionally removes stopwords

### Summary Generation
```python
generate_summary(input_seq)
```
- Uses beam search-like approach
- Generates summaries token by token
- Stops at end-of-sequence token or max length

### Attention Layer
```python
class AttentionLayer(Layer)
```
- Implements Bahdanau attention mechanism
- Computes attention weights for each decoder step
- Returns context vectors and attention weights

## Model Saving and Loading

The notebook saves multiple components for easy inference:

1. **Full Model**: Architecture (`summary.json`) + Weights (`summary.weights.h5`)
2. **Encoder Model**: For encoding input sequences
3. **Decoder Model**: For step-by-step decoding
4. **Tokenizers**: For text-to-sequence conversion

## Potential Improvements

1. **Larger Dataset**: Train on complete Amazon reviews dataset
2. **Beam Search**: Implement beam search for better summary quality
3. **ROUGE Evaluation**: Add quantitative evaluation metrics
4. **Transformer Architecture**: Consider using attention-only models
5. **Pre-trained Models**: Fine-tune BERT/GPT for summarization
6. **Coverage Mechanism**: Prevent repetition in generated summaries

## Technical Notes

### Memory Considerations
- Model uses significant GPU memory due to attention mechanism
- Consider reducing batch size if encountering memory issues
- Sequence lengths are capped to manage computational complexity

### Attention Visualization
The model saves attention weights, enabling visualization of which parts of the input the model focuses on during summary generation.

## References

- Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate.
- Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to sequence learning with neural networks.
- See, A., Liu, P. J., & Manning, C. D. (2017). Get to the point: Summarization with pointer-generator networks.

## License

This project is for educational and research purposes. Please ensure compliance with dataset licenses when using for commercial applications.