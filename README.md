# Deep Learning Practice (DLP)

Welcome to the **Deep Learning Practice (DLP)** repository. This repository is dedicated to practical implementations, notebook studies, and core concepts in deep learning.

Currently, this repository hosts resources for **Week 2**, which focuses on **Text Tokenization**—covering custom subword tokenizer design, pre-trained Hugging Face tokenizers, and tokenizer vocabulary optimization.

---

## 📁 Repository Structure

```text
DLP/
├── .gitignore             # Configured to ignore local credentials/secrets
├── README.md              # Project documentation (this file)
└── Week2/                 # Week 2: Tokenization Concepts & Implementations
    ├── dlp_w2_1.ipynb     # Notebook 1: BPE Tokenizer training & dataset mapping
    ├── dlp_w2_2.ipynb     # Notebook 2: Custom vs. Pre-trained tokenizers & token count metrics
    ├── Summary_1.md       # Technical summary & key highlights for Notebook 1
    └── Summary_2.md       # Technical summary & key highlights for Notebook 2
```

---

## 📚 Topics Covered (Week 2 - Tokenization)

### 1. Byte Pair Encoding (BPE) Tokenizer Training
- **Dataset Preparation:** Subsampling BookCorpus (selecting every 7th sample to optimize memory while preserving diversity) and loading train/test splits for the Stanford IMDB dataset.
- **BPE Construction:** Defining a custom tokenizer with a lowercase normalizer, whitespace pre-tokenizer, and a BPE decoder.
- **Vocabulary Size Experiments:** Training and testing BPE tokenizers with varying vocabulary sizes (`10,000`, `15,000`, and `32,000`) to observe how vocabulary limits affect the granularity of subword splits.

### 2. Custom vs. Pre-trained Tokenizers
- **Custom Tokenizer Wrappers:** Loading custom tokenizer backends (e.g., from `.json` configuration files) and wrapping them with Hugging Face's `PreTrainedTokenizerFast` to ensure seamless compatibility with the `transformers` ecosystem.
- **Vocabulary Tuning:** Dynamically extending tokenizers with task-specific tokens (e.g., adding `'FY'` for financial documents) to improve tokenization efficiency and prevent fragmentation.
- **Pre-trained Comparison:** Contrast custom setups against industry standards like BERT (`bert-base-uncased`) and GPT-2, analyzing special tokens (`[UNK]`, `[PAD]`, `[SEP]`, `[CLS]`, `[MASK]`) and their structural roles.

### 3. Token Count Compression Metrics
- Computing the total token count across datasets to measure and compare the compression efficiency of different tokenizers.
- Analyzing how vocabulary expansion affects sequence lengths and subsequent model compute requirements.

---

## 🚀 Setup & Getting Started

### Prerequisites

To run the notebooks in this repository, make sure you have Python installed along with the following libraries:

```bash
pip install tokenizers transformers datasets
```

### Credentials & Hugging Face Hub Access

For downloading datasets or pushing models to the Hugging Face Hub:
1. Create a local credentials file named `creds` in the `Week2/` folder.
2. Put your Hugging Face API Token in this file. 
3. *Note: The `creds` file is already listed in `.gitignore` to prevent credentials from being pushed to public repositories.*

---

## 📝 Summaries
Detailed theoretical and mathematical explanations, along with key code blocks, are documented inside:
- [Week2/Summary_1.md](file:///home/abhayks0189/DLP/Week2/Summary_1.md)
- [Week2/Summary_2.md](file:///home/abhayks0189/DLP/Week2/Summary_2.md)
