# Notebook Summary: Deep Learning Practice with Tokenizers

This notebook demonstrates the process of building and evaluating a Byte Pair Encoding (BPE) tokenizer using the `tokenizers` library, focusing on practical implementation with both the BookCorpus and IMDB datasets.

### Theory and Mathematical Concepts

#### 1. Dataset Preparation and Sampling
*   **BookCorpus Dataset:** A large dataset used for language modeling. For efficiency, only every 7th sample is selected to create a smaller, representative subset for tokenizer training (approximately 10.5 million samples from the original 74 million). This is a form of data subsampling to manage computational resources while retaining data diversity.
*   **IMDB Dataset:** A dataset commonly used for sentiment analysis, consisting of movie reviews. It is explicitly loaded with 'train' and 'test' splits, excluding the 'unsupervised' portion to focus on supervised learning tasks if applicable, or simply reduce data volume.

#### 2. Byte Pair Encoding (BPE) Tokenization
*   **Principle:** BPE is a subword tokenization algorithm. It works by iteratively merging the most frequent adjacent character pairs into new tokens until a desired vocabulary size is reached. This allows the tokenizer to handle out-of-vocabulary (OOV) words by breaking them down into known subword units.
*   **Normalizer (Lowercase):** Before tokenization, all text is converted to lowercase. This helps reduce vocabulary size and treats words like "The" and "the" as the same token, simplifying the model's learning process.
*   **Pre-tokenizer (Whitespace):** The text is initially split into words based on whitespace. BPE then operates on these words or character sequences. This is crucial for defining initial token boundaries.
*   **Special Tokens:** Specific tokens are added to the vocabulary to denote special semantic meanings or boundaries, such as:
    *   `[GO]` (Go/Beginning of Sentence)
    *   `[UNK]` (Unknown): For tokens not present in the vocabulary.
    *   `[PAD]` (Padding): To standardize sequence lengths.
    *   `[EOS]` (End of Sentence)
*   **Decoder (BPEDecoder):** Responsible for reconstructing the original text from token IDs. With a `Whitespace` pre-tokenizer and `BPEDecoder` (without explicit post-processing), the decoded text will appear concatenated without spaces. This is because the tokenizer handles spaces as part of the token boundaries during pre-tokenization but doesn't re-insert them during simple decoding unless specific post-processing rules are applied.
*   **Vocabulary Size:** The number of unique tokens the tokenizer can produce. Varying vocabulary sizes (e.g., 10K, 15K, 32K) directly impacts the granularity of tokenization:
    *   **Smaller vocab:** More subword tokens, shorter tokens, more common words are split (e.g., 'traders' -> 'trad', 'ers').
    *   **Larger vocab:** Fewer, longer tokens, more common words are represented as single tokens (e.g., 'traders' as a single token). This often leads to more efficient representation for frequent words.

### Code Implementation and Practice

#### 1. Environment Setup
*   Installation of `tokenizers` library using `!pip install tokenizers`.
*   Setting a Hugging Face token (`HF_TOKEN`) for dataset access.

#### 2. Dataset Loading
*   `load_dataset("rojagtap/bookcorpus")`: Loads the BookCorpus dataset.
*   `ds['train'].select(indices)`: Selects every 7th sample from the BookCorpus training split.
*   `load_dataset("stanfordnlp/imdb", split=['train', 'test'])`: Loads the IMDB dataset, specifically the train and test splits.

#### 3. Tokenizer Initialization and Configuration
*   `from tokenizers import Tokenizer`: Imports the main `Tokenizer` class.
*   `from tokenizers.models import BPE`: Imports the BPE model.
*   `from tokenizers.pre_tokenizers import Whitespace`: Imports the whitespace pre-tokenizer.
*   `from tokenizers.normalizers import Lowercase`: Imports the lowercase normalizer.
*   `from tokenizers import decoders`: Imports decoders.
*   `tokenizer = Tokenizer(BPE(unk_token=special_tokens["unk_token"]))`: Initializes the tokenizer with BPE model and an unknown token.
*   `tokenizer.normalizer = Lowercase()`: Sets the normalizer.
*   `tokenizer.pre_tokenizer = Whitespace()`: Sets the pre-tokenizer.
*   `tokenizer.decoder = decoders.BPEDecoder()`: Sets the decoder to BPEDecoder.

#### 4. Tokenizer Training
*   `batch_iterator` function: A Python generator that yields batches of text from the dataset. This is essential for training the tokenizer from a large dataset efficiently.
*   `from tokenizers.trainers import BpeTrainer`: Imports the BPE trainer class.
*   `trainer = BpeTrainer(...)`: Initializes the trainer with `vocab_size`, `min_frequency`, and `special_tokens`.
*   `tokenizer.train_from_iterator(batch_iterator(ds_copy), trainer=trainer, length=len(ds_copy))`: Trains the BPE tokenizer using the prepared BookCorpus subset.

#### 5. Tokenization and Decoding Sample Text
*   `input_text = "SEBI study finds..."`: Defines a sample text.
*   `output = tokenizer.encode(input_text)`: Tokenizes the input text, returning an `Encoding` object containing tokens and their IDs.
*   `tokenizer.decode(output.ids)`: Decodes the token IDs back to text.

#### 6. Retraining with Varying Vocabulary Sizes
*   A loop iterates through `vocab_sizes_to_test = [10000, 15000, 32000]`.
*   Inside the loop, the tokenizer and trainer are re-initialized and retrained with each specified vocabulary size.
*   The same sample text is tokenized and decoded for each vocabulary size, demonstrating the impact of `vocab_size` on token granularity.

#### 7. IMDB Dataset Tokenization
*   `tokenize_function(examples)`: A function defined to process text examples from the IMDB dataset.
    *   It calls `tokenizer.encode(examples["text"])` to get the `Encoding` object.
    *   **Crucially, it returns `{"tokens": encoding.tokens, "ids": encoding.ids}`**, converting the `Encoding` object into a dictionary. This is necessary because `dataset.map` expects a dictionary to add new columns to the dataset.
*   `tokenized_imdb_train = imdb_dataset[0].map(tokenize_function, batched=False)`: Applies the `tokenize_function` to the IMDB training split.
*   `tokenized_imdb_test = imdb_dataset[1].map(tokenize_function, batched=False)`: Applies the `tokenize_function` to the IMDB test split.
*   **Calculation of Total Tokens:** Sums the lengths of the `ids` lists across all entries in both `tokenized_imdb_train` and `tokenized_imdb_test` to get the total token count for the entire IMDB dataset. This provides a quantitative measure of the tokenization output.