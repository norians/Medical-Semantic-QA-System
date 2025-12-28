# Overview
This project implements a semantic retrieval-based medical question-answering system. It maps user queries to clinical answers by measuring proximity in a high-dimensional embedding space.

Rather than relying on generative models, the system encodes medical questions into vector representations and retrieves the closest matching answer from a curated dataset. This approach prioritizes interpretability, robustness, and controlled outputs, which are especially important in medical domains.

The project covers the full pipeline from data preprocessing and exploratory analysis to model fine-tuning and API deployment.

# Key features
- Deterministic Retrieval Pipeline, ensuring all answers originate from verified clinical sources
- Semantic Question Answering based on sentence embeddings over a curated medical QA dataset
- Fine-tuned Transformer Model trained with triplet loss for improved semantic alignment
- Embedding-based Similarity Search using cosine similarity for answer selection
- Clear Architectural Separation between data preprocessing (ETL), model training, and inference
- Lightweight Flask API for serving predictions in a production-ready setup

# Data processing
The system uses a curated medical question–answer dataset in XML format.

Processing steps include:
- XML parsing and structured extraction
- Removal of incomplete entries
- Text normalization and cleaning
- Exploratory data analysis (label distribution, imbalance inspection)

The resulting dataset is used both for model fine-tuning and inference-time retrieval.

# Model approach
Base model: all-MiniLM-L6-v2 (Sentence Transformers)

Fine-tuning strategy: 

To specialize the model in clinical semantics, we employ **Triplet Loss**. The objective is to minimize the distance between a question and its correct answer while maximizing the distance to incorrect ones.

The goal is to satisfy the condition:
$$d(a, p) + m < d(a, n)$$

Where:
- $a$ (anchor): User query.
- $p$ (positive): Ground truth answer.
- $n$ (negative): Randomly sampled clinical answer.
- $m$: Margin hyperparameter.

The loss function $L$ is defined as:
$$L = \max(d(a, p) - d(a, n) + m, 0)$$

Inference:
- Encode user query
- Compute cosine similarity against precomputed answer embeddings
- Return the highest-scoring answer

# Project structure
```
.
.
├── app/
│   ├── static/                 # CSS/JS assets for Flask
│   ├── templates/              # HTML templates (index.html)
│   ├── main.py                 # Flask API entry point
│   ├── model.py                # Model class and inference logic
│   ├── preprocess_data.py      # XML parsing and data cleaning
│   ├── train.py                # Fine-tuning implementation
│   └── utils.py                # Text preprocessing utilities
│
├── data/                       # Raw Medical XML datasets
│
├── model/
│   └── finetuned_model/        # Fine-tuned SentenceTransformer
│
├── notebooks/                  # Exploratory Data Analysis (EDA)
│
└── requirements.txt            # Project dependencies

```

# Running the project
1. Install dependencies
```
pip install -r requirements.txt
```
2. Train / fine-tune the model
```
python train.py
```
3. Run the API
```
python main.py
```
The API exposes a simple endpoint to submit medical queries and retrieve the most relevant answer.

# Future improvements
This project focuses on the core logic of semantic retrieval. Based on this implementation, several clear extension paths emerge:
- Scalability: For larger datasets, in-memory similarity search becomes inefficient. Integrating a vector database (e.g. FAISS) would enable scalable and faster retrieval.
- Negative Mining: Negatives are currently sampled at random. Incorporating hard negative mining (semantically similar but incorrect answers) would improve embedding discrimination and retrieval precision.
- Data Persistence: Replacing raw XML parsing with a structured storage format (e.g. Parquet) would significantly reduce startup time and improve data pipeline efficiency.
 
