# Anchor

A computer-vision integrity assurance framework.

## Project Structure

```
anchor/
├── data_integrity/       # Dataset validation and hash verification
├── model_integrity/      # Model weight verification and tampering detection
├── provenance/           # Image and model provenance tracking
├── distribution_shift/   # Distribution shift detection between datasets
├── orchestrator/         # Pipeline orchestration and evaluation runner
├── report_schema/        # Structured report generation (PDF / HTML)
├── attack_scenarios/     # Adversarial attack definitions and test cases
├── ui/                   # Streamlit dashboard for interactive evaluation
├── reference_data/       # Baseline reference datasets and embeddings
├── tests/                # Unit and integration tests
├── requirements.txt      # Pinned Python dependencies
├── docker-compose.yml    # Air-gapped evaluation environment
└── README.md
```

## Setup

Anchor is designed to run in network-isolated and air-gapped environments. Before running the pipeline, you must download the required neural network weights and datasets once with internet access:

```bash
# 1. Download OpenCLIP ViT-B-32 weights (~350 MB)
python scripts/download_weights.py

# 2. Download CIFAR-10 dataset (~163 MB, optional; fallback is synthetic data)
python scripts/download_cifar10.py
```

These scripts save the assets into the `reference_data/` directory, allowing subsequent pipeline runs to be fully offline.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run evaluation (air-gapped via Docker)
docker compose up anchor-eval
```

## License

(c)2026 Muhesh Kanna M
