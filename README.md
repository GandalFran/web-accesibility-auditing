# Visual-First Web Accessibility Auditing Framework

This repository contains the implementation and semantic artifacts for the paper **"Beyond the DOM: Visual-First Web Accessibility Auditing using Vision-Language Models and Semantically-Grounded Reports"** submitted to the Journal of Web Semantics.

## Overview

Traditional accessibility tools rely on DOM parsing, which misses visual-semantic violations (e.g., insufficient contrast on gradients, meaningful decorative images). This project introduces a neuro-symbolic framework that:
1.  **Captures High-Fidelity Screenshots**: Uses a headless browser to render pages exactly as users see them.
2.  **Perceives via VLM**: Uses LLaVA-1.5 to inspect visual elements.
3.  **Validates Outputs**: Enforces WCAG compliance through a symbolic firewall and SHACL constraints.

## Repository Structure

```
code/
├── ai/                  # Visual-First Perception Module (WCAG-VLM)
│   ├── src/             # Core pipeline (Perception, Firewall, Validation)
│   ├── prompts/         # Implicit Chain-of-Thought (ICoT) templates
│   ├── data/            # WebSight dataset subset and experimental results
│   ├── scripts/         # Reproducibility and setup scripts
│   └── orchestrate...   # Main entry point for audit execution
│
└── semantics/           # Formal semantic framework and ontology
    ├── ontology/        # RDF vocabulary, SHACL shapes, JSON-LD context
    ├── examples/        # Validated audit instances (Turtle format)
    └── queries/         # SPARQL competency questions and regression tests
```

## Components

### AI Module (`ai/`)

The **Visual-First Perception Module** is located in `ai/`. It utilizes LLaVA-1.5 and headless browsers (`pyppeteer`) for isomorphic rendering.

**Key Files:**
- `src/main_pipeline.py`: Main entry point for the auditing logic.
- `orchestrate_pipeline.py`: Top-level orchestrator to run the full audit workflow.
- `prompts/`: Contains the Implicit Chain-of-Thought (ICoT) prompt templates.

**Installation:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

**Usage:**
To execute the full audit pipeline on the sample data:
```bash
python orchestrate_pipeline.py
```

### Semantics Module (`semantics/`)

The `semantics/` directory provides the formal semantic framework for representing audit results, addressing the representational gap in accessibility auditing.

**Core Artifacts:**
- **RDF Vocabulary** (`ontology/vocab.ttl`): Formal ontology with 8 core classes (e.g., `waa:AuditRun`, `waa:Finding`, `waa:Criterion`) and 40+ properties.
- **SHACL Constraints** (`ontology/shapes.ttl`): Declarative validation rules (C1-C12) ensuring structural integrity.
- **JSON-LD Context** (`ontology/context.jsonld`): Interoperability mapping for JSON serialization.
- **SPARQL Queries** (`queries/`): Five competency questions demonstrating practical utility (cross-run regression detection, visual evidence distribution, provenance tracing, spatial clustering, recommendation retrieval), plus additional utility queries.

**Validation Workflow:**
You can validate the provided examples using `pyshacl`:
```bash
pyshacl -s ontology/shapes.ttl -e ontology/vocab.ttl examples/run1.ttl
```

### Dataset (`ai/data/`)

The framework is evaluated on the **WebSight v0.2** dataset (`HuggingFaceM4/WebSight`).

**Evaluation Subset:**
A stratified subset of **270 samples** was selected to ensure balanced representation across visual complexity, color palettes, and typographic structures.

**Experimental Results (N=270):**

> [!NOTE]
> **Reproducibility Information:** The provided `download_websight.py` script downloads the full **Development Set of 2,000 samples** to support large-scale experimentation. The specific results in Table 2 are derived from a **stratified Evaluation Subset of 270 samples** manually selected from this pool. Users should expect aggregate metrics on the full 2K set to differ slightly from the reported stratified subset.

| Criterion | Samples Flagged | Flag Rate | Verified Precision |
|-----------|----------------|-----------|-------------------|
| 1.1.1     | 85             | 31.5%     | 82%               |
| 1.4.3     | 67             | 24.8%     | 86%               |
| **Overall** | **152**      | **56.3%** | **84.0%**         |

**Performance stats:**
- **Symbolic Firewall Acceptance Rate**: 83.8%
- **Reviewers**: 2 WCAG-certified auditors (CPACC) with Cohen's κ = 0.81.

To reproduce the study, run `scripts/download_websight.py` to get the data, then `orchestrate_pipeline.py`.

## License

This project is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0). The underlying WebSight dataset is Apache 2.0.

## Contact

- **Francisco Pinto-Santos**: franpintosantos@usal.es
- **BISITE Research Group**: University of Salamanca, Spain
