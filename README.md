# CANN-Bench

A benchmarking framework for testing anatomy-constrained AI models against standard baselines.

Built as a portfolio project to explore a question from Sean Froudist-Walsh's lab at Oxford:
does injecting a biologically-motivated structure into a transformer's weights change how it
behaves — and if so, on what kinds of text?

---

## What this does

The core idea comes from Froudist-Walsh et al. (2021), who found that dopamine receptor
density follows a gradient across the primate cortex — higher in frontal regions involved
in working memory, lower in sensory regions at the back. This gradient appears to regulate
how strongly different brain regions sustain and manipulate information under cognitive load.

CANN-Bench operationalises this as a weight modulation scheme: take a transformer's attention
weights, stretch the atlas density values to match their dimensions, and multiply. High-density
regions get amplified, low-density regions get attenuated.

```
W_constrained = W_baseline ⊙ (1 + α · (d - 0.5) · 2)
```

The question is whether this makes the model behave differently in any meaningful way.

---

## Clinical extension: suicide ideation detection

The original benchmark runs on tweet_eval/emotion (4-class emotion classification).
I also extended it to a clinical domain using a fine-tuned DistilBERT model from my
MSc dissertation — trained on 232,074 Reddit posts to detect suicidal ideation.

This gave me a proper 2×2 factorial design:

|                  | No constraint | + Dopamine constraint |
|------------------|:-------------:|:--------------------:|
| **Fine-tuned**   | Model A        | Model B               |
| **Vanilla**      | Model C        | Model D               |

The finding I care most about: applying the constraint to the fine-tuned model improved
recall on the suicide class by +1.5pp (0.796 → 0.811) and improved accuracy on
high-entropy (linguistically complex) texts by +1.6pp. The constraint appears to help
most precisely where the model is under the most representational stress — which is
what the biological hypothesis would predict.

Full results and interpretation are in [FINDINGS.md](FINDINGS.md).

---

## Project structure

```
src/
  ingestion/       load tweet_eval from HuggingFace or Suicide_Detection.csv from Kaggle
  preprocessing/   text cleaning + Shannon entropy computation
  atlas/           load brain atlas CSV, stretch densities to weight dimensions
  benchmark/       model loading, constraint application, inference, evaluation
  utils/           config, SQLite logging

data/
  atlases/         dopamine_gradient.csv (21 cortical regions, normalised densities)
  raw/             original datasets (gitignored)
  processed/       cleaned + featurised data (gitignored)

results/           SQLite database with all benchmark runs (gitignored)
tests/             pytest unit tests for every module
```

---

## Quickstart

```bash
git clone https://github.com/mehmetfurkanakpinar/CANN-Bench.git
cd CANN-Bench
conda env create -f environment.yml
conda activate cann-bench

# Run the emotion benchmark (offline mock data, no HuggingFace needed)
python -m src.benchmark.run --offline --max-rows 200

# Run tests
pytest tests/ -v
```

---

## Shannon entropy

Text complexity is measured using Shannon entropy over the token distribution:

```
H(X) = -Σ p(x) log₂ p(x)
```

High entropy = complex, unpredictable text. Low entropy = simple, repetitive text.
This is used to stratify results and test whether the anatomy constraint interacts
with linguistic complexity — which it does, at least in the clinical benchmark.

---

## Results are stored in SQLite

Every run logs model name, atlas used, constraint strength, seed, and all metrics
to `results/benchmark.db`. This means results are queryable and reproducible across
machines without relying on print output.

```bash
sqlite3 results/benchmark.db \
  "SELECT model_name, accuracy, f1, notes FROM benchmark_runs ORDER BY run_timestamp DESC;"
```

---

## References

Froudist-Walsh et al. (2021). *A dopamine gradient controls access to distributed working
memory in the large-scale monkey cortex.* Neuron 109(21): 3500–3520.

Froudist-Walsh et al. (2023). *Gradients of neurotransmitter receptor expression in the
macaque cortex.* Nature Neuroscience 26(7): 1281–1294.
