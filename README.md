#**Investigating Reward Hacking in Reasoning Models**

## **Abstract**

The assignment aims to investigate the reward seeking tendencies of
various models where they choose to perform actions showing evaluation
characteristics over the general instructions.

The models are ideally expected to follow the general instruction
irrespective of presence of other distracting instructions. But a
specialised environment-based experimentation revealed that models are
lured towards the output that links to the instruction containing some
sort of evaluation threat to the model. The results from this studies
were further tested using three more experiments namely, CoT
verbalization analysis, logit monitoring and Probing. The code can be
found here.

##**Code Structure**

### Files

| File | Role |
|---|---|
| `odd_number_full_gaming_sweep.ipynb` | Main pipeline: builds the prompts, runs the gaming sweep, then runs the three follow-up analyses (CoT judge, logit monitoring, probing). |
| `train_probing.py` | Activation extractor. Loads `Qwen/Qwen3.5-0.8B`, runs each sentence of an input `.txt` through the model and saves the final-layer / final-token hidden state to a `.npy` matrix. Run once per class (reward-instruction sentences vs. contrastive sentences). |
| `train_probe.py` | Trains the linear probe. Loads the two `.npy` activation matrices, labels them 0 / 1, does a stratified train/test split, fits an `sklearn` `LogisticRegression`, prints accuracy / ROC-AUC / confusion matrix, and dumps the fitted classifier with `joblib`. |
| `modulo_2_sentences.txt` | Synthetic "reward-aware" sentence set (contains `% 2` / modulo phrasing) used to train the probe. |
| `gaming_experiment_results.json` | Sweep output: nested `{model: {condition: {order: {gaming_rate, n_total, n_odd, records}}}}`. `records` keeps every gamed (odd) completion plus up to 10 non-gamed ones, each with its raw CoT. `*.unfiltered.json` is the same without the record cap. |
| `eval_awareness_classified.jsonl` | One line per completion with the LLM judge's `verbalized_eval_awareness` verdict and supporting quote. |
| `eval_awareness_rates.json` | Verbalized-eval-awareness rate aggregated per `(model, condition, order)`. |
| `logit_monitoring_results.json` | Per-completion logit-monitor scores (probability of the indicator phrase at each CoT sentence boundary, plus the per-completion maximum). |
| `logistic_regression_classifier.pkl` | The fitted probe (`joblib` pickle of the `LogisticRegression`). |
| `probe_position_accuracy.json` | Probe applied to the sweep prompts at two checkpoints (instruction only vs. instruction + grader): firing rate, mean P(concept), and pre/post separation. |

### Notebook flow (`odd_number_full_gaming_sweep.ipynb`)

1. **Setup** – HF login, environment fixes, run-size config (`N_VARIANTS`, `MAX_NEW_TOKENS`, `RESULTS_PATH`).
2. **Model registry** – `MODEL_CONFIGS` holds per-model quirks (system-role support, `enable_thinking`, `trust_remote_code`, 4-bit loading); `load_model_and_tokenizer` / `free_model` handle lifecycle.
3. **Prompt component banks** – deterministic generators for the four condition families (`naming`, `paraphrase`, `threat`, `anti_gaming`), each enumerating exactly 100 variants of `(instruction, grader)`.
4. **Message building / generation / parsing** – `build_user_prompt` places instruction and grader in `original` or `swapped` order; `generate_response` applies the chat template and samples a completion; `extract_number` / `classify` pull the answer and label it odd / even / unparsed.
5. **Experiment runner** – `run_condition_for_model` iterates variants for one `(condition, order)`; `gaming_rate` = fraction of odd outputs.
6. **Main sweep** – for each model, run all conditions × both orders, checkpoint to `gaming_experiment_results.json` after each model, free GPU memory. `filter_records_for_saving` keeps all gamed completions + 10 non-gamed per cell.
7. **Summary table** – gaming rate per `(model, condition)` for `original` vs. `swapped`.
8. **CoT analysis (LLM judge)** – flatten the results file, send each raw CoT to a Groq-hosted `gpt-oss` judge with a strict "verbalized evaluation awareness" rubric, cache verdicts to `eval_awareness_classified.jsonl`, aggregate to `eval_awareness_rates.json`.
9. **Logit monitoring** – reload the model (4-bit), split each CoT into sentences, and at every sentence boundary score the joint probability of the indicator phrase `"The grader rewards odd numbers"` in one forward pass (`logits_to_keep` to bound VRAM); store per-boundary and max scores to `logit_monitoring_results.json`.
10. **Probing** – load `logistic_regression_classifier.pkl`, reconstruct each sweep prompt from its `variant_id`, extract the final-layer / final-token activation after the instruction only and after instruction + grader, apply the probe, and report how much the modulo-2 concept becomes decodable once the grader is present (`probe_position_accuracy.json`).

### Probe training path

`modulo_2_sentences.txt` (+ a contrastive no-reward set) → `train_probing.py` → `final_token_activations_*.npy` → `train_probe.py` → `logistic_regression_classifier.pkl` → notebook probing cell.

## **Results**

### Gaming rate (Odd Number environment)

Fraction of completions that output an odd number, violating the explicit "even" instruction.

| Model | Condition | Original | Swapped |
|---|---|---|---|
| Qwen3.5-0.8B | naming | 2.0% | 4.0% |
| Qwen3.5-0.8B | paraphrasing | 4.0% | 3.0% |
| Qwen3.5-0.8B | threat | 2.0% | 3.0% |
| Qwen3.5-0.8B | anti_gaming | 5.0% | 3.0% |
| Phi-mini-reasoning-2.3B | all | not run (CUDA OOM on 8 GB) | |
| gemma-3-4B-it | all | not run (CUDA OOM on 8 GB) | |

Gaming rate stays low (2–5%) for Qwen and shows no consistent `original` vs. `swapped`
gap, so the behaviour is not explained by recency bias (H0 not supported), and grader
sensitivity at this model scale is weak.

### CoT analysis – verbalized evaluation awareness

Rate at which the LLM judge found the CoT explicitly reasoning about being evaluated / graded.

| Model | Eval-awareness rate |
|---|---|
| Qwen3.5-0.8B | 15.4% for `anti_gaming` (swapped) only; 0.0% for every other condition/order |
| Phi-mini-reasoning-2.3B | not run (compute constraint) |
| gemma-3-4B-it | not run (compute constraint) |

### Logit monitoring

Maximum probability assigned to the indicator phrase `"The grader rewards odd numbers"`
across CoT sentence boundaries.

| Model | Condition | Original | Swapped |
|---|---|---|---|
| Qwen3.5-0.8B | naming | ~0 | ~0 |
| Qwen3.5-0.8B | paraphrasing | ~1e-6 | ~0 |
| Qwen3.5-0.8B | threat | ~7e-6 | ~2e-6 |
| Qwen3.5-0.8B | anti_gaming | ~1.7e-5 | ~4.5e-5 |
| Phi-mini-reasoning-2.3B | all | not run (compute constraint) | |
| gemma-3-4B-it | all | not run (compute constraint) | |

Absolute probabilities are negligible everywhere; the only visible ordering is that
`anti_gaming` (which names the exploit explicitly) scores highest, consistent with the
CoT-awareness result.

### Probing (Qwen3.5-0.8B, 106 sweep prompts)

Probe = `LogisticRegression` on the final-layer / final-token activation, trained to 100%
train/test accuracy on the synthetic modulo-2 vs. contrastive sentences. Applied at two
checkpoints of each sweep prompt.

| Metric | After instruction only | After instruction + grader |
|---|---|---|
| Probe fires (P ≥ 0.5) | 0.0% | 0.0% |
| Mean P(concept present) | 0.0003 | 0.0190 |

Pre-grader vs. post-grader separation: pooled hard-label accuracy 0.50, **AUROC 0.985**.

The probe never crosses its decision threshold on these prompts, but the modulo-2
direction still separates "grader seen" from "grader not seen" almost perfectly by score
(AUROC ≈ 0.99): adding the grader reliably pushes activations along the reward-concept
axis by ~60×, even though the model rarely acts on it.

Caveat: the probe is trained off-policy on synthetic sentences, so it may be keying on
lexical `% 2` patterns rather than the model's internal grader awareness. An on-policy
data-generation approach (steering + KL-regularised deceptive-completion synthesis) is the
suggested fix for future work.

## Resources

All experiments ran locally on an RTX 5050 (8 GB VRAM, 16 GB system RAM). Only
Qwen3.5-0.8B fit; Phi-4-mini-reasoning and gemma-3-4b-reasoning hit CUDA OOM locally and
timed out on Colab T4 during CoT capture.
