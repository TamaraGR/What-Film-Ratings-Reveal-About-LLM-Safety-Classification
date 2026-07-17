# What Film Ratings Reveal About LLM Safety Classification

A metadata-driven approach to predicting regulatory risk at scale.

**By Kristina Churilova & Tamara Grigoryeva** — Trust & Safety Policy, Amazon

| Films Analyzed | Independent Analysts | Models Trained | Top Accuracy |
|---|---|---|---|
| 7,707 | 2 | 8 | 79.6% |

## About This Project

This project was conducted independently by two colleagues on the same Trust & Safety policy team at Amazon, both working on content compliance and regulatory risk detection at scale.

- **Kristina Churilova** used Orange Data Mining, a visual no-code environment, running 5-fold cross-validation across five algorithms.
- **Tamara Grigoryeva** used Python in Google Colab, applying SMOTE oversampling, TF-IDF vectorization, and ordinal label mapping through XGBoost.

Neither saw the other's results until both analyses were complete.

## Abstract

Content platforms operating at scale face a structural efficiency problem: the volume of titles requiring regulatory and policy assessment consistently outpaces analyst capacity. This project tests whether machine learning can close that gap.

Two analysts independently built content severity classifiers on the same open-source dataset — the IMDb Parental Guide dataset and The Movies Dataset from Kaggle, joined and cleaned down to roughly 7,700 films — with no proprietary or confidential data involved. Both classifiers predicted MPAA rating (G, PG, PG-13, R, or NC-17) from structured metadata alone, without access to the film's actual content.

Both exceeded 77% accuracy on a four-class problem with severe class imbalance. Both independently converged on gradient boosting as the strongest algorithm, and both identified the same failure mode: the boundary between PG-13 and R, where content policy judgment is most contested in practice.

The result shows that non-engineering practitioners, working with open data and accessible tooling, can build classifiers that carry real predictive signal for content severity. The methodology — ordered severity labels, rare-class handling, and metadata-plus-text feature combination — generalizes directly to content moderation severity classification, AI-generated content detection, and training data labeling for LLM safety pipelines.

## 1. The Problem

Content platforms face a structural bottleneck: the volume of titles requiring regulatory and policy assessment consistently outpaces analyst capacity, and every unreviewed title carries real exposure — it may need restriction, age-gating, or removal, and it sits unclassified while review queues run into the thousands.

Regulators and brand partners increasingly expect proactive controls rather than reactive cleanup after violations reach audiences, and expanding analyst headcount linearly with catalog growth isn't economically viable. The practical need is a system that estimates a title's risk tier at the moment of ingest, before an analyst reviews it, so that human attention goes where it's needed most.

A title rated NC-17 — the most restrictive MPAA classification, reserved for content appropriate only for adults — that enters a platform before its rating is verified represents a direct failure: a minor encounters no restriction in place. The failure isn't malicious. It's a classification lag compounded by volume, and its consequences range from user harm to regulatory sanction to brand damage with distribution partners.

We used the MPAA rating as a stand-in for this broader problem. It's a structured decision, assessed across violence, sexual content, profanity, substance use, and intensity, that reflects the same judgment a T&S analyst applies when weighing content against platform policy, regional requirements, and distribution standards. A classifier that predicts MPAA rating from metadata is structurally a classifier that estimates content risk from metadata — the architecture is the same.

## 2. Data & Methodology

Both analyses used exclusively open-source Kaggle data, with no proprietary or confidential inputs of any kind — a deliberate choice, so the project would be reproducible by any practitioner with public data access.

- **IMDb Parental Guide Dataset** — content-intensity ratings across five categories (sex, violence, profanity, drugs, frightening/intense scenes), each scored None to Severe, plus MPAA rating and certification data.
- **The Movies Dataset** — film metadata including title, genre, runtime, original language, production country, release date, popularity, plot overview, and cast/crew.

The two were joined on IMDb title ID and film title. The combined raw source exceeded 40,000 rows; after removing null MPAA labels, dropping columns with systematic quality issues (budget and revenue were excluded for pervasive zero-value and placeholder entries), and filtering to complete feature sets, the working dataset came to approximately 7,700 films.

One field was deliberately excluded from every model: the MPAA explanation text (e.g. "Rated R for strong violence and language throughout"). Including it would be data leakage — the classification rationale embedded as an input feature, which the model would learn to reverse-engineer rather than predict.

### Dataset Composition

Class distribution across 7,707 films — severe imbalance toward R-rated content:

| Rating | Count |
|---|---|
| R | 4,631 |
| PG-13 | 2,124 |
| PG | 921 |
| NC-17 | 34 |

R-rated films represent ~60% of the dataset. NC-17 accounts for fewer than 0.5% — the central engineering challenge of this project.

![Dataset composition bar chart](blob/main/01%20dataset%20composition.png)

### Methodological Differences

| Decision | Kristina — Orange Data Mining | Tamara — Python / Colab |
|---|---|---|
| Evaluation strategy | 5-fold stratified cross-validation | 70/15/15 stratified hold-out split |
| Algorithms compared | LR, SVM, Random Forest, Neural Network, Gradient Boosting | LR, Random Forest, XGBoost |
| Text vectorization | N-grams / Bag of Words | TF-IDF (1,000 features, min doc freq 2) |
| NC-17 imbalance | Real distribution; stratified CV preserved proportion | SMOTE: 24 real examples expanded to 3,240 |
| Label encoding | Standard multiclass | Ordinal mapping: PG=0, PG-13=1, R=2, NC-17=3 |

## 3. Results

Both analyses converged on gradient boosting as the strongest algorithm, and both exceeded 77% accuracy on the four-class problem.

**Best Model Comparison**

| Analyst | Model | Accuracy |
|---|---|---|
| Kristina | Gradient Boosting | 79.6% |
| Tamara | XGBoost + SMOTE | 77.8% |
| Tamara | Random Forest | 76.5% |

![Best model comparison bar chart](blob/main/02%20model%20comparison.png)

Kristina's model additionally scored AUC 0.917, F1 0.793, MCC 0.621.

For a four-class problem with severe imbalance, both results carry meaningful signal. The two-point accuracy gap is less significant than what produced it: Tamara's SMOTE approach shifted the model's behavior on the NC-17 class — producing predictions where the natural-distribution model produced none — at a modest cost to overall accuracy. Whether that trade-off is worth it depends on the operational objective, covered in Section 6.

## 4. What the Errors Revealed

Accuracy tells you how often the model was right. Where it was wrong is the more useful story.

![Error concentration schematic showing the PG-13 to R boundary as the dominant failure mode](blob/main/04%20error%20concentration.png)

Errors concentrated almost entirely at the PG-13 ⇄ R boundary — adjacent-category confusion, not random misclassification. Films with elevated profanity or moderate violence codes that analysts had rated PG-13 were frequently predicted as R, and vice versa. This reflects genuine ambiguity in the underlying decision, not model weakness — it's contested in practice.

*The King's Speech* received an R in the United States solely for one scene of sustained profanity, while the same film was rated 12A in the UK. The MPAA's own guidelines don't specify a language threshold at that boundary; analysts apply judgment, and the model learned that contested zone accurately.

NC-17 remained the hardest class to detect. With only 34 labeled examples, a model trained on the natural distribution lacks enough signal to learn a reliable decision boundary. Tamara's SMOTE approach addressed this directly, generating synthetic examples so the model could learn the boundary at all — the methodologically appropriate response to training-data scarcity in a high-stakes minority class, not a workaround.

XGBoost was more confident but also more overconfident. Tamara's analysis found 86 high-confidence errors (above 70% predicted probability) versus 39 for Random Forest. In a compliance context, that matters: a model that's certain when it's wrong is harder to catch than one that expresses uncertainty.

Kristina's experiment removing NC-17 entirely and retraining as a three-class problem produced only a marginal accuracy gain — confirming that NC-17's rarity was a real challenge, but not the primary source of error. The PG-13/R boundary was.

## 5. What Drove the Predictions

Both models independently identified the same feature hierarchy, ranked by information gain:

1. Family (genre)
2. Profanity code
3. Sex code
4. Animation (genre)
5. Violence code
6. Release year
7. Drug code
8. Plot text (TF-IDF)

![Feature importance horizontal bar chart](blob/main/03%20feature%20importance.png

Genre and content-intensity codes dominate. Release year, an engineered feature, outranks most raw metadata.

- **Genre was the strongest structural signal.** Family and Animation genre flags topped both feature importance rankings — genre encodes content expectations more reliably than any single intensity score.
- **Content-intensity codes followed immediately**, validating the core hypothesis: structured severity scores carry real predictive signal for content classification outcomes.
- **Release year was independently discovered by both analysts.** Neither began the project intending to engineer it. Content standards shift over time — PG-13 didn't exist before 1984, so a 1978 film with moderate violence sits in a different classification context than a 2012 film with identical content codes. The model needed time as a dimension to interpret severity correctly across eras.
- **Plot text contributed less than expected.** Both analysts vectorized plot overviews, and both found it added signal, but less than the structured content codes. The most frequent terms in both models' top features were genre-adjacent, meaning the text was mostly re-deriving genre signal already captured by the genre flags.

## 6. The SMOTE Question

With 34 examples, no model trained on the real distribution has enough data to learn a reliable NC-17 boundary. The two analysts made opposite calls on how to handle that.

- **Kristina kept NC-17 at its natural prevalence.** The argument: a model should reflect the real distribution of content it will encounter. NC-17 is genuinely rare, and the model should express that rarity rather than over-predicting it.
- **Tamara applied SMOTE**, generating 3,216 synthetic NC-17 examples by interpolating between the 24 real ones, to bring the class to parity before training. The argument: in a compliance context, a false negative on the most extreme class is the costliest possible error. A title with NC-17-level content routed to R because the model never learned to look for NC-17 is worse than a false positive that routes a borderline R title to extra review.

Both positions are defensible within their own logic. Which one is correct depends on what the classifier is optimizing for — precision across the full distribution, or recall on the highest-severity class — and that's a policy decision, not a modeling one.

## 7. Shared Limitations

- **Labels are crowd-sourced, not authoritative.** IMDb parental guide scores are aggregated from user contributions — a useful proxy for intensity, but without the consistency or accountability of a formal T&S or regulatory decision.
- **The dataset reflects content through 2017.** Content norms and classification standards have kept evolving. Revalidation against recent titles would be required before any operational use.
- **NC-17 needs more ground-truth examples.** SMOTE synthesizes from existing examples; if the 34 real NC-17 films aren't representative of the broader class, synthetic examples inherit that gap. Expanding the labeled set through official classification databases is the right next step.
- **Budget and revenue data was unusable.** Both columns had systematic data quality issues at scale and were excluded. Popularity and vote count served as proxies for production scale.

This was a proof of concept, built on open data and accessible tooling. It shows the methodology is reachable and the signal is real. It is not a production-grade classifier.

## 8. Where Else This Applies

The structural pattern here — an ordered severity scale, a rare class that carries the highest consequence, multiple interacting signals, and the need to estimate risk before human review — shows up concretely across T&S and AI safety work:

- **Content Moderation Severity Tiers** — A harassment queue that runs from a rude comment to a targeted threat to a coordinated pile-on with doxxing risk. The rarest tier — direct threats paired with identifying information — is exactly the NC-17 problem: the highest-consequence class has the fewest labeled examples to learn from.
- **AI-Generated Content Detection** — A classifier flagging synthetic media from upload metadata (generation source, edit patterns, account behavior) faces the same imbalance: confirmed deepfake-for-harm content is a sliver of total uploads, but a missed one is a crisis, not a rounding error in an accuracy score.
- **Coordinated Inauthentic Behavior** — The spectrum from a single bot account boosting engagement to a state-linked network coordinating disinformation ahead of an election. Misreading a state-linked campaign as opportunistic spam is a materially different failure than the reverse — the same ordinal, rare-class architecture applies directly.
- **LLM Safety Training Data Labeling** — Building a labeled corpus for red-teaming and refusal evaluation, where confirmed high-severity prompts (e.g. CBRN-adjacent requests) are necessarily rare relative to total prompt volume in any responsibly curated set — but a false negative there matters far more than one on a mildly rude prompt. The SMOTE-plus-ordinal-mapping approach here is a direct methodological input to that labeling problem.
- **Cross-Jurisdiction Compliance Triage** — A title cleared for a US PG-13 audience but flagged under Germany's FSK or the UK's BBFC for a single scene. A metadata-driven classifier trained on jurisdiction-specific decisions, rather than one rating system, could surface titles likely to need regional review before distribution — without requiring a compliance analyst to rewatch every title for every market.

## Data Sources

- IMDb Parental Guide Dataset (Kaggle)
- The Movies Dataset (Kaggle)
- Joined on IMDb Title ID

## Tooling

- Orange Data Mining
- Python / Google Colab (XGBoost, SMOTE, TF-IDF)
- No proprietary or confidential data used

## Authors

- **Kristina Churilova** — Trust & Safety Policy, Amazon
- **Tamara Grigoryeva** — Trust & Safety Policy, Amazon
