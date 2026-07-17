# What Film Ratings Reveal About LLM Safety Classification

> **Trust & Safety · Machine Learning · LLM Policy**

*A metadata-driven approach to predicting regulatory risk at scale.*

**By Kristina Churilova & Tamara Grigoryeva, Trust & Safety Policy, Amazon**

| Films Analyzed | Independent Analysts | Models Trained | Top Accuracy |
|:-:|:-:|:-:|:-:|
| **7,707** | **2** | **8** | **79.6%** |

---

## About This Project

This project was conducted independently by two colleagues on the same Trust & Safety policy team at Amazon, both working on content compliance and regulatory risk detection at scale. Kristina Churilova used Orange Data Mining, a visual no-code environment, running 5-fold cross-validation across five algorithms. Tamara Grigoryeva used Python in Google Colab, applying SMOTE oversampling, TF-IDF vectorization, and ordinal label mapping through XGBoost. Neither saw the other's results until both analyses were complete.

---

## Abstract

Content platforms operating at scale face a structural efficiency problem: the volume of titles requiring regulatory and policy assessment consistently outpaces analyst capacity. This project tests whether machine learning can close that gap. Two analysts independently built content severity classifiers on the same open-source dataset — the IMDb Parental Guide dataset and The Movies Dataset from Kaggle, joined and cleaned down to roughly 7,700 films — with no proprietary or confidential data involved.

Both classifiers predicted MPAA rating (G, PG, PG-13, R, or NC-17) from structured metadata alone, without access to the film's actual content. Both exceeded 77% accuracy on a four-class problem with severe class imbalance. Both independently converged on gradient boosting as the strongest algorithm, and both identified the same failure mode: the boundary between PG-13 and R, where content policy judgment is most contested in practice.

The result shows that non-engineering practitioners, working with open data and accessible tooling, can build classifiers that carry real predictive signal for content severity. The methodology — ordered severity labels, rare-class handling, and metadata-plus-text feature combination — generalizes directly to content moderation severity classification, AI-generated content detection, and training data labeling for LLM safety pipelines.

---

## 1. The Problem

Content platforms face a structural bottleneck: the volume of titles requiring regulatory and policy assessment consistently outpaces analyst capacity, and every unreviewed title carries real exposure — it may need restriction, age-gating, or removal, and it sits unclassified while review queues run into the thousands. Regulators and brand partners increasingly expect proactive controls rather than reactive cleanup after violations reach audiences, and expanding analyst headcount linearly with catalog growth isn't economically viable. The practical need is a system that estimates a title's risk tier at the moment of ingest, before an analyst reviews it, so that human attention goes where it's needed most.

A title rated NC-17 — the most restrictive MPAA classification, reserved for content appropriate only for adults — that enters a platform before its rating is verified represents a direct failure: a minor encounters no restriction in place. The failure isn't malicious. It's a classification lag compounded by volume, and its consequences range from user harm to regulatory sanction to brand damage with distribution partners.

We used the MPAA rating as a stand-in for this broader problem. It's a structured decision, assessed across violence, sexual content, profanity, substance use, and intensity, that reflects the same judgment a T&S analyst applies when weighing content against platform policy, regional requirements, and distribution standards. A classifier that predicts MPAA rating from metadata is structurally a classifier that estimates content risk from metadata — the architecture is the same.

---

## 2. Data & Methodology

Both analyses used exclusively open-source Kaggle data, with no proprietary or confidential inputs of any kind — a deliberate choice, so the project would be reproducible by any practitioner with public data access.

**IMDb Parental Guide Dataset** — content-intensity ratings across five categories (sex, violence, profanity, drugs, frightening/intense scenes), each scored None to Severe, plus MPAA rating and certification data.

**The Movies Dataset** — film metadata including title, genre, runtime, original language, production country, release date, popularity, plot overview, and cast/crew.

The two were joined on IMDb title ID and film title. The combined raw source exceeded 40,000 rows; after removing null MPAA labels, dropping columns with systematic quality issues (budget and revenue were excluded for pervasive zero-value and plac
