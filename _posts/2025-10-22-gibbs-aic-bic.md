---
layout: default
title: "Gibbs-Based Information Criteria and the Over-Parameterized Regime"
date: 2025-10-22
author: Papay Ivan
---

# Gibbs-Based Information Criteria and the Over-Parameterized Regime

*Based on the paper by Haobo Chen, Yuheng Bu, and Gregory W. Wornell (arXiv:2306.05583)*

---

## Motivation

Classical model selection criteria—**AIC** and **BIC**—fail in the **over-parameterized regime** (\(p > n\)), where models interpolate training data perfectly yet generalize well (the *double descent* phenomenon). This paper proposes **Gibbs-based AIC and BIC**, grounded in **information-theoretic analysis** of the **Gibbs algorithm**, which remain valid even when \(p \gg n\).

---

## Key Concepts

### 1. Gibbs Algorithm

Instead of point estimates (e.g., MLE), the Gibbs algorithm defines a **posterior distribution** over parameters:

\[
P_{W|S}(w|s) = \frac{\pi(w) \exp(-\beta L_E(w, s))}{\mathbb{E}_\pi[\exp(-\beta L_E(W, s))]},
\]

where:
- \(\pi(w)\) is a prior,
- \(L_E(w, s)\) is empirical risk,
- \(\beta > 0\) controls the "temperature".

This distribution minimizes the **Information Risk**:
\[
\min_{P_{W|S}} \mathbb{E}[L_E(W,S)] + \frac{1}{\beta} D(P_{W|S} \| \pi | P_S).
\]

### 2. Gibbs-Based AIC

The expected generalization error of the Gibbs algorithm is:

\[
\text{gen}(P_{W|S}, P_S) = \frac{I_{\text{SKL}}(P_{W|S}, P_S)}{\beta},
\]

where \(I_{\text{SKL}}\) is the **symmetrized KL information**. This leads to:

\[
\boxed{\text{AIC}^+ = L_E(\hat{w}_{\text{Gibbs}}, z^n) + \frac{1}{\beta} I_{\text{SKL}}(P_{W|S}, P_S)}
\]

In the classical regime (\(n \to \infty, p\) fixed, \(\beta = n\)), this recovers:
\[
\text{AIC}^+ \to L_E + \frac{p}{n}.
\]

### 3. Gibbs-Based BIC

For log-loss and \(\beta = n\), the negative log-marginal likelihood becomes:

\[
-\frac{1}{n} \log m(z^n) = \mathbb{E}_{P_{W|S}}[L_E(W, z^n)] + \frac{1}{n} D(P_{W|S} \| \pi).
\]

This motivates two versions:

\[
\begin{aligned}
\text{BIC}^+ &= L_E(\hat{w}_{\text{Gibbs}}, z^n) + \frac{1}{n} D(P_{W|S} \| \pi), \\
\text{BIC}^- &= \mathbb{E}_\pi[L_E(W, z^n)] - \frac{1}{n} D(\pi \| P_{W|S}).
\end{aligned}
\]

In the classical regime, \(\text{BIC}^+ \to L_E + \frac{p \log n}{2n}\).

---

## Over-Parameterized Regime: Random Feature Model

In the **Random Feature (RF) model**:
\[
g(x) = f\left(\frac{x^\top F}{\sqrt{d}}\right) w,
\]
with \(F_{ij} \sim \mathcal{N}(0,1)\), the Gibbs posterior is Gaussian:
\[
P_{W|S} \sim \mathcal{N}(\hat{w}_\lambda, \Sigma_w),
\]
where \(\hat{w}_\lambda = (\lambda n I + B^\top B)^{-1} B^\top y\).

Using **random matrix theory**, the KL divergence admits a closed form. As \(n, p \to \infty\) with \(r = p/n\) fixed:

\[
\text{BIC}^+ = L_E(\hat{w}_{\text{Gibbs}}) + \underbrace{\frac{\lambda}{2\sigma^2} \|\hat{w}_\lambda\|_2^2}_{\ell_2\text{ term}} + \underbrace{\frac{1}{2} V(1/\lambda, r) - \frac{\lambda}{8} F(1/\lambda, r)}_{\text{covariance term}},
\]

where \(F\) and \(V\) are explicit functions derived from the Marchenko–Pastur law.

---

## Experimental Insights

### Double Descent vs. Marginal Likelihood

- **AIC⁺** (generalization error proxy) exhibits **double descent**.
- **BIC⁺** (marginal likelihood proxy) **does not** — it monotonically decreases or plateaus.

This reveals a **fundamental mismatch** between generalization and marginal likelihood in over-parameterized settings.

![Double descent and BIC comparison](/assets/images/double_descent.jpg)

### Role of the Prior

The hyperparameter \(\lambda\) (from Gaussian prior \(w \sim \mathcal{N}(0, \sigma^2/(\lambda n) I)\)) strongly influences both BIC⁺ and generalization:

- Smaller \(\lambda\) → flatter posterior → smaller \(\ell_2\) norm → better generalization.
- But BIC⁺ penalizes large \(\lambda\) more heavily.

![KL divergence vs generalization](/assets/images/kl_div.jpg)

### Decomposition of BIC⁺ Penalty

The penalty in BIC⁺ consists of:
1. \(\ell_2\) norm of weights (decreases with \(p\)),
2. Covariance term (captures eigenstructure of \(B^\top B\)).

![Covariance divergence term](/assets/images/cov_div.jpg)

### Model Selection Performance

Classical BIC fails to select over-parameterized models, while **Gibbs-based BIC⁺ correctly favors large \(p\)**.

![BIC comparison across criteria](/assets/images/BIC_comparison.jpg)

![Sample-wise comparison](/assets/images/sample_comparison.jpg)

---

## Conclusion

- **Gibbs-based AIC/BIC** unify classical and modern regimes via **information theory**.
- They remain **well-defined** even when MLE is non-unique (\(p > n\)).
- **Marginal likelihood (BIC) ≠ generalization (AIC)** in over-parameterized settings — a crucial insight for model selection.
- The **choice of prior** (\(\lambda\)) critically affects both criteria.

This framework offers a principled way to understand **double descent**, **interpolation**, and **Bayesian model selection** in deep learning.

---

## References

- Chen, H., Bu, Y., & Wornell, G. W. (2023). *Gibbs-Based Information Criteria and the Over-Parameterized Regime*. arXiv:2306.05583.
- Watanabe, S. (2013). *A Widely Applicable Bayesian Information Criterion*.
- Belkin, M., et al. (2019). *Reconciling modern machine learning and the bias-variance tradeoff*.

---

© 2025 Papay Ivan. All rights reserved.
