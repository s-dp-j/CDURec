# CDURec: Cross-Domain and Cross-User Recommendation for Inactive Users

This repository provides the implementation of **CDURec**, a Cross-Domain and
cross-User collaborative filtering Recommendation framework designed to improve
recommendation performance for inactive users.

CDURec jointly exploits information from auxiliary domains and relatively rich
target-domain feedback from active users. Instead of treating all target-domain
users under the same transfer strategy, CDURec uses active users as supervised
anchors to learn cross-domain preference correspondence and subsequently transfers
the learned knowledge to inactive users.

---

## Overview

The main procedure of CDURec consists of the following stages:

1. **User division**  
   Users are divided into active and inactive users according to their rating
   density in the target domain.

2. **Latent-factor learning**  
   Funk-SVD is used to learn user and item latent factors in the target and
   auxiliary domains.

3. **Active-user-guided preference mapping**  
   The latent factors of active users in the auxiliary and target domains are
   used to construct a supervised mapping problem.

4. **Self-taught representation learning**  
   A Stacked Denoising Autoencoder (SDAE) is first pretrained using the
   auxiliary-domain latent factors of all users. The encoder is then fine-tuned
   together with a linear regression head using active users to learn the
   cross-domain preference mapping \(F\).

5. **Inactive-user recommendation**  
   The learned mapping \(F\) estimates the target-domain latent factors of
   inactive users. These transferred user representations and the
   active-user-derived target-domain item representations are incorporated into
   Restricted-Funk-SVD as transfer constraints to obtain the final predictions.

---

## Requirements

The experiments in the paper were implemented using:

- Python
- PyTorch
- scikit-learn
