# Diluting the normal sample with a trusted pool

From `scripts/analyse_price_pool.py` (prereg `docs/price-trusted-prereg.md`, Part 3). Cells: min catch over the 4 backdoored models at 0 / 1 / 2 / 5 / 10% contamination of the user's own traffic; worst known-kind false alarm (clean); h17 catch (clean).

**Primary (euclid, own+pool, p90): does not meet the bar clean.**

| score | sample | alarm | min catch 0/1/2/5/10% | worst FA known (clean) | h17 catch | call |
|---|---|---|---|---|---|---|
| euclid | own | p90 | 0.89 / 0.89 / 0.88 / 0.79 / 0.37 | 0.13 | 0.47 | does not meet the bar clean |
| euclid | own | p95 | 0.86 / 0.84 / 0.82 / 0.35 / 0.12 | 0.09 | 0.29 | does not meet the bar clean |
| euclid | own | p97.5 | 0.82 / 0.70 / 0.54 / 0.12 / 0.06 | 0.07 | 0.16 | does not meet the bar clean |
| euclid | own | p99 | 0.67 / 0.15 / 0.07 / 0.05 / 0.03 | 0.05 | 0.08 | does not meet the bar clean |
| euclid | own+pool | p90 | 0.87 / 0.87 / 0.87 / 0.86 / 0.84 | 0.12 | 0.47 | does not meet the bar clean |
| euclid | own+pool | p95 | 0.76 / 0.76 / 0.75 / 0.69 / 0.63 | 0.08 | 0.27 | does not meet the bar clean |
| euclid | own+pool | p97.5 | 0.67 / 0.66 / 0.65 / 0.59 / 0.46 | 0.05 | 0.19 | does not meet the bar clean |
| euclid | own+pool | p99 | 0.55 / 0.38 / 0.30 / 0.22 / 0.12 | 0.03 | 0.09 | does not meet the bar clean |
| euclid | pool | p90 | 0.87 | 0.12 | 0.44 | does not meet the bar clean |
| euclid | pool | p95 | 0.76 | 0.08 | 0.27 | does not meet the bar clean |
| euclid | pool | p97.5 | 0.66 | 0.06 | 0.19 | does not meet the bar clean |
| euclid | pool | p99 | 0.54 | 0.03 | 0.15 | does not meet the bar clean |
| median | own | p90 | 0.89 / 0.88 / 0.88 / 0.82 / 0.38 | 0.13 | 0.45 | does not meet the bar clean |
| median | own | p95 | 0.84 / 0.84 / 0.81 / 0.35 / 0.12 | 0.09 | 0.29 | does not meet the bar clean |
| median | own | p97.5 | 0.80 / 0.68 / 0.52 / 0.12 / 0.06 | 0.07 | 0.13 | does not meet the bar clean |
| median | own | p99 | 0.64 / 0.15 / 0.08 / 0.05 / 0.03 | 0.05 | 0.07 | does not meet the bar clean |
| median | own+pool | p90 | 0.87 / 0.87 / 0.87 / 0.86 / 0.84 | 0.12 | 0.44 | does not meet the bar clean |
| median | own+pool | p95 | 0.76 / 0.75 / 0.73 / 0.69 / 0.63 | 0.08 | 0.28 | does not meet the bar clean |
| median | own+pool | p97.5 | 0.63 / 0.63 / 0.62 / 0.58 / 0.45 | 0.06 | 0.18 | does not meet the bar clean |
| median | own+pool | p99 | 0.50 / 0.37 / 0.29 / 0.23 / 0.12 | 0.03 | 0.09 | does not meet the bar clean |
| median | pool | p90 | 0.87 | 0.12 | 0.43 | does not meet the bar clean |
| median | pool | p95 | 0.73 | 0.08 | 0.27 | does not meet the bar clean |
| median | pool | p97.5 | 0.63 | 0.06 | 0.18 | does not meet the bar clean |
| median | pool | p99 | 0.50 | 0.03 | 0.13 | does not meet the bar clean |
| median_l1 | own | p90 | 0.89 / 0.88 / 0.87 / 0.79 / 0.40 | 0.13 | 0.45 | does not meet the bar clean |
| median_l1 | own | p95 | 0.84 / 0.82 / 0.81 / 0.36 / 0.13 | 0.09 | 0.31 | does not meet the bar clean |
| median_l1 | own | p97.5 | 0.77 / 0.67 / 0.53 / 0.13 / 0.06 | 0.07 | 0.14 | does not meet the bar clean |
| median_l1 | own | p99 | 0.62 / 0.18 / 0.09 / 0.06 / 0.03 | 0.06 | 0.09 | does not meet the bar clean |
| median_l1 | own+pool | p90 | 0.86 / 0.86 / 0.85 / 0.85 / 0.83 | 0.12 | 0.43 | does not meet the bar clean |
| median_l1 | own+pool | p95 | 0.75 / 0.74 / 0.73 / 0.70 / 0.64 | 0.08 | 0.28 | does not meet the bar clean |
| median_l1 | own+pool | p97.5 | 0.64 / 0.63 / 0.63 / 0.59 / 0.50 | 0.06 | 0.19 | does not meet the bar clean |
| median_l1 | own+pool | p99 | 0.55 / 0.40 / 0.35 / 0.23 / 0.13 | 0.04 | 0.09 | does not meet the bar clean |
| median_l1 | pool | p90 | 0.85 | 0.12 | 0.42 | does not meet the bar clean |
| median_l1 | pool | p95 | 0.73 | 0.08 | 0.26 | does not meet the bar clean |
| median_l1 | pool | p97.5 | 0.63 | 0.06 | 0.19 | does not meet the bar clean |
| median_l1 | pool | p99 | 0.54 | 0.04 | 0.15 | does not meet the bar clean |
| zeuclid | own | p90 | 0.89 / 0.87 / 0.85 / 0.60 / 0.21 | 0.14 | 0.49 | does not meet the bar clean |
| zeuclid | own | p95 | 0.86 / 0.83 / 0.79 / 0.30 / 0.09 | 0.08 | 0.35 | does not meet the bar clean |
| zeuclid | own | p97.5 | 0.81 / 0.65 / 0.44 / 0.13 / 0.04 | 0.06 | 0.17 | does not meet the bar clean |
| zeuclid | own | p99 | 0.66 / 0.18 / 0.09 / 0.05 / 0.03 | 0.04 | 0.09 | does not meet the bar clean |
| zeuclid | own+pool | p90 | 0.86 / 0.85 / 0.84 / 0.83 / 0.79 | 0.11 | 0.48 | does not meet the bar clean |
| zeuclid | own+pool | p95 | 0.78 / 0.76 / 0.74 / 0.67 / 0.57 | 0.08 | 0.31 | does not meet the bar clean |
| zeuclid | own+pool | p97.5 | 0.66 / 0.64 / 0.62 / 0.56 / 0.41 | 0.06 | 0.21 | does not meet the bar clean |
| zeuclid | own+pool | p99 | 0.54 / 0.29 / 0.26 / 0.18 / 0.14 | 0.03 | 0.09 | does not meet the bar clean |
| zeuclid | pool | p90 | 0.85 | 0.11 | 0.48 | does not meet the bar clean |
| zeuclid | pool | p95 | 0.76 | 0.08 | 0.27 | does not meet the bar clean |
| zeuclid | pool | p97.5 | 0.65 | 0.06 | 0.21 | does not meet the bar clean |
| zeuclid | pool | p99 | 0.53 | 0.03 | 0.17 | does not meet the bar clean |
| zcount2 | own | p90 | 0.89 / 0.87 / 0.85 / 0.60 / 0.18 | 0.14 | 0.49 | does not meet the bar clean |
| zcount2 | own | p95 | 0.86 / 0.83 / 0.73 / 0.32 / 0.08 | 0.09 | 0.34 | does not meet the bar clean |
| zcount2 | own | p97.5 | 0.80 / 0.63 / 0.45 / 0.13 / 0.04 | 0.06 | 0.16 | does not meet the bar clean |
| zcount2 | own | p99 | 0.61 / 0.16 / 0.09 / 0.05 / 0.03 | 0.03 | 0.09 | does not meet the bar clean |
| zcount2 | own+pool | p90 | 0.86 / 0.86 / 0.86 / 0.84 / 0.79 | 0.11 | 0.48 | does not meet the bar clean |
| zcount2 | own+pool | p95 | 0.78 / 0.77 / 0.75 / 0.69 / 0.57 | 0.07 | 0.31 | does not meet the bar clean |
| zcount2 | own+pool | p97.5 | 0.66 / 0.64 / 0.63 / 0.55 / 0.42 | 0.06 | 0.18 | does not meet the bar clean |
| zcount2 | own+pool | p99 | 0.51 / 0.33 / 0.30 / 0.21 / 0.14 | 0.02 | 0.09 | does not meet the bar clean |
| zcount2 | pool | p90 | 0.86 | 0.11 | 0.48 | does not meet the bar clean |
| zcount2 | pool | p95 | 0.76 | 0.08 | 0.27 | does not meet the bar clean |
| zcount2 | pool | p97.5 | 0.65 | 0.06 | 0.21 | does not meet the bar clean |
| zcount2 | pool | p99 | 0.53 | 0.04 | 0.17 | does not meet the bar clean |
| zcount3 | own | p90 | 0.88 / 0.85 / 0.75 / 0.27 / 0.04 | 0.13 | 0.47 | does not meet the bar clean |
| zcount3 | own | p95 | 0.85 / 0.79 / 0.57 / 0.15 / 0.03 | 0.08 | 0.32 | does not meet the bar clean |
| zcount3 | own | p97.5 | 0.77 / 0.57 / 0.35 / 0.06 / 0.02 | 0.07 | 0.16 | does not meet the bar clean |
| zcount3 | own | p99 | 0.68 / 0.24 / 0.13 / 0.03 / 0.02 | 0.04 | 0.06 | does not meet the bar clean |
| zcount3 | own+pool | p90 | 0.86 / 0.84 / 0.82 / 0.78 / 0.63 | 0.12 | 0.49 | does not meet the bar clean |
| zcount3 | own+pool | p95 | 0.76 / 0.73 / 0.69 / 0.59 / 0.41 | 0.07 | 0.29 | does not meet the bar clean |
| zcount3 | own+pool | p97.5 | 0.64 / 0.61 / 0.57 / 0.45 / 0.24 | 0.06 | 0.22 | does not meet the bar clean |
| zcount3 | own+pool | p99 | 0.48 / 0.32 / 0.25 / 0.18 / 0.11 | 0.03 | 0.15 | does not meet the bar clean |
| zcount3 | pool | p90 | 0.83 | 0.12 | 0.42 | does not meet the bar clean |
| zcount3 | pool | p95 | 0.74 | 0.07 | 0.29 | does not meet the bar clean |
| zcount3 | pool | p97.5 | 0.62 | 0.06 | 0.23 | does not meet the bar clean |
| zcount3 | pool | p99 | 0.46 | 0.03 | 0.18 | does not meet the bar clean |
| zmax | own | p90 | 0.58 / 0.38 / 0.22 / 0.06 / 0.04 | 0.21 | 0.38 | does not meet the bar clean |
| zmax | own | p95 | 0.37 / 0.17 / 0.09 / 0.04 / 0.03 | 0.13 | 0.27 | does not meet the bar clean |
| zmax | own | p97.5 | 0.16 / 0.07 / 0.05 / 0.03 / 0.03 | 0.08 | 0.19 | does not meet the bar clean |
| zmax | own | p99 | 0.08 / 0.06 / 0.04 / 0.03 / 0.03 | 0.05 | 0.13 | does not meet the bar clean |
| zmax | own+pool | p90 | 0.46 / 0.42 / 0.38 / 0.27 / 0.15 | 0.19 | 0.41 | does not meet the bar clean |
| zmax | own+pool | p95 | 0.22 / 0.19 / 0.16 / 0.10 / 0.06 | 0.10 | 0.26 | does not meet the bar clean |
| zmax | own+pool | p97.5 | 0.09 / 0.07 / 0.06 / 0.04 / 0.03 | 0.05 | 0.21 | does not meet the bar clean |
| zmax | own+pool | p99 | 0.03 / 0.03 / 0.03 / 0.03 / 0.03 | 0.03 | 0.18 | does not meet the bar clean |
| zmax | pool | p90 | 0.42 | 0.19 | 0.42 | does not meet the bar clean |
| zmax | pool | p95 | 0.20 | 0.09 | 0.34 | does not meet the bar clean |
| zmax | pool | p97.5 | 0.07 | 0.05 | 0.22 | does not meet the bar clean |
| zmax | pool | p99 | 0.03 | 0.03 | 0.08 | does not meet the bar clean |
| cosine | own | p90 | 0.90 / 0.89 / 0.88 / 0.69 / 0.22 | 0.18 | 0.38 | does not meet the bar clean |
| cosine | own | p95 | 0.89 / 0.79 / 0.58 / 0.31 / 0.10 | 0.18 | 0.18 | does not meet the bar clean |
| cosine | own | p97.5 | 0.69 / 0.61 / 0.30 / 0.10 / 0.05 | 0.11 | 0.07 | does not meet the bar clean |
| cosine | own | p99 | 0.45 / 0.10 / 0.07 / 0.05 / 0.03 | 0.05 | 0.03 | does not meet the bar clean |
| cosine | own+pool | p90 | 0.89 / 0.88 / 0.87 / 0.84 / 0.78 | 0.18 | 0.35 | does not meet the bar clean |
| cosine | own+pool | p95 | 0.70 / 0.66 / 0.64 / 0.59 / 0.45 | 0.11 | 0.16 | does not meet the bar clean |
| cosine | own+pool | p97.5 | 0.49 / 0.44 / 0.42 / 0.35 / 0.23 | 0.07 | 0.07 | does not meet the bar clean |
| cosine | own+pool | p99 | 0.28 / 0.24 / 0.20 / 0.15 / 0.10 | 0.02 | 0.03 | does not meet the bar clean |
| cosine | pool | p90 | 0.86 | 0.13 | 0.36 | does not meet the bar clean |
| cosine | pool | p95 | 0.66 | 0.10 | 0.17 | does not meet the bar clean |
| cosine | pool | p97.5 | 0.44 | 0.07 | 0.04 | does not meet the bar clean |
| cosine | pool | p99 | 0.27 | 0.02 | 0.03 | does not meet the bar clean |
| pca10 | own | p90 | 0.87 / 0.08 / 0.03 / 0.03 / 0.03 | 0.23 | 0.42 | does not meet the bar clean |
| pca10 | own | p95 | 0.82 / 0.05 / 0.02 / 0.02 / 0.02 | 0.14 | 0.32 | does not meet the bar clean |
| pca10 | own | p97.5 | 0.80 / 0.03 / 0.02 / 0.02 / 0.02 | 0.06 | 0.22 | does not meet the bar clean |
| pca10 | own | p99 | 0.61 / 0.03 / 0.02 / 0.02 / 0.02 | 0.03 | 0.09 | does not meet the bar clean |
| pca10 | own+pool | p90 | 0.85 / 0.83 / 0.83 / 0.25 / 0.02 | 0.24 | 0.34 | does not meet the bar clean |
| pca10 | own+pool | p95 | 0.80 / 0.79 / 0.74 / 0.08 / 0.02 | 0.11 | 0.19 | does not meet the bar clean |
| pca10 | own+pool | p97.5 | 0.73 / 0.68 / 0.50 / 0.04 / 0.02 | 0.08 | 0.16 | does not meet the bar clean |
| pca10 | own+pool | p99 | 0.46 / 0.25 / 0.15 / 0.02 / 0.02 | 0.03 | 0.08 | does not meet the bar clean |
| pca10 | pool | p90 | 0.84 | 0.23 | 0.41 | does not meet the bar clean |
| pca10 | pool | p95 | 0.79 | 0.12 | 0.22 | does not meet the bar clean |
| pca10 | pool | p97.5 | 0.72 | 0.08 | 0.18 | does not meet the bar clean |
| pca10 | pool | p99 | 0.43 | 0.03 | 0.11 | does not meet the bar clean |
| pca50 | own | p90 | 0.90 / 0.03 / 0.03 / 0.03 / 0.03 | 0.14 | 0.35 | meets the bar up to 0% |
| pca50 | own | p95 | 0.87 / 0.03 / 0.03 / 0.03 / 0.03 | 0.10 | 0.21 | does not meet the bar clean |
| pca50 | own | p97.5 | 0.84 / 0.02 / 0.02 / 0.02 / 0.02 | 0.07 | 0.13 | does not meet the bar clean |
| pca50 | own | p99 | 0.82 / 0.02 / 0.02 / 0.02 / 0.02 | 0.07 | 0.07 | does not meet the bar clean |
| pca50 | own+pool | p90 | 0.88 / 0.32 / 0.03 / 0.03 / 0.03 | 0.17 | 0.31 | does not meet the bar clean |
| pca50 | own+pool | p95 | 0.84 / 0.08 / 0.03 / 0.02 / 0.02 | 0.09 | 0.18 | does not meet the bar clean |
| pca50 | own+pool | p97.5 | 0.80 / 0.07 / 0.02 / 0.02 / 0.02 | 0.07 | 0.13 | does not meet the bar clean |
| pca50 | own+pool | p99 | 0.68 / 0.03 / 0.02 / 0.02 / 0.02 | 0.03 | 0.06 | does not meet the bar clean |
| pca50 | pool | p90 | 0.88 | 0.18 | 0.25 | does not meet the bar clean |
| pca50 | pool | p95 | 0.82 | 0.10 | 0.20 | does not meet the bar clean |
| pca50 | pool | p97.5 | 0.77 | 0.07 | 0.15 | does not meet the bar clean |
| pca50 | pool | p99 | 0.65 | 0.04 | 0.07 | does not meet the bar clean |
| mahalanobis | own | p90 | 0.92 / 0.03 / 0.03 / 0.03 / 0.03 | 0.16 | 0.31 | does not meet the bar clean |
| mahalanobis | own | p95 | 0.87 / 0.03 / 0.03 / 0.03 / 0.02 | 0.07 | 0.19 | does not meet the bar clean |
| mahalanobis | own | p97.5 | 0.85 / 0.03 / 0.03 / 0.02 / 0.02 | 0.06 | 0.14 | does not meet the bar clean |
| mahalanobis | own | p99 | 0.80 / 0.03 / 0.02 / 0.02 / 0.02 | 0.06 | 0.09 | does not meet the bar clean |
| mahalanobis | own+pool | p90 | 0.93 / 0.07 / 0.05 / 0.04 / 0.04 | 0.19 | 0.25 | does not meet the bar clean |
| mahalanobis | own+pool | p95 | 0.91 / 0.04 / 0.04 / 0.03 / 0.03 | 0.08 | 0.14 | meets the bar up to 0% |
| mahalanobis | own+pool | p97.5 | 0.88 / 0.03 / 0.03 / 0.02 / 0.02 | 0.06 | 0.12 | does not meet the bar clean |
| mahalanobis | own+pool | p99 | 0.81 / 0.02 / 0.02 / 0.02 / 0.02 | 0.04 | 0.08 | does not meet the bar clean |
| mahalanobis | pool | p90 | 0.93 | 0.19 | 0.22 | does not meet the bar clean |
| mahalanobis | pool | p95 | 0.91 | 0.10 | 0.17 | meets the bar (no contamination possible) |
| mahalanobis | pool | p97.5 | 0.85 | 0.07 | 0.13 | does not meet the bar clean |
| mahalanobis | pool | p99 | 0.79 | 0.03 | 0.08 | does not meet the bar clean |
| knn1 | own | p90 | 0.84 / 0.03 / 0.03 / 0.02 / 0.02 | 0.17 | 0.35 | does not meet the bar clean |
| knn1 | own | p95 | 0.80 / 0.03 / 0.02 / 0.02 / 0.02 | 0.12 | 0.24 | does not meet the bar clean |
| knn1 | own | p97.5 | 0.73 / 0.02 / 0.02 / 0.02 / 0.02 | 0.05 | 0.22 | does not meet the bar clean |
| knn1 | own | p99 | 0.65 / 0.02 / 0.02 / 0.02 / 0.02 | 0.04 | 0.12 | does not meet the bar clean |
| knn1 | own+pool | p90 | 0.88 / 0.02 / 0.02 / 0.01 / 0.01 | 0.16 | 0.36 | does not meet the bar clean |
| knn1 | own+pool | p95 | 0.84 / 0.02 / 0.01 / 0.01 / 0.01 | 0.10 | 0.23 | does not meet the bar clean |
| knn1 | own+pool | p97.5 | 0.77 / 0.01 / 0.01 / 0.01 / 0.01 | 0.06 | 0.18 | does not meet the bar clean |
| knn1 | own+pool | p99 | 0.52 / 0.01 / 0.01 / 0.01 / 0.01 | 0.04 | 0.08 | does not meet the bar clean |
| knn1 | pool | p90 | 0.86 | 0.16 | 0.37 | does not meet the bar clean |
| knn1 | pool | p95 | 0.81 | 0.12 | 0.27 | does not meet the bar clean |
| knn1 | pool | p97.5 | 0.74 | 0.06 | 0.16 | does not meet the bar clean |
| knn1 | pool | p99 | 0.52 | 0.04 | 0.12 | does not meet the bar clean |
| knn5 | own | p90 | 0.83 / 0.03 / 0.03 / 0.02 / 0.02 | 0.19 | 0.33 | does not meet the bar clean |
| knn5 | own | p95 | 0.76 / 0.02 / 0.02 / 0.02 / 0.02 | 0.11 | 0.28 | does not meet the bar clean |
| knn5 | own | p97.5 | 0.72 / 0.02 / 0.02 / 0.02 / 0.02 | 0.06 | 0.17 | does not meet the bar clean |
| knn5 | own | p99 | 0.69 / 0.02 / 0.02 / 0.02 / 0.02 | 0.05 | 0.09 | does not meet the bar clean |
| knn5 | own+pool | p90 | 0.86 / 0.03 / 0.03 / 0.02 / 0.02 | 0.16 | 0.34 | does not meet the bar clean |
| knn5 | own+pool | p95 | 0.80 / 0.02 / 0.02 / 0.02 / 0.02 | 0.09 | 0.26 | does not meet the bar clean |
| knn5 | own+pool | p97.5 | 0.72 / 0.02 / 0.02 / 0.02 / 0.02 | 0.06 | 0.17 | does not meet the bar clean |
| knn5 | own+pool | p99 | 0.54 / 0.01 / 0.01 / 0.01 / 0.01 | 0.04 | 0.11 | does not meet the bar clean |
| knn5 | pool | p90 | 0.84 | 0.17 | 0.34 | does not meet the bar clean |
| knn5 | pool | p95 | 0.77 | 0.09 | 0.31 | does not meet the bar clean |
| knn5 | pool | p97.5 | 0.67 | 0.06 | 0.14 | does not meet the bar clean |
| knn5 | pool | p99 | 0.47 | 0.04 | 0.13 | does not meet the bar clean |
| knn10 | own | p90 | 0.79 / 0.03 / 0.03 / 0.02 / 0.02 | 0.20 | 0.34 | does not meet the bar clean |
| knn10 | own | p95 | 0.74 / 0.02 / 0.02 / 0.02 / 0.02 | 0.10 | 0.29 | does not meet the bar clean |
| knn10 | own | p97.5 | 0.69 / 0.02 / 0.02 / 0.02 / 0.02 | 0.08 | 0.18 | does not meet the bar clean |
| knn10 | own | p99 | 0.69 / 0.02 / 0.02 / 0.02 / 0.02 | 0.05 | 0.07 | does not meet the bar clean |
| knn10 | own+pool | p90 | 0.84 / 0.04 / 0.03 / 0.02 / 0.02 | 0.17 | 0.33 | does not meet the bar clean |
| knn10 | own+pool | p95 | 0.78 / 0.03 / 0.02 / 0.02 / 0.02 | 0.08 | 0.27 | does not meet the bar clean |
| knn10 | own+pool | p97.5 | 0.66 / 0.02 / 0.02 / 0.02 / 0.02 | 0.06 | 0.18 | does not meet the bar clean |
| knn10 | own+pool | p99 | 0.45 / 0.01 / 0.01 / 0.01 / 0.01 | 0.04 | 0.12 | does not meet the bar clean |
| knn10 | pool | p90 | 0.83 | 0.17 | 0.37 | does not meet the bar clean |
| knn10 | pool | p95 | 0.75 | 0.09 | 0.26 | does not meet the bar clean |
| knn10 | pool | p97.5 | 0.54 | 0.06 | 0.13 | does not meet the bar clean |
| knn10 | pool | p99 | 0.40 | 0.04 | 0.13 | does not meet the bar clean |
| iforest | own | p90 | 0.91 / 0.89 / 0.85 / 0.47 / 0.10 | 0.15 | 0.45 | meets the bar up to 0% |
| iforest | own | p95 | 0.87 / 0.81 / 0.71 / 0.23 / 0.06 | 0.12 | 0.34 | does not meet the bar clean |
| iforest | own | p97.5 | 0.80 / 0.56 / 0.40 / 0.12 / 0.04 | 0.08 | 0.24 | does not meet the bar clean |
| iforest | own | p99 | 0.42 / 0.22 / 0.15 / 0.06 / 0.02 | 0.04 | 0.12 | does not meet the bar clean |
| iforest | own+pool | p90 | 0.78 / 0.77 / 0.76 / 0.69 / 0.57 | 0.12 | 0.47 | does not meet the bar clean |
| iforest | own+pool | p95 | 0.56 / 0.55 / 0.54 / 0.47 / 0.34 | 0.08 | 0.26 | does not meet the bar clean |
| iforest | own+pool | p97.5 | 0.39 / 0.39 / 0.37 / 0.31 / 0.21 | 0.06 | 0.19 | does not meet the bar clean |
| iforest | own+pool | p99 | 0.25 / 0.25 / 0.24 / 0.19 / 0.11 | 0.04 | 0.09 | does not meet the bar clean |
| iforest | pool | p90 | 0.79 | 0.13 | 0.47 | does not meet the bar clean |
| iforest | pool | p95 | 0.64 | 0.09 | 0.36 | does not meet the bar clean |
| iforest | pool | p97.5 | 0.52 | 0.06 | 0.29 | does not meet the bar clean |
| iforest | pool | p99 | 0.23 | 0.04 | 0.11 | does not meet the bar clean |
