# Method and data attribution

This repository contains the MECC estimator, its numerical routines, and a standalone
implementation of the same-report comparisons described in the accompanying paper.

- Optimized unary encoding: T. Wang, J. Blocki, N. Li and S. Jha,
  “Locally Differentially Private Protocols for Frequency Estimation,” USENIX Security, 2017.
- Calibrate: J. Jia and N. Z. Gong, “Calibrate: Frequency Estimation and Heavy Hitter
  Identification with Local Differential Privacy via Incorporating Prior Knowledge,”
  IEEE INFOCOM, 2019. The implemented finite, zero-inclusive shifted power-law prior
  and final simplex projection are the adaptations specified in the paper.
- Full-output likelihood estimation: M. Lopuhaä-Zwakenberg, Z. Li, B. Škorić and N. Li,
  “Improving Frequency Estimation under Local Differential Privacy,” 2019/2020 preprint.
- MovieLens: F. M. Harper and J. A. Konstan, “The MovieLens Datasets: History and Context,”
  ACM Transactions on Interactive Intelligent Systems, 2015; GroupLens Research.
- Kosarak clickstream: Ferenc Bodon, FIMI Dataset Repository.

No third-party framework source or dataset archive is bundled. NumPy, SciPy, and
Matplotlib are installed as dependencies under their own licenses. Dataset downloads
remain subject to their providers' terms. A project-wide code license has not been
specified in this anonymous release; this notice does not grant additional rights.
