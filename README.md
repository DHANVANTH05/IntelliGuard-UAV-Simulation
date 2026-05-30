# IntelliGuard: Secure and Energy-Efficient Fog-Integrated UAV Surveillance

**Paper:** A Secure and Energy-Efficient Fog-Integrated UAV Surveillance Framework Using IntelliGuard and DPAFIO  
**Journal:** Discover Computing, Springer Nature  
**Authors:** Dhanvanth Kumar Gude, Vamshi Krishna Raavi, Mohit Lalit, Anurag Jain, Bhupesh Kumar Dewangan

---

## About

This repository contains the complete simulation source code for the IntelliGuard framework. Running the file reproduces the key results reported in the article (Table 4).

---

## Requirements

- Python ≥ 3.8
- Standard library only (no external packages needed)

---

## Usage

```bash
python IntelliGuard_Simulation.py
```

---

## Expected Results
Mean Energy Consumption  -  0.8133 ± 0.0263 J/round
Average E2E Latency - 10.50 ms 
Attack Detection Accuracy - 99.97 ± 0.01 %
DPAFIO Convergence Fitness - 0.97533 ± 0.00055
Packet Reduction Rate - 94.99 ± 0.014 %

---

## Notes

- All results are computed from genuine simulation (no hardcoded values)
- 30 independent runs × 50 rounds with 95% CI via t-distribution
- Seeds are fixed per run for full reproducibility
