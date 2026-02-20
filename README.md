# Harder, shorter, sharper, forward: A comparison of women's and men's elite football gameplay (2020-2025)

Data, code, and materials for "[Harder, shorter, sharper, forward: A comparison of women's and men's elite football gameplay (2020-2025)](https://arxiv.org/abs/2506.22119)"

---

## Data

Although the data used in this study are not publicly available (provided under license from [Hudl StatsBomb](https://statsbomb.com/)), an equivalent surrogate dataset (same provider and same format) can be found in the [StatsBomb open-data repository](https://github.com/statsbomb/open-data).

---

## Code

| Script | Description |
|--------|-------------|
| [`pitch_network_construction.py`](./pitch_passing_network/pitch_network_construction.py) | Builds directed weighted pitch-passing networks from match event data using a 10×5 spatial grid |
| [`network_metrics.py`](./pitch_passing_network/network_metrics.py) | Computes network-level metrics: outreach, maximum eigenvalue, average shortest path length |
| [`kpis.py`](./kpis/kpis.py) | Computes match-level KPIs: pass accuracy, passes per possession, passes under pressure, vertical play, and more |


## Acknowledgements

This work is the output of the [Complexity72h](https://www.complexity72h.com) workshop, held at Universidad Carlos III de Madrid, Spain, 23–27 June 2025.

---

## Citation

If you use this code or findings from this paper in your work, please cite:

```bibtex
@article{carstens2025harder,
  title     = {Harder, shorter, sharper, forward: A comparison of women's and men's elite football gameplay (2020-2025)},
  author    = {Carstens, Rebecca and Deshpande, Raj and Esteve, Pau and Fidelibus, Nicol\`o and Linde Neven, Sara and Ottow, Ramona and Lokamruth, K. R. and Rodr\'iguez-S\'anchez, Paula and Santagata, Luca and Buld\'u, Javier M. and Klein, Brennan and Torricelli, Maddalena},
  journal   = {arXiv preprint arXiv:2506.22119},
  year      = {2025}
}
```
