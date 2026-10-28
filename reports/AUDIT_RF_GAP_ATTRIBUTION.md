# RF--semiparametric gap attribution

Deterministic diagnostics computed after the results.

## Oracle comparison

| target | semi MAE | RF MAE | projected-RF MAE | true-triple oracle MAE | RF gaps > 0 | projected RF advantage retained | triple gap closed |
|---|---:|---:|---:|---:|---:|---:|---:|
| heat of formation | 0.2401 | 0.1874 | 0.2206 | 0.1869 | 41/52 | 16.1% | 94.4% |
| magnetic moment | 0.9331 | 0.7095 | 0.8797 | 0.6443 | 49/52 | 9.9% | 112.6% |
| volume per fu | 3.1544 | 3.6275 | 3.8890 | 2.5919 | 23/52 | 31.8% | 55.2% |

Fractions use only outer levels where RF beats the semiparametric model and are not clipped. `Projected RF advantage retained` measures how much of RF's level-wise MAE advantage survives removal of its predicted triple field. `Triple gap closed` is the unusable truth oracle's level-wise upper-bound diagnostic.

## Random forest minus semiparametric model, left-out element

Pooled share of squared node-prediction difference by orthogonal block, giving every outer level the same number of cells. The four entries sum to 100%:

| target | element mean | A conditional | X conditional | A-by-X residual |
|---|---:|---:|---:|---:|
| heat of formation | 54.0% | 11.7% | 15.1% | 19.2% |
| magnetic moment | 18.8% | 18.9% | 9.4% | 52.9% |
| volume per fu | 79.2% | 3.8% | 10.2% | 6.8% |

## As/Cs weighting and boundary check

| target | level | balanced B edges | true triple mean | margin weighting error | semi margin | triple-oracle margin |
|---|---|---|---:|---:|---:|---:|
| heat of formation | As | yes (18564/18564) | -7.40e-17 | +2.50e-16 | -0.2171 | -0.2171 |
| heat of formation | Cs | yes (18564/18564) | -2.10e-17 | +0.00e+00 | +1.6687 | +1.6687 |
| magnetic moment | As | yes (18564/18564) | +1.83e-18 | +1.11e-16 | -0.1577 | -0.1577 |
| magnetic moment | Cs | yes (18564/18564) | +1.36e-16 | -1.67e-16 | +0.1477 | +0.1477 |
| volume per fu | As | yes (18564/18564) | +7.27e-15 | -4.44e-15 | -12.3028 | -12.3028 |
| volume per fu | Cs | yes (18564/18564) | +1.78e-14 | -3.55e-15 | +34.1611 | +34.1611 |

A balanced held-B direction averages uniformly over A, X and partner B. Therefore a centred held triple field cannot change that direction; any As/Cs mean-direction flip that survives the numerical checks belongs to B-main transfer, not the omitted triple field.

## Interpretation rule

Read the table target by target. The projection and truth-oracle columns diagnose different restrictions and must not be described as deployable models.

