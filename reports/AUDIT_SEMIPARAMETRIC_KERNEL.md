# Full-grid semiparametric kernel validation

Decision rules fixed before the run.

## Locked decision

- PASS — lower median MAE than the shared-trend model on both primary targets
- FAIL — within 5% of random forest on both primary targets
- PASS — fewer direction reversals on the primary targets than the shared-trend model
- PASS — formation-Cs MAE and sign are no worse than the shared-trend model
- PASS — 60% and 80% selective prediction lower sign error on both primary targets

**Semiparametric-kernel status: FUTURE WORK.**

## Paired outer-B comparisons

| target | comparator | semi MAE | reference MAE | median change | wins | sign change | reversals semi/ref | p |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| heat of formation | hybrid | 0.2401 | 0.3146 | -19.6% | 45/52 | +0.037 | 6/9 | 6.31e-07 |
| heat of formation | node random forest | 0.2401 | 0.1874 | +20.8% | 11/52 | -0.016 | 6/5 | 0.000234 |
| heat of formation | node gradient boosting | 0.2401 | 0.2238 | +1.4% | 23/52 | -0.001 | 6/4 | 0.111 |
| magnetic moment | hybrid | 0.9331 | 0.9604 | -2.8% | 43/52 | +0.010 | 17/17 | 2.855e-06 |
| magnetic moment | node random forest | 0.9331 | 0.7095 | +33.5% | 3/52 | -0.084 | 17/12 | 1.05e-09 |
| magnetic moment | node gradient boosting | 0.9331 | 0.8421 | +8.3% | 4/52 | -0.020 | 17/14 | 5.016e-09 |
| volume per fu | hybrid | 3.1544 | 4.3192 | -16.0% | 39/52 | +0.012 | 1/4 | 9.601e-06 |
| volume per fu | node random forest | 3.1544 | 3.6275 | -2.2% | 29/52 | +0.002 | 1/7 | 0.1306 |
| volume per fu | node gradient boosting | 3.1544 | 3.1785 | +1.4% | 24/52 | +0.000 | 1/4 | 0.4443 |

## Risk-coverage summary

| target | coverage | sign error | MAE | Brier | 90% interval coverage |
|---|---:|---:|---:|---:|---:|
| heat of formation | 60% | 0.038 | 0.2536 | 0.040 | 0.932 |
| heat of formation | 80% | 0.075 | 0.2444 | 0.072 | 0.938 |
| heat of formation | 100% | 0.135 | 0.2401 | 0.104 | 0.938 |
| magnetic moment | 60% | 0.247 | 1.0573 | 0.179 | 0.909 |
| magnetic moment | 80% | 0.290 | 0.9894 | 0.195 | 0.915 |
| magnetic moment | 100% | 0.325 | 0.9331 | 0.205 | 0.921 |

## As/Cs cells

| target | level | model | MAE | sign | B-edge MAE | B-edge sign | margin |
|---|---|---|---:|---:|---:|---:|---:|
| heat of formation | As | hybrid | 0.2023 | 0.863 | 0.2062 | 0.871 | -0.2280 |
| heat of formation | As | node random forest | 0.1885 | 0.870 | 0.1961 | 0.866 | -0.1471 |
| heat of formation | As | node gradient boosting | 0.2003 | 0.870 | 0.2124 | 0.868 | -0.2013 |
| heat of formation | As | semiparametric kernel | 0.1969 | 0.871 | 0.2053 | 0.875 | -0.2171 |
| heat of formation | Cs | hybrid | 0.4075 | 0.900 | 0.3893 | 0.973 | +1.5063 |
| heat of formation | Cs | node random forest | 0.3699 | 0.899 | 0.3220 | 0.977 | +1.5791 |
| heat of formation | Cs | node gradient boosting | 0.3863 | 0.897 | 0.3561 | 0.972 | +1.6163 |
| heat of formation | Cs | semiparametric kernel | 0.3809 | 0.903 | 0.3605 | 0.969 | +1.6687 |
| magnetic moment | As | hybrid | 0.8463 | 0.655 | 0.8797 | 0.616 | +0.0014 |
| magnetic moment | As | node random forest | 0.6581 | 0.723 | 0.5928 | 0.729 | -0.2160 |
| magnetic moment | As | node gradient boosting | 0.7025 | 0.709 | 0.7200 | 0.677 | -0.3410 |
| magnetic moment | As | semiparametric kernel | 0.7852 | 0.674 | 0.8182 | 0.637 | -0.1577 |
| magnetic moment | Cs | hybrid | 1.2180 | 0.578 | 1.2052 | 0.579 | -0.0036 |
| magnetic moment | Cs | node random forest | 0.9504 | 0.687 | 0.8581 | 0.708 | -0.0913 |
| magnetic moment | Cs | node gradient boosting | 0.9907 | 0.637 | 0.9696 | 0.640 | -0.1199 |
| magnetic moment | Cs | semiparametric kernel | 1.2162 | 0.574 | 1.2162 | 0.571 | +0.1477 |
| volume per fu | As | hybrid | 5.7334 | 0.810 | 6.8718 | 0.805 | -12.2521 |
| volume per fu | As | node random forest | 6.9341 | 0.763 | 8.6919 | 0.748 | -15.0232 |
| volume per fu | As | node gradient boosting | 4.5310 | 0.846 | 4.9820 | 0.866 | -9.2997 |
| volume per fu | As | semiparametric kernel | 5.2509 | 0.821 | 6.3203 | 0.818 | -12.3028 |
| volume per fu | Cs | hybrid | 13.9900 | 0.872 | 16.8012 | 0.947 | +45.4441 |
| volume per fu | Cs | node random forest | 7.2745 | 0.893 | 7.0105 | 0.972 | +33.1043 |
| volume per fu | Cs | node gradient boosting | 7.0996 | 0.883 | 7.0816 | 0.968 | +33.6193 |
| volume per fu | Cs | semiparametric kernel | 10.4421 | 0.867 | 10.8055 | 0.953 | +34.1611 |

## Formation-Cs support diagnosis

| kNN distance | kernel leverage | extrapolation instability | convex-hull distance | true/train/shared-trend/RF/HGB/semi element means |
|---:|---:|---:|---:|---|
| 1.774 | 0.993 | 0.029 | 1.719 | 2.919 / 1.443 / 2.949 / 3.021 / 3.058 / 3.111 |

The support diagnostics test shrinkage and extrapolation explanations only. They do not establish a caesium or arsenic mechanism.

