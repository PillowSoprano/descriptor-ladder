# Matched regularisation baselines for the coverage claim

The primary score uses only substitution edges whose two endpoint compositions were hidden. Every tuned arm receives three identical training-only inner masks and nine candidate configurations. One outer mask is the independent paired unit.

## Median two-hidden-endpoint MAE and chemically structured gain

| property | coverage | free | ordinary ridge | block ridge | descriptor metric | shuffled metric | random metric | descriptor vs ordinary | descriptor vs block |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| heat of formation | 5% | 0.3855 | 0.3904 | 0.3564 | 0.3525 | 0.3811 | 0.3639 | +9.7% | +1.1% |
| heat of formation | 10% | 0.3485 | 0.3491 | 0.3314 | 0.3088 | 0.3401 | 0.3397 | +11.5% | +6.8% |
| heat of formation | 15% | 0.3603 | 0.3183 | 0.3194 | 0.2915 | 0.3216 | 0.3180 | +8.4% | +8.7% |
| heat of formation | 20% | 0.4810 | 0.3041 | 0.3137 | 0.2847 | 0.3019 | 0.3029 | +6.4% | +9.2% |
| heat of formation | 30% | 0.3932 | 0.2874 | 0.3089 | 0.2765 | 0.2858 | 0.2882 | +3.8% | +10.5% |
| heat of formation | 40% | 0.3276 | 0.2707 | 0.2890 | 0.2645 | 0.2702 | 0.2738 | +2.3% | +8.5% |
| heat of formation | 50% | 0.2952 | 0.2573 | 0.2674 | 0.2531 | 0.2586 | 0.2584 | +1.6% | +5.4% |
| heat of formation | 70% | 0.2436 | 0.2246 | 0.2339 | 0.2261 | 0.2259 | 0.2251 | -0.7% | +3.3% |
| heat of formation | 90% | 0.2347 | 0.2310 | 0.2328 | 0.2348 | 0.2307 | 0.2300 | -1.7% | -0.9% |
| volume per fu | 5% | 7.9583 | 8.0556 | 6.1322 | 6.0776 | 6.9501 | 7.1787 | +24.6% | +0.9% |
| volume per fu | 10% | 6.5829 | 6.5813 | 5.5181 | 5.1789 | 6.2045 | 5.8809 | +21.3% | +6.1% |
| volume per fu | 15% | 6.3204 | 5.9420 | 5.2483 | 4.8863 | 5.8410 | 5.1972 | +17.8% | +6.9% |
| volume per fu | 20% | 8.6157 | 5.9510 | 5.2259 | 4.8081 | 5.0229 | 4.8685 | +19.2% | +8.0% |
| volume per fu | 30% | 7.2739 | 4.6894 | 5.0713 | 4.5703 | 4.6697 | 4.6003 | +2.5% | +9.9% |
| volume per fu | 40% | 5.2364 | 4.3265 | 4.5891 | 4.2941 | 4.3182 | 4.2755 | +0.7% | +6.4% |
| volume per fu | 50% | 4.7976 | 4.2406 | 4.5030 | 4.1665 | 4.2900 | 4.2479 | +1.7% | +7.5% |
| volume per fu | 70% | 4.1793 | 3.9512 | 4.0634 | 3.9995 | 3.9601 | 3.9347 | -1.2% | +1.6% |
| volume per fu | 90% | 3.7361 | 3.5939 | 3.6833 | 3.6148 | 3.6144 | 3.6203 | -0.6% | +1.9% |
| magnetic moment | 5% | 1.1593 | 1.1338 | 1.1378 | 1.1290 | 1.1322 | 1.1275 | +0.4% | +0.8% |
| magnetic moment | 10% | 1.2042 | 1.1168 | 1.1376 | 1.0985 | 1.1124 | 1.1032 | +1.6% | +3.4% |
| magnetic moment | 15% | 1.3686 | 1.0817 | 1.1226 | 1.0674 | 1.0925 | 1.0704 | +1.3% | +4.9% |
| magnetic moment | 20% | 2.2844 | 1.0665 | 1.0988 | 1.0454 | 1.0641 | 1.0548 | +2.0% | +4.9% |
| magnetic moment | 30% | 1.8470 | 1.0465 | 1.0778 | 1.0290 | 1.0495 | 1.0439 | +1.7% | +4.5% |
| magnetic moment | 40% | 1.3941 | 1.0227 | 1.0635 | 1.0066 | 1.0270 | 1.0239 | +1.6% | +5.3% |
| magnetic moment | 50% | 1.2318 | 1.0214 | 1.0702 | 1.0065 | 1.0219 | 1.0197 | +1.5% | +6.0% |
| magnetic moment | 70% | 1.1119 | 1.0112 | 1.0579 | 0.9948 | 1.0093 | 1.0109 | +1.6% | +6.0% |
| magnetic moment | 90% | 1.0441 | 0.9967 | 1.0651 | 0.9924 | 1.0008 | 0.9924 | +0.4% | +6.8% |

## Paired summary

| property | median descriptor-vs-ordinary gain | wins | median descriptor-vs-block gain | wins |
|---|---:|---:|---:|---:|
| heat of formation | +2.96% | 23/27 | +7.08% | 26/27 |
| magnetic moment | +1.50% | 77/90 | +5.08% | 85/90 |
| volume per fu | +2.21% | 23/27 | +6.65% | 25/27 |

## Endpoint-estimand audit

The median change from the historical touching-edge score to the primary two-hidden-endpoint score is +4.4%.

Runtime: 2.4 minutes.
