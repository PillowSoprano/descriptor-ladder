<!-- GENERATED: everything above the ANALYSIS marker is written by the script named in
     this file's first paragraph and is overwritten on every rerun. -->

# As/Cs mechanism audit

This audit separates localisation, magnetic-target construction and cross-target co-location. It is diagnostic: none of the comparisons assigns an elemental mechanism.

## Site localisation

| target | level | model | A sign | B sign | X sign | worst site |
|---|---|---|---:|---:|---:|---|
| heat of formation | As | direct | 0.770 | 0.312 | 0.608 | B |
| heat of formation | As | metric | 0.846 | 0.765 | 0.698 | X |
| heat of formation | Cs | direct | 0.745 | 0.973 | 0.667 | X |
| heat of formation | Cs | metric | 0.788 | 0.968 | 0.608 | X |
| volume per fu | As | direct | 0.732 | 0.779 | 0.816 | A |
| volume per fu | As | metric | 0.772 | 0.837 | 0.724 | X |
| volume per fu | Cs | direct | 0.706 | 0.924 | 0.857 | A |
| volume per fu | Cs | metric | 0.728 | 0.944 | 0.861 | A |
| magnetic moment | As | direct | 0.511 | 0.483 | 0.646 | B |
| magnetic moment | As | metric | 0.582 | 0.503 | 0.636 | B |
| magnetic moment | Cs | direct | 0.545 | 0.625 | 0.526 | X |
| magnetic moment | Cs | metric | 0.560 | 0.625 | 0.422 | X |
| magnetic moment signed | As | direct | 0.480 | 0.505 | 0.667 | A |
| magnetic moment signed | As | metric | 0.538 | 0.517 | 0.639 | B |
| magnetic moment signed | Cs | direct | 0.482 | 0.642 | 0.400 | X |
| magnetic moment signed | Cs | metric | 0.551 | 0.666 | 0.489 | X |

## Magnetic folding control

| level | model | magnitude sign | signed-source sign | change |
|---|---|---:|---:|---:|
| As | direct | 0.513 | 0.519 | +0.006 |
| As | metric | 0.544 | 0.540 | -0.004 |
| Cs | direct | 0.589 | 0.566 | -0.024 |
| Cs | metric | 0.580 | 0.611 | +0.031 |

The signed-source control is not a physical replacement for moment magnitude. A large improvement would implicate the absolute-value fold; no improvement would rule out that simple explanation.

## Cross-target co-location

| level | model | targets | failure Jaccard | error-rank rho |
|---|---|---|---:|---:|
| As | direct | heat_of_formation / magnetic_moment | 0.350 | +0.092 |
| As | direct | heat_of_formation / magnetic_moment_signed | 0.358 | +0.097 |
| As | direct | magnetic_moment / magnetic_moment_signed | 0.492 | +0.723 |
| As | metric | heat_of_formation / magnetic_moment | 0.199 | +0.212 |
| As | metric | heat_of_formation / magnetic_moment_signed | 0.191 | +0.172 |
| As | metric | magnetic_moment / magnetic_moment_signed | 0.511 | +0.791 |
| Cs | direct | heat_of_formation / magnetic_moment | 0.123 | +0.019 |
| Cs | direct | heat_of_formation / magnetic_moment_signed | 0.142 | -0.023 |
| Cs | direct | magnetic_moment / magnetic_moment_signed | 0.457 | +0.739 |
| Cs | metric | heat_of_formation / magnetic_moment | 0.124 | +0.044 |
| Cs | metric | heat_of_formation / magnetic_moment_signed | 0.118 | -0.001 |
| Cs | metric | magnetic_moment / magnetic_moment_signed | 0.415 | +0.672 |

## Most fragile contexts

Direct-model groups with at least 14 edges, ranked within each target and level.

| target | level | site | context | n | sign | median |delta| |
|---|---|---|---|---:|---:|---:|
| heat of formation | As | B | O3 | 182 | 0.206 | 0.570 |
| heat of formation | As | X | N3->O2N | 14 | 0.214 | 0.560 |
| heat of formation | As | X | N3->ON2 | 14 | 0.214 | 0.510 |
| heat of formation | Cs | X | O2N->O2S | 14 | 0.000 | 0.630 |
| heat of formation | Cs | X | O2F->O2S | 14 | 0.143 | 0.540 |
| heat of formation | Cs | X | O2S->O3 | 14 | 0.143 | 0.590 |
| volume per fu | As | X | O3->OFN | 14 | 0.000 | 7.569 |
| volume per fu | As | X | O2N->ON2 | 14 | 0.286 | 1.558 |
| volume per fu | As | X | N3->O2N | 14 | 0.429 | 1.819 |
| volume per fu | Cs | X | N3->ON2 | 14 | 0.429 | 4.137 |
| volume per fu | Cs | X | O2F->OFN | 14 | 0.429 | 5.037 |
| volume per fu | Cs | X | O2N->O3 | 14 | 0.429 | 4.720 |
| magnetic moment | As | X | N3->O2S | 14 | 0.214 | 1.213 |
| magnetic moment | As | X | O2N->O2S | 14 | 0.214 | 0.525 |
| magnetic moment | As | X | O2S->ON2 | 14 | 0.357 | 0.029 |
| magnetic moment | Cs | X | N3->ON2 | 14 | 0.286 | 0.683 |
| magnetic moment | Cs | X | O2F->O2N | 14 | 0.286 | 0.363 |
| magnetic moment | Cs | A | N3 | 91 | 0.297 | 1.483 |
| magnetic moment signed | As | X | O2F->O2S | 14 | 0.214 | 0.001 |
| magnetic moment signed | As | A | O2F | 91 | 0.407 | 0.065 |
| magnetic moment signed | As | A | O2S | 91 | 0.407 | 0.497 |
| magnetic moment signed | Cs | X | N3->O2N | 14 | 0.071 | 1.301 |
| magnetic moment signed | Cs | X | N3->O3 | 14 | 0.154 | 1.144 |
| magnetic moment signed | Cs | X | N3->OFN | 14 | 0.154 | 1.227 |

## B-substitution partner decomposition

Direct-model B edges, averaged over all 14 A levels and seven anions.

| target | held level | partner | n | sign | mean true delta | mean predicted delta |
|---|---|---|---:|---:|---:|---:|
| heat of formation | As | Au | 98 | 0.031 | +0.626 | -0.517 |
| heat of formation | As | Be | 98 | 0.062 | +0.400 | -0.597 |
| heat of formation | As | Cu | 98 | 0.092 | +0.356 | -0.581 |
| heat of formation | As | Ag | 98 | 0.122 | -0.680 | +0.296 |
| heat of formation | As | Ca | 98 | 0.124 | +0.661 | -0.413 |
| heat of formation | As | B | 98 | 0.155 | +0.342 | -0.948 |
| heat of formation | As | Bi | 98 | 0.206 | +0.185 | -0.830 |
| heat of formation | As | Cd | 98 | 0.214 | +0.642 | -0.383 |
| heat of formation | As | Co | 98 | 0.370 | -0.008 | -1.088 |
| heat of formation | As | Ba | 98 | 0.429 | +1.043 | -0.150 |
| heat of formation | As | Al | 98 | 0.532 | +0.034 | +0.912 |
| heat of formation | As | Cs | 98 | 0.816 | +1.607 | +0.611 |
| heat of formation | As | Cr | 98 | 0.918 | -0.373 | -1.306 |
| heat of formation | Cs | Ba | 98 | 0.837 | +0.563 | +1.355 |
| heat of formation | Cs | Ag | 98 | 0.949 | +0.926 | +1.582 |
| heat of formation | Cs | Cd | 98 | 0.949 | +0.965 | +1.699 |
| heat of formation | Cs | B | 98 | 0.969 | +1.264 | +2.183 |
| heat of formation | Cs | Ca | 98 | 0.969 | +0.946 | +1.754 |
| heat of formation | Cs | Be | 98 | 0.980 | +1.207 | +1.816 |
| heat of formation | Cs | Au | 98 | 0.990 | +0.980 | +1.817 |
| heat of formation | Cs | Al | 98 | 1.000 | +1.640 | +2.340 |
| heat of formation | Cs | As | 98 | 1.000 | +1.607 | +2.302 |
| heat of formation | Cs | Bi | 98 | 1.000 | +1.422 | +2.062 |
| heat of formation | Cs | Co | 98 | 1.000 | +1.614 | +2.442 |
| heat of formation | Cs | Cr | 98 | 1.000 | +1.980 | +2.524 |
| heat of formation | Cs | Cu | 98 | 1.000 | -1.250 | -1.950 |
| volume per fu | As | Au | 98 | 0.378 | +8.932 | -4.862 |
| volume per fu | As | Ag | 98 | 0.480 | -7.948 | +4.354 |
| volume per fu | As | Cd | 98 | 0.582 | +14.105 | +2.514 |
| volume per fu | As | Cu | 98 | 0.694 | -4.961 | -14.464 |
| volume per fu | As | Ca | 98 | 0.714 | +20.004 | +5.163 |
| volume per fu | As | Bi | 98 | 0.724 | +20.881 | +7.916 |
| volume per fu | As | Al | 98 | 0.837 | +7.543 | +14.285 |
| volume per fu | As | Cr | 98 | 0.898 | -6.408 | -17.509 |
| volume per fu | As | Co | 98 | 0.908 | -7.895 | -19.940 |
| volume per fu | As | Cs | 98 | 0.949 | +35.339 | +24.723 |
| volume per fu | As | Ba | 98 | 0.969 | +45.529 | +30.187 |
| volume per fu | As | Be | 98 | 0.990 | -15.763 | -26.610 |
| volume per fu | As | B | 98 | 1.000 | -17.554 | -36.640 |
| volume per fu | Cs | Ba | 98 | 0.133 | -10.190 | +27.162 |
| volume per fu | Cs | Bi | 98 | 0.949 | +14.458 | +49.903 |
| volume per fu | Cs | Ca | 98 | 0.949 | +15.335 | +53.221 |
| volume per fu | Cs | Cd | 98 | 0.980 | +21.234 | +57.673 |
| volume per fu | Cs | Ag | 98 | 1.000 | +27.391 | +61.960 |
| volume per fu | Cs | Al | 98 | 1.000 | +42.881 | +77.095 |
| volume per fu | Cs | As | 98 | 1.000 | +35.339 | +71.123 |
| volume per fu | Cs | Au | 98 | 1.000 | +26.406 | +63.591 |
| volume per fu | Cs | B | 98 | 1.000 | +52.893 | +92.781 |
| volume per fu | Cs | Be | 98 | 1.000 | +51.102 | +85.248 |
| volume per fu | Cs | Co | 98 | 1.000 | +43.233 | +80.698 |
| volume per fu | Cs | Cr | 98 | 1.000 | +41.747 | +76.468 |
| volume per fu | Cs | Cu | 98 | 1.000 | -40.300 | -75.806 |
| magnetic moment | As | Al | 98 | 0.378 | -0.495 | +0.233 |
| magnetic moment | As | Ca | 98 | 0.392 | -0.096 | -0.264 |
| magnetic moment | As | Cu | 98 | 0.418 | +0.396 | +0.021 |
| magnetic moment | As | B | 98 | 0.429 | -0.066 | -0.095 |
| magnetic moment | As | Au | 98 | 0.449 | +0.289 | -0.219 |
| magnetic moment | As | Be | 98 | 0.449 | +0.528 | -0.031 |
| magnetic moment | As | Cd | 98 | 0.459 | +0.641 | +0.209 |
| magnetic moment | As | Bi | 98 | 0.469 | +0.091 | -0.186 |
| magnetic moment | As | Cs | 98 | 0.485 | -0.089 | -0.515 |
| magnetic moment | As | Ag | 98 | 0.490 | -0.280 | -0.040 |
| magnetic moment | As | Ba | 98 | 0.602 | +0.187 | -0.154 |
| magnetic moment | As | Cr | 98 | 0.622 | +0.740 | +0.470 |
| magnetic moment | As | Co | 98 | 0.633 | +1.623 | +1.149 |
| magnetic moment | Cs | Ca | 98 | 0.444 | +0.007 | -2.469 |
| magnetic moment | Cs | As | 98 | 0.495 | -0.089 | -2.420 |
| magnetic moment | Cs | Ba | 98 | 0.537 | -0.276 | -2.689 |
| magnetic moment | Cs | Bi | 98 | 0.546 | -0.180 | -2.660 |
| magnetic moment | Cs | B | 98 | 0.582 | -0.023 | -2.580 |
| magnetic moment | Cs | Au | 98 | 0.604 | -0.378 | -2.610 |
| magnetic moment | Cs | Ag | 98 | 0.621 | -0.369 | -2.588 |
| magnetic moment | Cs | Cu | 98 | 0.632 | +0.485 | +2.791 |
| magnetic moment | Cs | Be | 98 | 0.646 | -0.617 | -2.918 |
| magnetic moment | Cs | Al | 98 | 0.680 | -0.584 | -2.598 |
| magnetic moment | Cs | Cd | 98 | 0.684 | -0.730 | -3.157 |
| magnetic moment | Cs | Cr | 98 | 0.771 | -0.829 | -3.449 |
| magnetic moment | Cs | Co | 98 | 0.835 | -1.712 | -4.095 |
| magnetic moment signed | As | Al | 98 | 0.378 | -0.233 | +0.694 |
| magnetic moment signed | As | Cd | 98 | 0.398 | +0.208 | -0.457 |
| magnetic moment signed | As | Cu | 98 | 0.429 | +0.014 | -0.582 |
| magnetic moment signed | As | Bi | 98 | 0.449 | -0.242 | -0.741 |
| magnetic moment signed | As | B | 98 | 0.459 | -0.228 | -0.554 |
| magnetic moment signed | As | Ag | 98 | 0.469 | +0.163 | +0.607 |
| magnetic moment signed | As | Au | 98 | 0.480 | -0.081 | -0.842 |
| magnetic moment signed | As | Ba | 98 | 0.480 | -0.141 | -0.732 |
| magnetic moment signed | As | Be | 98 | 0.500 | +0.118 | -0.638 |
| magnetic moment signed | As | Ca | 98 | 0.567 | -0.337 | -0.768 |
| magnetic moment signed | As | Cs | 98 | 0.567 | -0.356 | -1.015 |
| magnetic moment signed | As | Co | 98 | 0.694 | +1.540 | +0.822 |
| magnetic moment signed | As | Cr | 98 | 0.694 | +0.496 | +0.019 |
| magnetic moment signed | Cs | Ca | 98 | 0.481 | -0.019 | -2.352 |
| magnetic moment signed | Cs | Ba | 98 | 0.550 | -0.215 | -2.527 |
| magnetic moment signed | Cs | As | 98 | 0.577 | -0.356 | -2.611 |
| magnetic moment signed | Cs | Bi | 98 | 0.588 | -0.114 | -2.518 |
| magnetic moment signed | Cs | Ag | 98 | 0.600 | -0.193 | -2.366 |
| magnetic moment signed | Cs | Be | 98 | 0.615 | -0.474 | -2.726 |
| magnetic moment signed | Cs | Cd | 98 | 0.621 | -0.564 | -2.886 |
| magnetic moment signed | Cs | Cu | 98 | 0.632 | +0.369 | +2.605 |
| magnetic moment signed | Cs | B | 98 | 0.643 | -0.128 | -2.529 |
| magnetic moment signed | Cs | Au | 98 | 0.667 | -0.275 | -2.385 |
| magnetic moment signed | Cs | Cr | 98 | 0.740 | -0.852 | -3.416 |
| magnetic moment signed | Cs | Al | 98 | 0.742 | -0.589 | -2.541 |
| magnetic moment signed | Cs | Co | 98 | 0.845 | -1.896 | -4.153 |

## Model-class control

| target | level | model | MAE | sign |
|---|---|---|---:|---:|
| heat of formation | As | node linear | 0.393 | 0.729 |
| heat of formation | As | node random forest | 0.240 | 0.856 |
| heat of formation | As | node gradient boosting | 0.247 | 0.851 |
| heat of formation | As | direct linear response | 0.778 | 0.484 |
| heat of formation | As | free plus metric | 0.345 | 0.779 |
| heat of formation | Cs | node linear | 0.536 | 0.858 |
| heat of formation | Cs | node random forest | 0.542 | 0.858 |
| heat of formation | Cs | node gradient boosting | 0.509 | 0.874 |
| heat of formation | Cs | direct linear response | 0.685 | 0.867 |
| heat of formation | Cs | free plus metric | 0.500 | 0.868 |
| volume per fu | As | node linear | 7.719 | 0.811 |
| volume per fu | As | node random forest | 7.806 | 0.773 |
| volume per fu | As | node gradient boosting | 7.059 | 0.796 |
| volume per fu | As | direct linear response | 12.208 | 0.770 |
| volume per fu | As | free plus metric | 7.681 | 0.803 |
| volume per fu | Cs | node linear | 13.604 | 0.861 |
| volume per fu | Cs | node random forest | 9.639 | 0.844 |
| volume per fu | Cs | node gradient boosting | 9.482 | 0.854 |
| volume per fu | Cs | direct linear response | 25.612 | 0.852 |
| volume per fu | Cs | free plus metric | 12.869 | 0.871 |
| magnetic moment | As | node linear | 1.405 | 0.524 |
| magnetic moment | As | node random forest | 0.792 | 0.663 |
| magnetic moment | As | node gradient boosting | 0.834 | 0.665 |
| magnetic moment | As | direct linear response | 1.776 | 0.513 |
| magnetic moment | As | free plus metric | 1.348 | 0.544 |
| magnetic moment | Cs | node linear | 1.282 | 0.561 |
| magnetic moment | Cs | node random forest | 0.867 | 0.687 |
| magnetic moment | Cs | node gradient boosting | 0.913 | 0.657 |
| magnetic moment | Cs | direct linear response | 1.926 | 0.589 |
| magnetic moment | Cs | free plus metric | 1.242 | 0.580 |
| magnetic moment signed | As | node linear | 1.559 | 0.515 |
| magnetic moment signed | As | node random forest | 1.057 | 0.653 |
| magnetic moment signed | As | node gradient boosting | 1.114 | 0.624 |
| magnetic moment signed | As | direct linear response | 1.879 | 0.519 |
| magnetic moment signed | As | free plus metric | 1.502 | 0.540 |
| magnetic moment signed | Cs | node linear | 1.456 | 0.588 |
| magnetic moment signed | Cs | node random forest | 0.944 | 0.662 |
| magnetic moment signed | Cs | node gradient boosting | 1.013 | 0.645 |
| magnetic moment signed | Cs | direct linear response | 2.020 | 0.566 |
| magnetic moment signed | Cs | free plus metric | 1.340 | 0.611 |

## Direct-response block attribution

Formation-energy B substitutions, oriented as held level minus partner.

| level | component | mean contribution | mean absolute contribution |
|---|---|---:|---:|
| As | intercept | +0.056 | 0.081 |
| As | theta | +0.514 | 0.620 |
| As | context A | -0.000 | 0.212 |
| As | context X | +0.000 | 0.260 |
| As | true response | -0.471 | 0.584 |
| As | predicted response | +0.570 | 0.728 |
| Cs | intercept | -0.021 | 0.025 |
| Cs | theta | +2.008 | 2.008 |
| Cs | context A | +0.000 | 0.231 |
| Cs | context X | +0.000 | 0.113 |
| Cs | true response | +1.259 | 1.271 |
| Cs | predicted response | +1.987 | 1.987 |

## Descriptor-family ablation

Formation-energy B substitutions only; every row is refitted without the held level.

| level | configuration | MAE | sign | predicted held-partner | bias |
|---|---|---:|---:|---:|---:|
| As | full | 1.069 | 0.312 | +0.570 | +1.041 |
| As | no context | 1.043 | 0.264 | +0.570 | +1.041 |
| As | no intercept | 0.997 | 0.342 | +0.491 | +0.962 |
| As | drop periodic identity from B | 0.447 | 0.717 | -0.147 | +0.324 |
| As | drop periodic identity from A/X context | 1.067 | 0.324 | +0.570 | +1.041 |
| As | drop size from B | 0.357 | 0.806 | -0.507 | -0.036 |
| As | drop size from A/X context | 1.070 | 0.315 | +0.570 | +1.041 |
| As | drop electron energetics from B | 0.305 | 0.808 | -0.589 | -0.117 |
| As | drop electron energetics from A/X context | 1.069 | 0.319 | +0.570 | +1.041 |
| As | drop valence from B | 0.943 | 0.324 | +0.464 | +0.935 |
| As | drop valence from A/X context | 1.067 | 0.311 | +0.570 | +1.041 |
| Cs | full | 0.811 | 0.973 | +1.987 | +0.728 |
| Cs | no context | 0.780 | 0.973 | +1.987 | +0.728 |
| Cs | no intercept | 0.922 | 0.973 | +2.126 | +0.868 |
| Cs | drop periodic identity from B | 0.634 | 0.921 | +0.765 | -0.494 |
| Cs | drop periodic identity from A/X context | 0.821 | 0.973 | +1.987 | +0.728 |
| Cs | drop size from B | 0.781 | 0.876 | +0.518 | -0.741 |
| Cs | drop size from A/X context | 0.817 | 0.973 | +1.987 | +0.728 |
| Cs | drop electron energetics from B | 0.654 | 0.907 | +0.774 | -0.485 |
| Cs | drop electron energetics from A/X context | 0.795 | 0.973 | +1.987 | +0.728 |
| Cs | drop valence from B | 1.764 | 0.973 | +3.021 | +1.763 |
| Cs | drop valence from A/X context | 0.809 | 0.973 | +1.987 | +0.728 |

Because the elemental descriptors are strongly correlated, these refitted family ablations are more defensible than interpreting an individual ridge coefficient. They identify model dependence, not an elemental cause.


## Metadata control

Every As and Cs calculation in this grid uses the same recorded stack: RPBE, GPAW, PAW, cubic Pm-3m ABX3. The available metadata therefore contain no element-specific code, functional, pseudopotential family or prototype branch that can explain the gap.


<!-- ANALYSIS: hand-written below; preserved across reruns -->
