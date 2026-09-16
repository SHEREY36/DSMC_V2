# Multivariate excitation repair: offline validation

- Physical nodes: 18
- Held-out excitations: 648
- Parameter response pass: `True`
- Distribution response pass: `True`
- Offline validation pass: `True`

The model fits the central ±0.25 perturbations and validates on both signs of the |eta|=0.5 boundary without reusing those points for fitting. The comparison below is distribution-level: lower Wasserstein-1 distance is better.

```json
{
  "angular_quantile_wasserstein1": {
    "baseline": {
      "maximum": 0.06616825387947332,
      "median": 0.004026596445010732,
      "p90": 0.02774623270898479,
      "p95": 0.04120479898597924
    },
    "multivariate": {
      "maximum": 0.014744399360525481,
      "median": 0.0012867973187594048,
      "p90": 0.0065855642111137345,
      "p95": 0.00930634347116771
    }
  },
  "artifact_tangent_to_exact_response_wasserstein1": {
    "maximum": 0.06282728422103877,
    "median": 1.979005501322008e-05,
    "p90": 0.0008303268558531852,
    "p95": 0.002125619569221982
  },
  "conditional_energy_quantile_wasserstein1": {
    "artifact_tangent": {
      "maximum": 0.04633168410690494,
      "median": 0.0006151013638000443,
      "p90": 0.002253777485266552,
      "p95": 0.004082494631699539
    },
    "baseline": {
      "maximum": 0.052318019031476135,
      "median": 0.0009793266062275582,
      "p90": 0.004968266696609557,
      "p95": 0.00821131730629399
    },
    "lambda1_only": {
      "maximum": 0.054930189069092895,
      "median": 0.0022530444222707696,
      "p90": 0.013733274600637243,
      "p95": 0.019169804453549367
    },
    "multivariate_exact": {
      "maximum": 0.024572546012814927,
      "median": 0.0005958234792978721,
      "p90": 0.0020596035005926282,
      "p95": 0.0031253898416174417
    }
  },
  "global_heldout_relative_rmse": {
    "eta1": 0.027279975236697382,
    "eta2": 0.032175512023254686,
    "lambda1": 0.029326438770478278,
    "lambda2": 0.026149081140889437,
    "lambda3": 0.02967608319939937,
    "lambda4": 0.018818238370794087
  }
}
```
