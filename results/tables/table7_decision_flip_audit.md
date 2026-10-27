| Target vs PyTorch FP32 | Thresholds | Target boxes | Matched | Flip rate | Lost TPs | New TPs | New FPs | Suppressed FPs | FP share of flips |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PyTorch FP32 (CPU) | default | 14185 | 14185 | 0.0000 | 0 | 0 | 0 | 0 | 0.0% |
| PyTorch FP32 (CPU) | calibrated | 731 | 731 | 0.0000 | 0 | 0 | 0 | 0 | 0.0% |
| ORT FP32 (CPU) | default | 14185 | 14185 | 0.0000 | 0 | 0 | 0 | 0 | 0.0% |
| ORT FP32 (CPU) | calibrated | 731 | 731 | 0.0000 | 0 | 0 | 0 | 0 | 0.0% |
| ORT FP16 (CPU) | default | 14193 | 14143 | 0.0065 | 0 | 1 | 49 | 42 | 98.9% |
| ORT FP16 (CPU) | calibrated | 731 | 731 | 0.0000 | 0 | 0 | 0 | 0 | 0.0% |
| ORT INT8 (CPU) | default | 14998 | 9854 | 0.4902 | 18 | 10 | 5134 | 4313 | 99.7% |
| ORT INT8 (CPU) | calibrated | 704 | 604 | 0.2732 | 11 | 38 | 62 | 116 | 78.4% |
