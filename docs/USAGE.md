<!-- 
SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and Institute of Radiology, Uniklinikum Erlangen, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU).
SPDX-License-Identifier: CC BY-NC 4.0
-->

# How to use

### Overview

See all CLI commands:

```bash
aorta_aneurysm_v1 -h
```

Each command is documented, see:

```bash
aorta_aneurysm_v1 <command> -h
```

### Diameter calculation example

```bash
aorta_aneurysm_v1 diam_batch -j 8 --rm2 \
    /path/to/segmentations/dir \
    /path/to/output/csv
```

* Images should be in format: `<case_name>_0000.nii.gz`
* Segs should be in format: `<case_name>.nii.gz`
