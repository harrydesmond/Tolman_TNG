# TNG Image Segmentation Diagnostic

Threshold: `background + 3.0 sigma`; minimum island size: `8` pixels.
Selected objects with object-level summaries: 2,901.

| Quantity | alpha | N |
|---|---:|---:|
| median_aperture_ratio | 0.412 +/- 0.035 | 2901 |
| median_segment_ratio | -0.021 +/- 0.091 | 2901 |
| median_isolated_segment_ratio | -0.047 +/- 0.071 | 312 |

Caveat: this is threshold segmentation with catalogue-position label assignment, not a full ASTRODEEP deblender.
