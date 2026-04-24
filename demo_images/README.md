# Demo Images

Preloaded sample images used by the **Demo Mode** feature.

## Structure

```
demo_images/
├── brain_tumor/         ← Brain MRI JPGs (tracked in git)
│   ├── glioma_sample.jpg
│   ├── meningioma_sample.jpg
│   ├── pituitary_sample.jpg
│   └── no_tumor_sample.jpg
└── pneumonia/           ← Chest X-Ray DICOMs (NOT tracked in git - too large)
    ├── pneumonia_positive.dcm
    └── pneumonia_negative.dcm
```

## Setup

Brain tumor images are included in the repo.

For **pneumonia demo samples**, place the following DICOM files manually:

| File | Source |
|------|--------|
| `pneumonia_positive.dcm` | RSNA Pneumonia Dataset — `0004cfab-14fd-4e49-80ba-63a80b6bddd6.dcm` |
| `pneumonia_negative.dcm` | RSNA Pneumonia Dataset — `000e3a7d-c0ca-4349-bb26-5af2d8993c3d.dcm` |

Without the DICOM files, only Brain Tumor demos will work. Pneumonia demos will show a 404 error.
