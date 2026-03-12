# Data Directory

Place your FITS files here.

## Expected Data

Download the synthetic ALMA observations from the test instructions link.

Each FITS file should:
- Be in `.fits` or `.fit` format
- Contain a 4-layer data cube
- Have dimensions 600×600 pixels
- Represent 1250 micron continuum observations

## Example Structure

```
data/
├── disk_001.fits
├── disk_002.fits
├── disk_003.fits
└── ...
```

## Note

FITS files are excluded from git tracking due to size.
See `.gitignore` for details.
