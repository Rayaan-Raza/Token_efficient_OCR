"""Image preprocessing baselines and the BOPS method.

Modules:
    resize: Area-ratio downscaling baseline
    compression: JPEG/WebP byte-budget baseline
    overview: Low-resolution global context image
    patch_grid: Candidate patch grid and cropping
    patch_scoring: OCR-guided patch importance scores
    patch_nms: Overlap suppression for patch selection
    bops: Full overview-plus-patch pipeline
"""
