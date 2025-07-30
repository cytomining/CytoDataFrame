# SBI2 2025 Abstract Submission - CytoDataFrame

## Authors

Dave Bunten, Jenna Tomkinson, Gregory P. Way

## Title (under 100 characters)

Exploring single‑cell images and profiles together with CytoDataFrame

## Abstract (under 500 words)

Image‑based profiling involves the analysis of vast tables of profiles alongside many related cellular images.
Context-switching between tables of feature data and image viewers makes it challenging to build an intuition of profiles and their corresponding cellular objects.
CytoDataFrame solves this by bringing each cell or compartment object picture and feature data into one interactive table for Jupyter notebook environments.

With CytoDataFrame, you can:

- Keep context at a glance. Every row shows both the features (for example, size, shape, or intensity) and an image of the same cell, so you immediately know what the data represent.
- Adjust on the fly. Brightness sliders and mask toggles let you explore the images directly in jupyter. For example, highlight segmentation errors or unusual cells without writing extra code.
- Filter and explore. Select subpopulations such as outliers, specific phenotypes, or quality‑control failures so you can make decisions alongside the visual representation of biological objects.
- Stay in your workflow. Extending Pandas DataFrames, it works with your favorite analysis and plotting commands, and exports results for downstream sharing or publication.
- Integration with coSMicQC. CytoDataFrame integrates directly with coSMicQC, which is quality control software for single-cell segmentations. coSMicQC functions return CytoDataFrames so you can make quality control decisions alongside images of cells.

By unifying images and profiles in a single view, CytoDataFrame accelerates troubleshooting, improves data quality checks, and helps teams spot biological patterns faster.
It makes your next discovery just a click away, perfect for anyone doing high‑content imaging.
