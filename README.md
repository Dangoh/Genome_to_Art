# genome_to_art

FASTA-driven abstract art generator where randomness is **deterministically derived from nucleotide content** (k-mers).  
Same FASTA + same CLI arguments ⇒ **identical image output** (reproducible).

This tool can render multiple “styles” (fluid flow fields, contour/lava fields, cellular blobs, radial/circular DNA-inspired motifs, etc.) and multiple color “themes” (including curated psychedelic palettes).

## Features

- **Genome-driven randomness** (k-mers → deterministic RNG stream)
- **Reproducible output**: identical input FASTA + identical args = identical PNG
- Many **styles** and **themes**
- `--all` mode to generate a batch of images (styles × themes)
- `--fill` convenience flag to increase coverage and reduce white space
- Pure Python, minimal dependencies

## Installation

Python 3.8+ recommended.

```bash
pip install numpy matplotlib
