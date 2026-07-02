# Build instructions

Compile with:

```bash
pdflatex bops_publishable_ieee.tex
bibtex bops_publishable_ieee
pdflatex bops_publishable_ieee.tex
pdflatex bops_publishable_ieee.tex
```

The figures are stored in `figures/` as PDFs.

This draft is written as a publishable empirical-study manuscript, not as a method-win paper. Do not claim BOPS improves VLM QA unless a future question-aware patch selector changes the DocVQA results.
