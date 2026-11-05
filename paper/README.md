# EMBC 2027 paper

`main.pdf` is the compiled paper. Every number, table and figure in it is generated from `results/`:

```bash
python experiments/paper_assets.py   # writes paper/numbers.tex, paper/tables/, paper/figures/
cd paper && pdflatex main.tex && pdflatex main.tex
```

`IEEEtran.cls` is the standard IEEE conference class (v1.8b). To edit on Overleaf, upload this folder as a project.
