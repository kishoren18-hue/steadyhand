# Contributing

Issues and pull requests are welcome, especially:

- results on other EMG datasets (amputee data and multi-day recordings most of all)
- real-time evaluations with LibEMG's online classifier
- better confidence scores to plug into the cutoff rules

Before opening a pull request, run the tests:

```bash
pip install -e ".[test]"
python -m pytest tests -q
```

Every number in the README comes from a script in `experiments/`. If you
change a method, rerun the affected experiment and update `results/` in the
same pull request.
