# PyRouge WordNet Database Error Fix

## Error Message

```
Cannot open exception db file for reading: /path/to/pyrouge/tools/ROUGE-1.5.5/data/WordNet-2.0.exc.db
```

This error occurs when using `tilse` for ROUGE evaluation because it depends on `pyrouge`, which requires a WordNet exception database file.

## Solution

### Step 1: Find pyrouge installation path

```bash
python -c "import pyrouge; print(pyrouge.__file__)"
# Example output: /opt/anaconda3/lib/python3.12/site-packages/pyrouge/__init__.py
```

### Step 2: Navigate to ROUGE data directory

```bash
cd /opt/anaconda3/lib/python3.12/site-packages/pyrouge/tools/ROUGE-1.5.5/data
```

### Step 3: Build the WordNet exception database

```bash
# Remove old database files if they exist
rm -f WordNet-2.0.exc.db

# Go to exceptions directory
cd WordNet-2.0-Exceptions

# Remove old database here too
rm -f WordNet-2.0.exc.db

# Build the database (requires Perl)
./buildExeptionDB.pl . exc WordNet-2.0.exc.db

# Go back to parent and create symlink
cd ..
ln -sf WordNet-2.0-Exceptions/WordNet-2.0.exc.db WordNet-2.0.exc.db
```

### One-liner

```bash
PYROUGE_PATH=$(python -c "import pyrouge; import os; print(os.path.dirname(pyrouge.__file__))")
cd "$PYROUGE_PATH/tools/ROUGE-1.5.5/data" && \
rm -f WordNet-2.0.exc.db && \
cd WordNet-2.0-Exceptions && \
rm -f WordNet-2.0.exc.db && \
./buildExeptionDB.pl . exc WordNet-2.0.exc.db && \
cd .. && \
ln -sf WordNet-2.0-Exceptions/WordNet-2.0.exc.db WordNet-2.0.exc.db
```

## Prerequisites

- **Perl** must be installed
- **DB_File** Perl module (usually pre-installed)

If `DB_File` is missing:
```bash
# macOS
brew install berkeley-db

# Ubuntu/Debian
sudo apt-get install libdb-dev
cpan DB_File
```

## Verification

After fixing, test with:

```python
from tilse.data.timelines import Timeline, GroundTruth
from tilse.evaluation.rouge import TimelineRougeEvaluator
from datetime import date

pred = Timeline({date(2011, 1, 25): ["Test sentence."]})
gold = Timeline({date(2011, 1, 25): ["Another sentence."]})
gt = GroundTruth([gold])

evaluator = TimelineRougeEvaluator(measures=["rouge_1"])
scores = evaluator.evaluate_concat(pred, gt)
print(scores)  # Should print scores without error
```

## References

- https://github.com/tagucci/pythonrouge#error-handling
- https://github.com/bheinzerling/pyrouge/issues/8
