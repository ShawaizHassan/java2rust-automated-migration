# Data Sync: DVC vs Git LFS

**Recommendation:** Use **DVC** for large datasets and experiment artifacts. It keeps Git lean and integrates with remote storage.

## Option A: DVC (recommended)

```bash
pip install dvc
dvc init
dvc remote add -d origin <REMOTE_URL>   # e.g., s3://bucket/path or gdrive remote
dvc add data/raw data/processed
git add data/.gitignore data/*.dvc .dvc .dvcignore
git commit -m "data: track with DVC"
dvc push
```
