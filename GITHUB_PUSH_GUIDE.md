# Publish DermaSeg to GitHub

## Before publishing

1. Replace student placeholders in the report.
2. Keep the dataset archive and raw PH² images out of GitHub.
3. Keep trained `.pt`/`.pth` checkpoint files out of normal Git history. Share them separately with the TA only using an approved route, or use an artifact host if a public demo needs them.
4. Check that the source, configuration, split protocol, requirements, README and result summaries correspond to the final runs.
5. Do not commit API keys, tokens, passwords or private files.

## Recommended Windows workflow (GitHub Desktop)

1. Download the **final** `DermaSeg_Project` folder from Google Drive. Do not use an older starter ZIP if it predates the final experiment edits.
2. Extract it locally. Add `app.py`, `STREAMLIT_DEMO.md` and this guide to the project root. Copy the actual `src/` and `configs/` from the final Colab/Drive project.
3. Open GitHub in a browser and create a new empty repository, for example `DermaSeg-PH2-Skin-Lesion-Segmentation`. Do not initialize it with a README or license if the local folder already has files.
4. Install/open GitHub Desktop, choose **File → Add local repository**, select the extracted project folder, review the changed files, commit, then choose **Publish repository**. Select Public only if the project contains no private data and all included material is permitted to share.
5. Open the repository page on GitHub and verify `README.md`, `src/`, `configs/`, `requirements.txt`, `app.py`, and result-summary files are visible. Copy the repository URL into the report appendix.

## Command-line alternative

Open PowerShell in the project root after creating an empty GitHub repository:

```powershell
git init
git branch -M main
git add .
git status
git commit -m "Initial DermaSeg project: PH2 U-Net segmentation"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

Replace the remote URL with your own repository URL. If Git asks you to authenticate, use GitHub's supported sign-in flow or a credential manager; a GitHub account password is not accepted for Git operations over HTTPS.

## Important

- Do not run `git add archive.zip`, raw data, or checkpoints.
- Run `git status` before committing and inspect the staged files.
- The public repository should not contain absolute paths from Colab or Windows. Regenerate the split manifest after placing the dataset in a documented local path, or make the loader resolve case IDs against the local dataset root.
