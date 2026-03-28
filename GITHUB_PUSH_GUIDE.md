# 📤 Push to GitHub Guide

Your local repository is ready! Follow these steps to push to GitHub:

## Step 1: Create a GitHub Repository

1. Go to [GitHub.com](https://github.com) and log in
2. Click **"+"** → **"New repository"**
3. Fill in:
   - **Repository name**: `sentiment-multilingual` (or your preference)
   - **Description**: "Multilingual sentiment analysis with Darija support (XLM-RoBERTa)"
   - **Visibility**: Public (or Private)
   - **Do NOT initialize** with README or .gitignore (keep these unchecked!)
4. Click **"Create repository"**

## Step 2: Copy Remote URL

After creating, GitHub shows you the commands. **Copy the HTTPS URL**, e.g.:
```
https://github.com/your-username/sentiment-multilingual.git
```

## Step 3: Add Remote & Push

Run these commands in PowerShell (in the project directory):

```powershell
# Add the remote
git remote add origin https://github.com/your-username/sentiment-multilingual.git

# Rename branch (optional, for consistency)
git branch -M main

# Push to GitHub
git push -u origin main
```

**Replace** `your-username` with your actual GitHub username!

## Step 4: Verify

- Visit your repository on GitHub: `https://github.com/your-username/sentiment-multilingual`
- You should see all files with the README preview

## Common Issues

### "Repository already exists"
→ You already have a remote. Use:
```powershell
git remote set-url origin https://github.com/your-username/sentiment-multilingual.git
```

### "Authentication failed"
→ Use Personal Access Token (PAT) instead:
1. Generate PAT: GitHub Settings → Developer settings → Personal access tokens
2. Use as password when prompted

### "Nothing to push"
→ Your commits are already pushed. Check with:
```powershell
git log --oneline
git remote -v
```

## Future Commits

After the initial push, use:
```powershell
git add .
git commit -m "Description of changes"
git push origin main
```

## Useful Commands

```powershell
# Check status
git status

# View commits
git log --oneline --graph

# See remotes
git remote -v

# Create a branch
git checkout -b feature/new-feature
git push -u origin feature/new-feature
```

---

**Repository is initialized and ready to push! 🚀**

Questions? Check the GitHub docs: https://docs.github.com/en/get-started/using-git/pushing-commits-to-a-remote-repository
