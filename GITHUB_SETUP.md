# GitHub Setup Guide

## Step 1: Create a GitHub Repository

1. Go to [GitHub.com](https://github.com) and sign in
2. Click the "+" icon in the top right → "New repository"
3. Name it (e.g., `LangChain_RAG` or `rag-qa-system`)
4. **DO NOT** initialize with README, .gitignore, or license (we already have these)
5. Click "Create repository"

## Step 2: Add All Files and Commit

Run these commands in your terminal (from the project root):

```bash
# Add all files
git add .

# Check what will be committed
git status

# Create initial commit
git commit -m "Initial commit: RAG Q&A System with LangChain and Ollama"
```

## Step 3: Connect to GitHub and Push

After creating the repository on GitHub, you'll see a page with setup instructions. Use these commands:

```bash
# Add the remote repository (replace YOUR_USERNAME and REPO_NAME)
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

**Example:**
If your GitHub username is `johndoe` and repository name is `rag-qa-system`:
```bash
git remote add origin https://github.com/johndoe/rag-qa-system.git
git branch -M main
git push -u origin main
```

## Step 4: Verify

1. Go to your GitHub repository page
2. You should see all your files there
3. The README.md will be displayed on the repository homepage

## Troubleshooting

### If you get authentication errors:
- Use a Personal Access Token instead of password
- Generate one at: GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
- Use the token as your password when pushing

### If you need to update later:
```bash
git add .
git commit -m "Your commit message"
git push
```

