# Push to GitHub

Git is initialized. To push to a new GitHub repo:

1. **Create repo on GitHub** (github.com → New repository)
   - Name: `Langchain-examples` (or any name)
   - Do NOT initialize with README (we already have one)

2. **Add remote and push:**
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/Langchain-examples.git
   git push -u origin dev
   git push origin main   # if you want main branch on remote too
   ```

3. **Branches:** Current branch is `dev` with all commits. `main` exists but has no commits yet. To sync main:
   ```bash
   git checkout main
   git merge dev
   git push -u origin main
   ```
