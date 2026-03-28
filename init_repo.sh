#!/bin/bash
# CONFLUX — GitHub Repo Init Script
# Run this after extracting the archive to push to GitHub.
#
# Prerequisites:
#   - GitHub CLI (gh) authenticated, OR git remote configured
#   - PyPI account + API token (for releases)
#
# Usage:
#   tar xzf conflux-repo-v0.1.0.tar.gz
#   cd conflux
#   chmod +x init_repo.sh
#   ./init_repo.sh

set -e

REPO_NAME="nage-ai/conflux"
REPO_DESC="Cross-architecture Optimized N-source Fine-tuning via Low-rank Unified eXtraction"

echo "═══════════════════════════════════════"
echo " CONFLUX — GitHub Repository Setup"
echo "═══════════════════════════════════════"

# Step 1: Init git
if [ ! -d ".git" ]; then
    echo ""
    echo "[1/5] Initializing git repo..."
    git init
    git branch -M main
else
    echo "[1/5] Git already initialized"
fi

# Step 2: Create GitHub repo (if gh is available)
if command -v gh &> /dev/null; then
    echo ""
    echo "[2/5] Creating GitHub repository..."
    gh repo create "$REPO_NAME" \
        --public \
        --description "$REPO_DESC" \
        --source . \
        --remote origin \
        2>/dev/null || echo "  → Repo may already exist, continuing..."
else
    echo ""
    echo "[2/5] GitHub CLI not found. Set remote manually:"
    echo "  git remote add origin git@github.com:${REPO_NAME}.git"
fi

# Step 3: Initial commit
echo ""
echo "[3/5] Creating initial commit..."
git add -A
git commit -m "feat: CONFLUX v0.1.0 — initial release

Cross-architecture Optimized N-source Fine-tuning via Low-rank Unified eXtraction.

5 modules: CKA matching, Residual-SVD init, adaptive rank allocation,
informativeness profiling, and multi-source switching.

Part of the Nage AI ecosystem." 2>/dev/null || echo "  → Already committed"

# Step 4: Push
echo ""
echo "[4/5] Pushing to GitHub..."
git push -u origin main 2>/dev/null || echo "  → Push failed. Run: git push -u origin main"

# Step 5: Post-setup instructions
echo ""
echo "[5/5] Post-setup checklist:"
echo ""
echo "  1. GitHub Secrets (Settings → Secrets → Actions):"
echo "     → PYPI_API_TOKEN  : Your PyPI API token"
echo ""
echo "  2. GitHub Environments:"
echo "     → Create 'pypi' environment (Settings → Environments)"
echo "     → Add required reviewers for release approval"
echo ""
echo "  3. Branch protection (Settings → Branches):"
echo "     → Protect 'main': require PR reviews + status checks"
echo "     → Required checks: lint, test, syntax-validate, build"
echo ""
echo "  4. First release:"
echo "     git tag v0.1.0"
echo "     git push origin v0.1.0"
echo "     → CI runs tests → publishes to PyPI → creates GitHub Release"
echo ""
echo "═══════════════════════════════════════"
echo " Done. Repository ready at:"
echo " https://github.com/${REPO_NAME}"
echo "═══════════════════════════════════════"
