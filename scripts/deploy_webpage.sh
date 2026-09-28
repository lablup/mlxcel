#!/bin/bash

# Deploy mlxcel download webpage to GitHub Pages
# Usage: ./scripts/deploy_webpage.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WEBPAGE_DIR="$PROJECT_ROOT/webpage/site"

# Repository settings
REPO_URL="git@github.com:lablup/mlxcel-releases.git"
BRANCH="gh-pages"

echo "Starting deployment to $REPO_URL [$BRANCH]..."

# 1. Build the project
echo "Building the project..."
cd "$WEBPAGE_DIR"
pnpm install
pnpm run build

# 2. Navigate to the build output directory
cd out

# 3. Create .nojekyll to prevent Jekyll from ignoring _next directory
touch .nojekyll

# 4. Replace index.html with a simple redirect page
echo "Creating redirect index.html..."
cat > index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>mlxcel - High-Performance LLM Inference on Apple Silicon</title>
  <meta name="description" content="Run 60+ LLM and VLM models natively on Apple Silicon with Metal acceleration. Rust-powered, zero Python dependencies, OpenAI-compatible API.">

  <!-- Open Graph -->
  <meta property="og:type" content="website">
  <meta property="og:url" content="https://mlxcel.ai">
  <meta property="og:site_name" content="mlxcel">
  <meta property="og:title" content="mlxcel - High-Performance LLM Inference on Apple Silicon">
  <meta property="og:description" content="Run 60+ LLM and VLM models natively on Apple Silicon with Metal acceleration. Rust-powered, zero Python dependencies, OpenAI-compatible API.">
  <meta property="og:image" content="https://mlxcel.ai/og-image.png">
  <meta property="og:image:width" content="1200">
  <meta property="og:image:height" content="630">
  <meta property="og:locale" content="en_US">
  <meta property="og:locale:alternate" content="ko_KR">

  <!-- Twitter Card -->
  <meta name="twitter:card" content="summary_large_image">
  <meta name="twitter:title" content="mlxcel - High-Performance LLM Inference on Apple Silicon">
  <meta name="twitter:description" content="Run 60+ LLM and VLM models natively on Apple Silicon. Rust-powered, Metal-accelerated, OpenAI-compatible API.">
  <meta name="twitter:image" content="https://mlxcel.ai/og-image.png">
  <meta name="twitter:creator" content="@lablupinc">

  <meta http-equiv="refresh" content="0;url=./en">
  <script>
    // Detect browser language and redirect accordingly
    const lang = navigator.language.toLowerCase().startsWith('ko') ? 'ko' : 'en';
    window.location.replace('./' + lang);
  </script>
</head>
<body>
  <p>Redirecting to <a href="./en">English page</a>...</p>
</body>
</html>
EOF

# 5. Create a fresh git repo in the output directory
echo "Initializing git..."
git init
git branch -m $BRANCH

# 6. Add all files
git add .

# 7. Commit
git commit -m "Deploy: $(date '+%Y-%m-%d %H:%M:%S')"

# 8. Push to the remote repository
# Warning: This will force push and overwrite existing content
echo "Pushing to remote..."
git push -f $REPO_URL $BRANCH

echo "Deployment complete!"
