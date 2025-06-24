#!/bin/bash
# Final cleanup script to complete the restructuring

echo "🧹 Final Cleanup for GitHub Commit"
echo "=================================="

# Move remaining files to archive
echo "📦 Moving remaining files to archive..."

# Move old documentation
mv -f PROJECT_IMPROVEMENTS.md archive/old_versions/ 2>/dev/null
mv -f TOMORROW_GUIDE.md archive/old_versions/ 2>/dev/null
mv -f usage.txt archive/old_versions/ 2>/dev/null
mv -f pinescript.txt archive/old_versions/ 2>/dev/null

# Move data files
mkdir -p archive/data_files
mv -f "NIFTY 50_minute_data.csv" archive/data_files/ 2>/dev/null
mv -f nifty_backtest_results.csv archive/data_files/ 2>/dev/null

# Move remaining test/check scripts
mv -f check_config.py archive/old_versions/ 2>/dev/null

# Clean up Python cache
echo "🗑️  Removing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
find . -type f -name "*.pyo" -delete 2>/dev/null

# Remove virtual environments from main directory
echo "📁 Note: Virtual environments detected..."
echo "   - 'new' directory"
echo "   - 'venv' directory"
echo "   - 'test' directory (if not tests)"
echo "   These should not be committed to Git"

# Update .gitignore to ensure all are excluded
echo "📝 Updating .gitignore..."
cat >> .gitignore << 'EOF'

# Additional exclusions
new/
test/
venv/
__pycache__/
*.pyc
*.pyo
.pytest_cache/
EOF

# Create a clean directory listing
echo ""
echo "📊 Final Directory Structure:"
echo "============================"
ls -la | grep -E "^d" | grep -v "\."
echo ""
echo "📄 Core Files:"
echo "============="
ls -1 *.py 2>/dev/null
echo ""

# Summary of what's ready for commit
echo "✅ Ready for GitHub commit:"
echo "  - README.md (documentation)"
echo "  - LICENSE (MIT license)"
echo "  - CHANGELOG.md (version history)"
echo "  - requirements.txt (dependencies)"
echo "  - setup.py (package setup)"
echo "  - .env.example (config template)"
echo "  - .gitignore (ignore rules)"
echo "  - main.py (core bot)"
echo "  - cli.py (CLI interface)"
echo "  - backtest_strategy.py (backtesting)"
echo "  - auth/ (authentication module)"
echo "  - trading/ (trading logic)"
echo "  - config/ (configuration)"
echo "  - utils/ (utilities)"
echo "  - tests/ (test suite)"
echo ""
echo "📦 Archived (not for commit):"
echo "  - archive/ (old files, test scripts)"
echo "  - data/ (runtime data)"
echo "  - logs/ (log files)"
echo "  - Virtual environments (new/, venv/, test/)"
echo ""
echo "🎯 Next Steps:"
echo "1. Review the changes: git status"
echo "2. Add files: git add ."
echo "3. Commit: git commit -m 'v2.0.0: Dynamic capital management'"
echo "4. Push: git push origin main"
EOF

chmod +x final_cleanup.sh
