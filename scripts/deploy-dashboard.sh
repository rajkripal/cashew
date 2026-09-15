#!/bin/bash
# Auto-redeploy cashew dashboard to Cloudflare Pages
# Exports current graph.db → dashboard JSON, then deploys
set -e

CASHEW_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DASHBOARD_DIR="$CASHEW_DIR/dashboard"
DB_PATH="$CASHEW_DIR/data/graph.db"

# Load Cloudflare credentials
export CLOUDFLARE_EMAIL="bot@example.com"
export CLOUDFLARE_API_KEY=$(security find-generic-password -s 'cloudflare-global-api-key' -w 2>/dev/null)

if [ -z "$CLOUDFLARE_API_KEY" ]; then
    echo "❌ Missing Cloudflare API key from keychain"
    exit 1
fi

# Export current graph to dashboard JSON
echo "📊 Exporting graph to dashboard..."
cd "$CASHEW_DIR"
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/export_dashboard.py "$DB_PATH" "$DASHBOARD_DIR/data/graph.json" 2>/dev/null

# Deploy to Cloudflare Pages
echo "🚀 Deploying to Cloudflare Pages..."
npx wrangler pages deploy "$DASHBOARD_DIR" --project-name cashew-dashboard 2>&1 | tee /tmp/deploy-dashboard-output.log | tail -5
DEPLOY_STATUS=${PIPESTATUS[0]}

if [ "$DEPLOY_STATUS" -ne 0 ]; then
    echo "❌ Dashboard deploy failed (wrangler exit $DEPLOY_STATUS)"
    exit "$DEPLOY_STATUS"
fi

echo "✅ Dashboard deployed: https://cashew-dashboard.pages.dev"
