#!/bin/bash
# Deploy cashew metrics dashboard to Cloudflare Pages
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CASHEW_DIR="$(dirname "$SCRIPT_DIR")"
METRICS_DIR="$CASHEW_DIR/metrics-dashboard"
DATA_DIR="$METRICS_DIR/data"

# Load Cloudflare credentials
export CLOUDFLARE_EMAIL="bot@example.com"
export CLOUDFLARE_API_KEY=$(security find-generic-password -s 'cloudflare-global-api-key' -w 2>/dev/null)
if [ -z "$CLOUDFLARE_API_KEY" ]; then
    echo "❌ Missing Cloudflare API key from keychain"
    exit 1
fi

# Export fresh metrics data from the local server
echo "📊 Exporting metrics data..."
mkdir -p "$DATA_DIR"

if curl -sf http://localhost:8787/api/summary > "$DATA_DIR/summary.json" 2>/dev/null && \
   curl -sf "http://localhost:8787/api/timeseries?type=retrieval&hours=24" > "$DATA_DIR/timeseries.json" 2>/dev/null && \
   curl -sf http://localhost:8787/api/recent > "$DATA_DIR/recent.json" 2>/dev/null; then
    echo "   ✓ Exported from live metrics server"
else
    # Fallback: generate data directly from the metrics DB
    echo "   ⚠️ Live server not available, generating from DB..."
    cd "$CASHEW_DIR"
    python3 -c "
import json, sys
sys.path.insert(0, '.')
from core.metrics import get_metrics_summary, get_metrics_timeseries, get_recent_metrics
from core.config import get_db_path
db = get_db_path()
with open('$DATA_DIR/summary.json', 'w') as f: json.dump(get_metrics_summary(db), f)
with open('$DATA_DIR/timeseries.json', 'w') as f: json.dump(get_metrics_timeseries(db, 'retrieval', 24), f)
with open('$DATA_DIR/recent.json', 'w') as f: json.dump(get_recent_metrics(db), f)
print('   ✓ Generated from DB')
"
fi

# Deploy to Cloudflare Pages
echo "🚀 Deploying to Cloudflare Pages..."
cd "$METRICS_DIR"
npx wrangler pages deploy . --project-name cashew-metrics --commit-dirty=true 2>&1

echo "✅ Metrics dashboard deployed: https://cashew-metrics.pages.dev"
