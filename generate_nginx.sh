#!/usr/bin/env bash
set -euo pipefail

# --- CONFIGURATION ---
DOMAIN="violet-koala-19661.zap.cloud"
PORT=5000

# --- PRECHECKS ---
if [ "$(id -u)" -ne 0 ]; then
  echo "⚠️  Please run as root or with sudo."
  exit 1
fi

# --- INSTALL PACKAGES ---
apt update
apt install -y nginx certbot python3-certbot-nginx

# --- OPEN FIREWALL (if ufw is installed) ---
if command -v ufw &>/dev/null; then
  ufw allow 'Nginx Full'
fi

# --- CREATE NGINX CONFIG ---
NGINX_CONF="/etc/nginx/sites-available/${DOMAIN}"
cat > "${NGINX_CONF}" <<EOF
# Redirect all HTTP traffic to HTTPS
server {
    listen 80;
    server_name ${DOMAIN};
    return 301 https://\$host\$request_uri;
}

# HTTPS reverse proxy
server {
    listen 443 ssl http2;
    server_name ${DOMAIN};

    # Paths managed by Certbot
    ssl_certificate     /etc/letsencrypt/live/${DOMAIN}/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/${DOMAIN}/privkey.pem;

    # Security headers (optional but recommended)
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;

    location / {
        proxy_pass         http://127.0.0.1:${PORT};
        proxy_http_version 1.1;
        proxy_set_header   Upgrade \$http_upgrade;
        proxy_set_header   Connection keep-alive;
        proxy_set_header   Host \$host;
        proxy_cache_bypass \$http_upgrade;
        proxy_set_header   X-Real-IP \$remote_addr;
        proxy_set_header   X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto \$scheme;
    }
}
EOF

# --- ENABLE & VERIFY NGINX SITE ---
ln -sf "${NGINX_CONF}" /etc/nginx/sites-enabled/
nginx -t
systemctl reload nginx

# --- OBTAIN & INSTALL TLS CERTIFICATE ---
certbot --nginx -d "${DOMAIN}" --redirect --agree-tos --non-interactive --email admin@${DOMAIN}

echo
echo "✅  Setup complete!"
echo "🔒  Your site https://${DOMAIN}/ is now live with auto-renewed Let's Encrypt SSL."
