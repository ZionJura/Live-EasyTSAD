#!/usr/bin/env bash
# Publish the static build to the configured SSH host. Keep releases for rollback.
set -euo pipefail

frontend_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" != "--skip-build" ]]; then
    (cd "$frontend_dir" && npm run build)
fi
test -f "$frontend_dir/dist/index.html"

release_id="$(date -u +%Y%m%dT%H%M%SZ)"
remote_release="/var/www/liveadboard/releases/$release_id"
ssh_options=(-T -q -o BatchMode=yes -o ConnectTimeout=15)

ssh "${ssh_options[@]}" webhost "mkdir -p '$remote_release' /var/www/liveadboard/config-backups"
tar -C "$frontend_dir/dist" -czf - . | ssh "${ssh_options[@]}" webhost "tar -xzf - -C '$remote_release'"
scp -q -o BatchMode=yes "$frontend_dir/deploy/webhost.nginx.conf" "webhost:$remote_release.nginx.conf"
scp -q -o BatchMode=yes "$frontend_dir/deploy/liveadboard-route.nginx.conf" "webhost:$remote_release.route.conf"

ssh "${ssh_options[@]}" webhost bash -s -- "$release_id" <<'REMOTE'
set -euo pipefail
release_id="$1"
site_root=/var/www/liveadboard
release_path="$site_root/releases/$release_id"
config_path=/etc/nginx/conf.d/zzz-liveadboard.conf
backup_path="$site_root/config-backups/$release_id.conf"
tls_config_path=/etc/nginx/conf.d/zr-zhang25.cn.conf
route_path=/etc/nginx/snippets/liveadboard-route.conf

exec 9>"$site_root/deploy.lock"
flock -x 9
previous_release="$(readlink "$site_root/current" || true)"
cp -p "$tls_config_path" "$backup_path.tls"
had_config=false
if [[ -f "$config_path" ]]; then
    cp -p "$config_path" "$backup_path"
    had_config=true
fi
had_route=false
if [[ -f "$route_path" ]]; then
    cp -p "$route_path" "$backup_path.route"
    had_route=true
fi

rollback() {
    trap - ERR
    if [[ -n "$previous_release" ]]; then
        ln -sfn "$previous_release" "$site_root/.rollback"
        mv -Tf "$site_root/.rollback" "$site_root/current"
    else
        rm -f "$site_root/current"
    fi
    if "$had_config"; then
        cp -p "$backup_path" "$config_path"
    else
        rm -f "$config_path"
    fi
    cp -p "$backup_path.tls" "$tls_config_path"
    if "$had_route"; then
        cp -p "$backup_path.route" "$route_path"
    else
        rm -f "$route_path"
    fi
    nginx -t && systemctl reload nginx
    printf 'Deployment failed; restored the previous release and configuration.\n' >&2
}
trap rollback ERR

chmod -R a+rX "$release_path"
ln -s "$release_path" "$site_root/.current-$release_id"
mv -Tf "$site_root/.current-$release_id" "$site_root/current"
install -m 644 "$release_path.nginx.conf" "$config_path"
install -D -m 644 "$release_path.route.conf" "$route_path"
python3 - "$tls_config_path" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
content = path.read_text()
include = "    include /etc/nginx/snippets/liveadboard-route.conf;\n"
if include not in content:
    anchor = "    root /usr/share/nginx/html;\n"
    if content.count(anchor) != 1 or "listen 443 ssl;" not in content:
        raise SystemExit("HTTPS virtual host layout changed; refusing to guess where to add the route")
    path.write_text(content.replace(anchor, include + "\n" + anchor, 1))
PY
nginx -t
systemctl reload nginx

# Allow the new Nginx workers to become ready after the graceful reload.
curl --noproxy '*' --fail --silent --show-error --retry 5 --retry-connrefused --retry-delay 1 \
    --max-time 10 http://127.0.0.1:8766/ -o "$site_root/.health-$release_id.html"
cmp "$release_path/index.html" "$site_root/.health-$release_id.html"
route_ready=false
for attempt in {1..10}; do
    if curl --noproxy '*' --fail --silent --show-error --max-time 5 -H 'Host: 49.232.30.3' \
        http://127.0.0.1/liveadboard/ -o "$site_root/.health-$release_id.html" \
        && cmp -s "$release_path/index.html" "$site_root/.health-$release_id.html" \
        && curl --noproxy '*' --fail --silent --show-error --max-time 5 \
            --resolve zr-zhang25.cn:443:127.0.0.1 https://zr-zhang25.cn/liveadboard/ \
            -o "$site_root/.health-$release_id.html" \
        && cmp -s "$release_path/index.html" "$site_root/.health-$release_id.html"; then
        route_ready=true
        break
    fi
    sleep 1
done
"$route_ready"
rm -f "$site_root/.health-$release_id.html"

printf '%s\n' "$previous_release" > "$site_root/previous-release"
printf 'Published release: %s\nURL: http://49.232.30.3/liveadboard/\nHTTPS: https://zr-zhang25.cn/liveadboard/\nBackend: 127.0.0.1:8766\n' "$release_path"
REMOTE

# Check the public route from the deployment machine as well.
curl --noproxy '*' --fail --silent --show-error --location --max-time 20 \
    http://49.232.30.3/liveadboard -o "$frontend_dir/artifacts-deploy-check.html"
cmp "$frontend_dir/dist/index.html" "$frontend_dir/artifacts-deploy-check.html"
curl --noproxy '*' --fail --silent --show-error --location --max-time 20 \
    https://zr-zhang25.cn/liveadboard -o "$frontend_dir/artifacts-deploy-check.html"
cmp "$frontend_dir/dist/index.html" "$frontend_dir/artifacts-deploy-check.html"
rm -f "$frontend_dir/artifacts-deploy-check.html"
