#!/usr/bin/env bash
# Install an isolated static directory and a periodic read-only discovery job.
set -euo pipefail
directory_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
release_id="$(date -u +%Y%m%dT%H%M%SZ)"
remote_release="/opt/project-directory/releases/$release_id"
ssh_options=(-T -q -o BatchMode=yes -o ConnectTimeout=15)

ssh "${ssh_options[@]}" webhost "mkdir -p '$remote_release'"
tar -C "$directory_root" --exclude=artifacts --exclude=__pycache__ -czf - . | \
    ssh "${ssh_options[@]}" webhost "tar -xzf - -C '$remote_release'"

ssh "${ssh_options[@]}" webhost bash -s -- "$release_id" <<'REMOTE'
set -euo pipefail
release_id="$1"
release="/opt/project-directory/releases/$release_id"
backup="/var/backups/project-directory/$release_id"
mkdir -p "$backup" /var/www/project-directory/data /var/lib/project-directory /etc/project-directory
exec 9>/var/lib/project-directory/deploy.lock
flock -x 9
files=(/etc/nginx/conf.d/zzz-liveadboard.conf /etc/nginx/conf.d/zr-zhang25.cn.conf
       /etc/nginx/snippets/project-directory-route.conf /etc/project-directory/config.json
       /etc/systemd/system/project-directory.service /etc/systemd/system/project-directory.timer)
for file in "${files[@]}"; do
    if [[ -f "$file" ]]; then cp -p --parents "$file" "$backup"; fi
done
previous_app="$(readlink /opt/project-directory/current || true)"
previous_site="$(readlink /var/www/project-directory/current || true)"
timer_enabled="$(systemctl is-enabled project-directory.timer 2>/dev/null || true)"
timer_active="$(systemctl is-active project-directory.timer 2>/dev/null || true)"

restore_link() {
    local target="$1" link="$2"
    if [[ -n "$target" ]]; then ln -sfn "$target" "$link"; else rm -f "$link"; fi
}
rollback() {
    trap - ERR
    systemctl stop project-directory.timer project-directory.service 2>/dev/null || true
    systemctl disable project-directory.timer 2>/dev/null || true
    for file in "${files[@]}"; do
        if [[ -f "$backup$file" ]]; then cp -p "$backup$file" "$file"; else rm -f "$file"; fi
    done
    restore_link "$previous_app" /opt/project-directory/current
    restore_link "$previous_site" /var/www/project-directory/current
    systemctl daemon-reload
    if [[ "$timer_enabled" == enabled ]]; then systemctl enable project-directory.timer; fi
    if [[ "$timer_active" == active ]]; then systemctl start project-directory.timer; fi
    nginx -t && systemctl reload nginx
    printf 'Project directory deployment failed; previous configuration restored.\n' >&2
}
trap rollback ERR

systemctl stop project-directory.timer 2>/dev/null || true
chmod -R a+rX "$release"
for link in /opt/project-directory/current /var/www/project-directory/current; do
    ln -s "$release" "$link.next-$release_id"
    mv -Tf "$link.next-$release_id" "$link"
done
if [[ ! -f /etc/project-directory/config.json ]]; then
    install -m 644 "$release/config.json" /etc/project-directory/config.json
fi
install -m 644 "$release/deploy/project-directory-route.conf" /etc/nginx/snippets/project-directory-route.conf
install -m 644 "$release/deploy/project-directory.service" /etc/systemd/system/project-directory.service
install -m 644 "$release/deploy/project-directory.timer" /etc/systemd/system/project-directory.timer
python3 - <<'PY'
from pathlib import Path
anchor = '    include /etc/nginx/snippets/liveadboard-route.conf;'
include = '    include /etc/nginx/snippets/project-directory-route*.conf;'
for filename in ('zzz-liveadboard.conf', 'zr-zhang25.cn.conf'):
    path = Path('/etc/nginx/conf.d') / filename
    source = path.read_text()
    if include in source:
        continue
    if source.count(anchor) != 1:
        raise SystemExit('Nginx layout changed: expected one leaderboard include in ' + filename)
    path.write_text(source.replace(anchor, anchor + '\n' + include, 1))
PY
nginx -t
systemctl daemon-reload
systemctl start project-directory.service
systemctl reload nginx
ready=false
for attempt in {1..10}; do
    if curl --noproxy '*' --fail --silent --show-error --max-time 5 -H 'Host: 49.232.30.3' \
         http://127.0.0.1/projects/ -o "$backup/health.html" \
       && cmp -s "$release/static/index.html" "$backup/health.html" \
       && curl --noproxy '*' --fail --silent --show-error --max-time 5 \
         --resolve zr-zhang25.cn:443:127.0.0.1 https://zr-zhang25.cn/projects/ -o "$backup/health.html" \
       && cmp -s "$release/static/index.html" "$backup/health.html" \
       && curl --noproxy '*' --fail --silent --show-error --max-time 5 -H 'Host: 49.232.30.3' \
         http://127.0.0.1/projects/projects.json -o "$backup/health.json" \
       && cmp -s /var/www/project-directory/data/projects.json "$backup/health.json"; then
       ready=true
       break
    fi
    sleep 1
done
"$ready"
systemctl enable --now project-directory.timer
printf 'Published: http://49.232.30.3/projects/\nHTTPS: https://zr-zhang25.cn/projects/\nRelease: %s\n' "$release"
cat /var/www/project-directory/data/projects.json
REMOTE
