#!/usr/bin/env bash
#
# Treblo Tag Explorer - one-command setup.
#
#   ./install.sh              interactive: asks the few questions that matter
#   ./install.sh --keyless    no API key, no signup, AI assist on
#   ./install.sh --offline    no AI, no network calls at all
#   ./install.sh --local      fully offline AI via Ollama (needs ~1.5GB)
#   ./install.sh --byok       your own API key (OpenAI, Groq, ...)
#   ./install.sh --uninstall  tear everything down and remove .env
#
# If Docker is missing, the script will attempt to install it automatically
# on common distributions (Ubuntu, Debian, Fedora, RHEL, CentOS, Arch,
# openSUSE, Alpine). If auto-install fails, it prints manual instructions
# and exits cleanly.
#
# Everything runs in Docker on YOUR machine. Nothing is sent anywhere except,
# in keyless/online modes, the optional AI assist request.
#
# Safe to re-run: it reuses an existing .env and never overwrites a token.

set -euo pipefail

cd "$(dirname "$0")"

BOLD=""; DIM=""; RED=""; GREEN=""; YELLOW=""; RESET=""
if [ -t 1 ]; then
  BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'
  GREEN=$'\033[32m'; GELLOW=$'\033[33m'; RESET=$'\033[0m'
fi

# ---------------------------------------------------------------------------
# Docker auto-install helpers
# ---------------------------------------------------------------------------

DISTRO=""
DETECTED=""

detect_distro() {
  if [ -f /etc/os-release ]; then
    . /etc/os-release
    DISTRO="${ID:-unknown}"
    DETECTED="${PRETTY_NAME:-$DISTRO}"
  elif [ -f /etc/redhat-release ]; then
    DISTRO="rhel"
    DETECTED=$(cat /etc/redhat-release)
  elif [ -f /etc/arch-release ]; then
    DISTRO="arch"
    DETECTED="Arch Linux"
  else
    DISTRO="unknown"
    DETECTED="$(uname -s)"
  fi
}

have_sudo() {
  command -v sudo >/dev/null 2>&1
}

run_root() {
  # Run a command as root. Uses sudo if not already root.
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  elif have_sudo; then
    sudo "$@"
  else
    echo "root privileges required but sudo is not installed." >&2
    return 1
  fi
}

add_user_to_docker_group() {
  local user
  user=$(id -un)
  if getent group docker >/dev/null 2>&1; then
    run_root usermod -aG docker "$user" 2>/dev/null || true
    ok "added $user to docker group (log out and back in for it to take effect)"
  fi
}

install_docker_ubuntu_debian() {
  run_root apt-get update -qq
  run_root apt-get install -y -qq ca-certificates curl gnupg
  install -d -m 0755 /etc/apt/keyrings 2>/dev/null || run_root mkdir -p /etc/apt/keyrings
  curl -fsSL "https://download.docker.com/linux/${DISTRO}/gpg" | run_root gpg --dearmor -o /etc/apt/keyrings/docker.gpg 2>/dev/null || true
  run_root chmod a+r /etc/apt/keyrings/docker.gpg 2>/dev/null || true
  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/${DISTRO} \
    $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
    | run_root tee /etc/apt/sources.list.d/docker.list > /dev/null
  run_root apt-get update -qq
  run_root apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
}

install_docker_fedora_rhel() {
  run_root dnf -y install dnf-plugins-core
  run_root dnf config-manager --add-repo "https://download.docker.com/linux/${DISTRO}/docker-ce.repo"
  run_root dnf -y install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
}

install_docker_arch() {
  run_root pacman -Sy --noconfirm docker docker-compose
}

install_docker_opensuse() {
  run_root zypper --non-interactive install docker docker-compose
  run_root systemctl enable --now docker
}

install_docker_alpine() {
  run_root apk add docker docker-compose
  run_root rc-update add docker boot 2>/dev/null || true
  run_root service docker start 2>/dev/null || true
}

install_docker() {
  say ""
  say "  ${YELLOW}Docker not found. Attempting to install it automatically...${RESET}"
  say "  ${DIM}Detected: ${DETECTED}${RESET}"
  say ""

  case "$DISTRO" in
    ubuntu|debian)
      install_docker_ubuntu_debian
      ;;
    fedora)
      install_docker_fedora_rhel
      ;;
    rhel|centos|rocky|almalinux|ol)
      install_docker_fedora_rhel
      ;;
    arch|manjaro|endeavouros)
      install_docker_arch
      ;;
    opensuse*|sles)
      install_docker_opensuse
      ;;
    alpine)
      install_docker_alpine
      ;;
    *)
      say "  ${RED}Cannot auto-install Docker on this distribution.${RESET}"
      say ""
      say "  Install Docker manually, then re-run this script:"
      say "    https://docs.docker.com/engine/install/"
      say ""
      say "  Or if you already installed Docker but the daemon isn't running:"
      say "    sudo systemctl start docker   (Linux)"
      say "    open -a Docker               (macOS)"
      say ""
      exit 1
      ;;
  esac

  if command -v docker >/dev/null 2>&1; then
    ok "Docker installed: $(docker --version 2>/dev/null | sed 's/Docker version //;s/,.*//')"
    # Start the daemon if it is not running.
    if ! docker info >/dev/null 2>&1; then
      say "  Starting Docker daemon..."
      if command -v systemctl >/dev/null 2>&1; then
        run_root systemctl enable --now docker 2>/dev/null || run_root service docker start 2>/dev/null || true
      elif command -v service >/dev/null 2>&1; then
        run_root service docker start 2>/dev/null || true
      fi
      sleep 2
    fi
    add_user_to_docker_group
    ok "Docker is ready. You may need to log out and back in for group changes."
  else
    die "Docker auto-install appeared to succeed but 'docker' is not in PATH. Try re-running."
  fi
}

# ---------------------------------------------------------------------------
# Ask helper
# ---------------------------------------------------------------------------

ask() {
  # ask <varname> <envvar> <prompt> <default>
  local __var="$1" __env="$2" __prompt="$3" __default="$4" __answer=""
  local __pre=""
  eval "__pre=\${$__env:-}"
  if [ -n "$__pre" ]; then
    printf '%s' "$__prompt" >&2
    printf '%s\n' "${DIM}${__pre}${RESET} ${DIM}(from ${__env})${RESET}" >&2
    eval "$__var=\$__pre"
    return 0
  fi
  if [ ! -t 0 ] && [ ! -e /dev/tty ]; then
    eval "$__var=\$__default"
    return 0
  fi
  printf '%s' "$__prompt" >&2
  if [ -e /dev/tty ]; then
    read -r __answer </dev/tty 2>/dev/null || __answer=""
  else
    __answer=""
  fi
  eval "$__var=\${__answer:-\$__default}"
}

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

say()  { printf '%s\n' "$*"; }
ok()   { printf '%s\n' "  ${GREEN}ok${RESET}   $*"; }
warn() { printf '%s\n' "  ${YELLOW}warn${RESET} $*"; }
die()  { printf '%s\n' "  ${RED}error${RESET} $*" >&2; exit 1; }
step() { printf '\n%s\n' "${BOLD}$*${RESET}"; }

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

show_help() {
  cat <<'HELP'
Treblo Tag Explorer - one-command setup.

Usage:
  ./install.sh [OPTION]

Options:
  (no option)       Interactive setup. Asks the few questions that matter.
  --keyless         AI assist via free public endpoint. No API key, no signup.
  --offline         No AI at all. Zero network calls. Fully self-contained.
  --local           Fully offline AI via Ollama (needs ~1.5GB model download).
  --byok            Bring your own API key (OpenAI, Groq, OpenRouter, ...).
  --uninstall       Stop and remove the container, image, and .env.
  --help            Show this help and exit.

Environment variables (override interactive prompts):
  APP_PORT=8123     Port to publish (default: 8000)
  BIND_ADDR=0.0.0.0 Bind address (default: 127.0.0.1 = this machine only)
  AUTH_TOKEN=...    Access token (auto-generated if BIND_ADDR is not 127.0.0.1)
  MODE=hybrid       Runtime mode: online | offline | hybrid
  AI_PROVIDER=...   keyless | compat | off
  INSTALL_AI_CHOICE=1|2|3|4  Skip the mode prompt
  INSTALL_SHARE=1|2           Skip the network share prompt
  INSTALL_HOST_IP=1.2.3.4     Pre-fill the host IP for LAN sharing

Examples:
  ./install.sh                          # interactive, keyless (default)
  ./install.sh --keyless                # same, non-interactive
  ./install.sh --offline                # no AI, no network
  ./install.sh --local                  # local Ollama AI
  ./install.sh --byok                   # will ask for your API key
  APP_PORT=8123 ./install.sh           # use port 8123
  ./install.sh --uninstall              # tear it all down

After install:
  docker compose logs -f                # follow the logs
  docker compose restart                # restart
  docker compose down                   # stop (data and config kept)
  ./install.sh --uninstall              # stop, remove container + .env

The app is bound to 127.0.0.1 by default. Only this machine can reach it.
To share on your LAN, set BIND_ADDR=0.0.0.0 and AUTH_TOKEN in .env.

HELP
  exit 0
}

# ---------------------------------------------------------------------------
# Uninstall
# ---------------------------------------------------------------------------

do_uninstall() {
  say ""
  say "${BOLD}Treblo Tag Explorer${RESET} - uninstall"
  say ""

  if ! command -v docker >/dev/null 2>&1; then
    die "docker is not installed. Nothing to uninstall."
  fi

  COMPOSE="docker compose"
  if ! docker compose version >/dev/null 2>&1; then
    if command -v docker-compose >/dev/null 2>&1; then
      COMPOSE="docker-compose"
    else
      die "docker compose is not available."
    fi
  fi

  step "Stopping and removing containers"
  $COMPOSE --profile local down --rmi local 2>/dev/null || true
  $COMPOSE down --rmi local 2>/dev/null || true
  $COMPOSE -f docker-compose.keyless.yml down --rmi local 2>/dev/null || true
  docker rm -f treblo-tag-explorer 2>/dev/null || true
  docker rmi treblo-tag-explorer:latest 2>/dev/null || true
  ok "containers and images removed"

  step "Removing data"
  if [ -f .env ]; then
    rm -f .env
    ok ".env removed"
  fi
  rm -f .env.backup.* 2>/dev/null
  ok "backups cleaned"

  step "Done"
  say "  The Treblo Tag Explorer has been removed from this machine."
  say "  To install again: ./install.sh"
  say ""
  exit 0
}

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------

MODE=""
UNINSTALL=0

for arg in "$@"; do
  case "$arg" in
    --keyless) MODE=keyless ;;
    --offline) MODE=offline ;;
    --local)   MODE=local ;;
    --byok)    MODE=byok ;;
    --uninstall) UNINSTALL=1 ;;
    -h|--help) show_help ;;
    *) die "unknown option: $arg  (try --help)" ;;
  esac
done

if [ "$UNINSTALL" -eq 1 ]; then
  do_uninstall
fi

say ""
say "${BOLD}Treblo Tag Explorer${RESET} - setup"
say "${DIM}Self-hosted. Runs on your machine, in your Docker.${RESET}"

# ---------------------------------------------------------------------------
# 1. Check the prerequisites
# ---------------------------------------------------------------------------

step "1. Checking what you have"

detect_distro

if ! command -v docker >/dev/null 2>&1; then
  say "  ${YELLOW}Docker is not installed.${RESET}"
  say ""
  ANSWER=""
  if [ -e /dev/tty ]; then
    printf "  Attempt to install Docker automatically? [Y/n]: " >&2
    read -r ANSWER </dev/tty 2>/dev/null || ANSWER=""
  fi
  case "${ANSWER:-Y}" in
    y|Y|yes|YES|"")
      install_docker
      ;;
    *)
      say ""
      say "  Install Docker manually, then re-run this script:"
      say "    https://docs.docker.com/engine/install/"
      say ""
      exit 1
      ;;
  esac
fi

if ! command -v docker >/dev/null 2>&1; then
  die "docker is not installed."
fi
ok "docker $(docker --version 2>/dev/null | sed 's/Docker version //;s/,.*//')"

if docker compose version >/dev/null 2>&1; then
  COMPOSE="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE="docker-compose"
else
  die "docker compose is not available.
     It ships with Docker Desktop, and with modern Docker Engine it is the
     'docker-compose-plugin' package. Install it and re-run."
fi
ok "compose ($COMPOSE)"

if ! docker info >/dev/null 2>&1; then
  die "cannot talk to the Docker daemon.
     Start Docker, and on Linux make sure your user is in the docker group:
       sudo usermod -aG docker \$USER   (then log out and back in)"
fi
ok "docker daemon reachable"

# Port availability. If our own container already holds it, that is fine: a
# re-run is meant to be safe and should just reconfigure and restart.
PORT="${APP_PORT:-8000}"
if command -v ss >/dev/null 2>&1 && ss -tln 2>/dev/null | grep -q ":${PORT} "; then
  if docker ps --filter "name=treblo-tag-explorer" --format '{{.Ports}}' 2>/dev/null \
       | grep -q ":${PORT}->"; then
    ok "port ${PORT} already held by this app (re-run)"
  else
    warn "port ${PORT} is already in use by something else. Either stop it, or"
    warn "pick another port:  APP_PORT=8123 ./install.sh"
    die "choose another port and re-run"
  fi
else
  ok "port ${PORT} is free"
fi

# ---------------------------------------------------------------------------
# 2. Pick a mode
# ---------------------------------------------------------------------------

step "2. Choosing how it runs"

if [ -z "$MODE" ]; then
  if [ -f .env ] && grep -q '^INSTALL_MODE=' .env 2>/dev/null; then
    MODE=$(grep '^INSTALL_MODE=' .env | cut -d= -f2-)
    ok "reusing mode from .env: ${MODE}"
  else
    say ""
    say "  How should the optional AI work?"
    say ""
    say "    ${BOLD}1${RESET}) keyless  - free public AI, no API key, no signup  ${DIM}(default)${RESET}"
    say "    ${BOLD}2${RESET}) offline  - no AI at all, zero network calls"
    say "    ${BOLD}3${RESET}) local    - AI runs on your machine via Ollama, ~1.5GB download"
    say "    ${BOLD}4${RESET}) byok     - your own API key (OpenAI, Groq, OpenRouter...)"
    say ""
    say "  The search engine itself always works offline. The AI only re-ranks"
    say "  results it is given, so any of these are fully usable."
    say ""
    ask choice INSTALL_AI_CHOICE "  Choice [1]: " "1"
    case "${choice:-1}" in
      1|"") MODE=keyless ;;
      2)    MODE=offline ;;
      3)    MODE=local ;;
      4)    MODE=byok ;;
      *)    die "pick 1, 2, 3 or 4" ;;
    esac
  fi
fi
ok "mode: ${MODE}"

# ---------------------------------------------------------------------------
# 3. Security
# ---------------------------------------------------------------------------

step "3. Access control"

TOKEN=""
EXPOSE="127.0.0.1"

if [ -f .env ]; then
  EXISTING_TOKEN=$(grep '^AUTH_TOKEN=' .env 2>/dev/null | cut -d= -f2- || true)
  if [ -n "$EXISTING_TOKEN" ]; then
    TOKEN="$EXISTING_TOKEN"
    ok "reusing the existing AUTH_TOKEN"
  fi
fi

# If BIND_ADDR is already set in .env, use that as the default.
if [ -f .env ] && grep -q '^BIND_ADDR=' .env 2>/dev/null; then
  EXISTING_BIND=$(grep '^BIND_ADDR=' .env | cut -d= -f2-)
  if [ -n "$EXISTING_BIND" ] && [ "$EXISTING_BIND" != "127.0.0.1" ]; then
    EXPOSE="$EXISTING_BIND"
  fi
fi

# Allow env override for BIND_ADDR without a prompt.
if [ -n "${BIND_ADDR:-}" ]; then
  EXPOSE="$BIND_ADDR"
fi

say ""
say "  Who should be able to open it?"
say ""
say "    ${BOLD}1${RESET}) just this machine  ${DIM}(default, safest, no token needed)${RESET}"
say "    ${BOLD}2${RESET}) my whole network  ${DIM}(needs a token)${RESET}"
say ""

# Skip the prompt if BIND_ADDR was explicitly set or INSTALL_SHARE is set.
if [ -n "${INSTALL_SHARE:-}" ]; then
  netchoice="$INSTALL_SHARE"
elif [ -n "${BIND_ADDR:-}" ] && [ "$BIND_ADDR" != "127.0.0.1" ]; then
  netchoice="2"
  ok "BIND_ADDR=${BIND_ADDR} set via environment"
elif [ -n "${BIND_ADDR:-}" ] && [ "$BIND_ADDR" = "127.0.0.1" ]; then
  netchoice="1"
  ok "BIND_ADDR=127.0.0.1 set via environment"
else
  ask netchoice INSTALL_SHARE "  Choice [1]: " "1"
fi

ALLOWED_HOSTS=""
if [ "${netchoice:-1}" = "2" ]; then
  EXPOSE="0.0.0.0"
  if [ -z "$TOKEN" ]; then
    if command -v python3 >/dev/null 2>&1; then
      TOKEN=$(python3 -c 'import secrets;print(secrets.token_urlsafe(32))')
    else
      TOKEN=$(docker run --rm python:3.12-slim python -c \
        'import secrets;print(secrets.token_urlsafe(32))')
    fi
  fi
  ok "generated an access token"

  # Pre-fill host IP from env if provided.
  if [ -n "${INSTALL_HOST_IP:-}" ]; then
    hostip="$INSTALL_HOST_IP"
    ok "will accept requests for ${hostip}"
  else
    say ""
    say "  Which address will people use? Find it with ${BOLD}ip addr${RESET} (Linux)"
    say "  or ${BOLD}ipconfig${RESET} (Windows). Example: 192.168.1.50"
    ask hostip INSTALL_HOST_IP "  Address [leave blank to add later]: " ""
  fi
  if [ -n "$hostip" ]; then
    ALLOWED_HOSTS="$hostip"
    ok "will accept requests for ${hostip}"
  else
    warn "no address given. Requests by IP will be refused until you set"
    warn "ALLOWED_HOSTS in .env and restart. Loopback still works."
  fi
else
  ok "bound to 127.0.0.1 - only this machine can reach it"
fi

# ---------------------------------------------------------------------------
# 3b. API key for --byok mode
# ---------------------------------------------------------------------------

BYOK_API_KEY=""
BYOK_BASE_URL="https://api.openai.com/v1"
BYOK_MODEL="gpt-4o-mini"

if [ "$MODE" = "byok" ]; then
  step "3b. Your API key"

  if [ -f .env ] && grep -q '^OPENAI_API_KEY=' .env 2>/dev/null; then
    EXISTING_KEY=$(grep '^OPENAI_API_KEY=' .env | cut -d= -f2-)
    if [ -n "$EXISTING_KEY" ]; then
      BYOK_API_KEY="$EXISTING_KEY"
      ok "reusing the existing OPENAI_API_KEY from .env"
    fi
  fi

  if [ -z "$BYOK_API_KEY" ]; then
    if [ -n "${OPENAI_API_KEY:-}" ]; then
      BYOK_API_KEY="$OPENAI_API_KEY"
      ok "using OPENAI_API_KEY from environment"
    else
      say ""
      say "  Enter your API key (OpenAI, Groq, OpenRouter, Together, etc)."
      say "  ${DIM}It is stored in .env with mode 600. Never printed or logged.${RESET}"
      say ""
      if [ -e /dev/tty ]; then
        printf "  API key: " >&2
        read -rs BYOK_API_KEY </dev/tty 2>/dev/null || BYOK_API_KEY=""
        echo "" >&2
      fi
      if [ -z "$BYOK_API_KEY" ]; then
        ask key_input OPENAI_API_KEY "  API key: " ""
        BYOK_API_KEY="$key_input"
      fi
      if [ -z "$BYOK_API_KEY" ]; then
        die "an API key is required for --byok mode"
      fi
      ok "API key received"
    fi
  fi

  # Ask for base URL (allows non-OpenAI providers).
  if [ -z "${OPENAI_BASE_URL:-}" ] || [ "$OPENAI_BASE_URL" = "https://api.openai.com/v1" ]; then
    ask base_url_input OPENAI_BASE_URL "  Base URL [${BYOK_BASE_URL}]: " "$BYOK_BASE_URL"
    BYOK_BASE_URL="${base_url_input:-$BYOK_BASE_URL}"
  else
    BYOK_BASE_URL="$OPENAI_BASE_URL"
  fi
  ok "base URL: ${BYOK_BASE_URL}"

  ask byok_model OPENAI_MODEL "  Model [${BYOK_MODEL}]: " "$BYOK_MODEL"
  BYOK_MODEL="${byok_model:-$BYOK_MODEL}"
  ok "model: ${BYOK_MODEL}"
fi

# ---------------------------------------------------------------------------
# 4. Write .env
# ---------------------------------------------------------------------------

step "4. Writing configuration"

if [ -f .env ]; then
  cp .env ".env.backup.$(date +%s)"
  ok "backed up the previous .env"
fi

{
  echo "# Treblo Tag Explorer configuration"
  echo "# Written by install.sh. Edit freely, then: $COMPOSE up -d"
  echo ""
  echo "INSTALL_MODE=${MODE}"
  echo "APP_PORT=${PORT}"
  echo ""
  echo "# Where the port is published. 127.0.0.1 = this machine only."
  echo "BIND_ADDR=${EXPOSE}"
  echo ""
  echo "# Access token. Required whenever BIND_ADDR is not 127.0.0.1."
  echo "AUTH_TOKEN=${TOKEN}"
  echo "AUTH_REQUIRED=$([ -n "$TOKEN" ] && echo 1 || echo 0)"
  echo ""
  echo "# Names this instance is reached by. Inside a container the app cannot"
  echo "# see your LAN IP, so any non-loopback address must be listed here."
  echo "ALLOWED_HOSTS=${ALLOWED_HOSTS}"
  echo "ALLOWED_ORIGINS="
  echo ""
  case "$MODE" in
    offline)
      echo "MODE=offline"
      echo "AI_ENABLED=0"
      echo "AI_PROVIDER=off"
      ;;
    local)
      echo "MODE=hybrid"
      echo "AI_ENABLED=1"
      echo "AI_PROVIDER=compat"
      echo "OPENAI_BASE_URL=http://ollama:11434/v1"
      echo "OPENAI_MODEL=llama3.2:1b"
      echo "LOCAL_MODEL=llama3.2:1b"
      ;;
    byok)
      echo "MODE=hybrid"
      echo "AI_ENABLED=1"
      echo "AI_PROVIDER=compat"
      echo "OPENAI_BASE_URL=${BYOK_BASE_URL}"
      echo "OPENAI_MODEL=${BYOK_MODEL}"
      echo "OPENAI_API_KEY=${BYOK_API_KEY}"
      ;;
    *)
      echo "MODE=hybrid"
      echo "AI_ENABLED=1"
      echo "AI_PROVIDER=keyless"
      echo "KEYLESS_MODEL=openai-fast"
      ;;
  esac
} > .env

chmod 600 .env
ok ".env written (mode 600, readable only by you)"

# ---------------------------------------------------------------------------
# 5. Build and start
# ---------------------------------------------------------------------------

step "5. Building and starting (first build takes a couple of minutes)"

if [ "$MODE" = "local" ]; then
  $COMPOSE --profile local up -d --build
else
  $COMPOSE up -d --build
fi

# ---------------------------------------------------------------------------
# 6. Wait for it to be healthy
# ---------------------------------------------------------------------------

step "6. Waiting for it to come up"

HEALTH="http://127.0.0.1:${PORT}/api/health"
UP=""
for i in $(seq 1 45); do
  if curl -fsS -m 3 "$HEALTH" >/dev/null 2>&1; then UP=1; break; fi
  sleep 2
done

if [ -z "$UP" ]; then
  say ""
  warn "it did not become healthy in 90 seconds. Last 30 log lines:"
  say ""
  $COMPOSE logs --tail=30 2>&1 | sed 's/^/    /'
  die "start failed. The logs above say why."
fi
ok "healthy"

# ---------------------------------------------------------------------------
# 7. Prove it actually works
# ---------------------------------------------------------------------------

step "7. Proving it actually works"

RESULT=$(curl -fsS -m 30 -X POST "http://127.0.0.1:${PORT}/api/search" \
  -H 'Content-Type: application/json' \
  -d '{"query":"sad slow piano for a rainy night","limit":5}' 2>/dev/null || echo "")

if [ -z "$RESULT" ]; then
  warn "the health check passed but a search did not. Check the logs:"
  warn "  $COMPOSE logs --tail=30"
  exit 1
fi

if command -v python3 >/dev/null 2>&1; then
  echo "$RESULT" | python3 -c '
import json,sys
try:
    d = json.load(sys.stdin)
except Exception:
    print("  (could not parse the response, but the server answered)"); sys.exit(0)
tags = [t["tag"] for t in d.get("tags", [])]
print("  query :", d.get("query"))
print("  tags  :", ", ".join(tags[:6]) if tags else "(none)")
print("  engine:", d.get("deterministic_count"), "hits in", d.get("elapsed_ms"), "ms")
ai = d.get("ai", {})
if ai.get("used"):
    print("  ai    : used,", ai.get("model"))
elif ai.get("error"):
    print("  ai    : not used (", ai.get("error"), ") - engine results still returned")
if not tags:
    print("  WARNING: no tags came back. The tag file may be missing or empty.")
    sys.exit(1)
'
else
  say "  server answered. Tags:"
  echo "$RESULT" | head -c 300 | sed 's/^/    /'
  say ""
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

step "Done"

if [ -n "$TOKEN" ]; then
  say "  Open:  ${BOLD}http://127.0.0.1:${PORT}${RESET}"
  if [ "$EXPOSE" != "127.0.0.1" ]; then
    say "  Others on your network: ${BOLD}http://<your-ip>:${PORT}${RESET}"
    if [ -n "$ALLOWED_HOSTS" ]; then
      say "  They will need the token shown below."
    fi
  fi
  say ""
  say "  ${YELLOW}Access token (save it - needed to log in):${RESET}"
  say "    ${BOLD}${TOKEN}${RESET}"
  say ""
  say "  ${DIM}It is also in .env. The UI shows a token box on first load.${RESET}"
else
  say "  Open: ${BOLD}http://127.0.0.1:${PORT}${RESET}"
fi

say ""
say "  Useful commands:"
say "    $COMPOSE logs -f          follow the logs"
say "    $COMPOSE restart          restart it"
say "    $COMPOSE down             stop it"
say "    ./install.sh --uninstall  remove everything"
say ""
say "  ${DIM}Full documentation: README.md${RESET}"
say ""
