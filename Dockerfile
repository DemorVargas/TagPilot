# Treblo Tag Explorer
#
# Standard-library-only Python: nothing is pip-installed, so there is no
# dependency tree to patch and no supply-chain surface to watch.
#
# Security choices in this image:
#   - non-root user, fixed uid/gid
#   - no shell tools beyond what the base image ships (no curl, no wget)
#   - read-only-friendly: the app writes nothing to disk at runtime
#   - healthcheck hits 127.0.0.1 explicitly (localhost may resolve to ::1)
FROM python:3.12-slim

LABEL org.opencontainers.image.title="Treblo Tag Explorer" \
      org.opencontainers.image.description="Turn plain-English descriptions into real tags. Works fully offline; optional keyless AI." \
      org.opencontainers.image.version="2.0.0" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    HOST=0.0.0.0 \
    MODE=hybrid \
    AI_ENABLED=1 \
    AI_PROVIDER=keyless \
    TAG_FILE=/app/data/treblo_tags_v3.txt \
    MAX_THREADS=24 \
    RATE_LIMIT_PER_MIN=60 \
    RATE_LIMIT_BURST=15 \
    AI_RATE_LIMIT_PER_MIN=12 \
    MAX_BODY_BYTES=65536 \
    MAX_QUERY_CHARS=500 \
    SOCKET_TIMEOUT=30

WORKDIR /app

# No requirements to install: the app is stdlib only. The file is copied so a
# future optional extra has an obvious home, and the layer stays cacheable.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --disable-pip-version-check -r /app/requirements.txt

COPY app/ /app/app/
COPY web/ /app/web/
COPY data/ /app/data/

# Fixed uid/gid so bind-mount permissions are predictable, and a nologin shell.
RUN groupadd --gid 10001 explorer \
    && useradd --uid 10001 --gid 10001 --no-create-home \
               --shell /usr/sbin/nologin explorer \
    && chown -R explorer:explorer /app \
    && chmod -R a-w /app/data /app/app /app/web

USER 10001:10001

EXPOSE 8000

# 127.0.0.1 rather than localhost: on hosts where localhost resolves to ::1
# first, this check would fail against a healthy IPv4-bound server.
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD ["python3", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status==200 else 1)"]

CMD ["python3", "app/server.py"]
