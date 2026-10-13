# Treblo Tag Explorer - shortcuts
# Everything runs on the Python standard library. No install step.

.PHONY: help up keyless local byok offline down logs test test-fast stats run cli \
        shell token related interpret sec clean ps

help:
	@echo "Treblo Tag Explorer v2"
	@echo ""
	@echo "  --- run it ---"
	@echo "  make up        start with Docker (keyless AI, no API key)"
	@echo "  make keyless   same thing, minimal compose file"
	@echo "  make local     start + local Ollama (fully offline AI)"
	@echo "  make byok      start with your own API key (needs OPENAI_API_KEY)"
	@echo "  make offline   start with NO AI and zero network calls"
	@echo "  make run       run directly, no Docker (localhost:8000)"
	@echo "  make down      stop everything"
	@echo "  make logs      follow container logs"
	@echo "  make ps        show container status"
	@echo ""
	@echo "  --- check it ---"
	@echo "  make test      run the full test suite"
	@echo "  make sec       verify the container hardening"
	@echo "  make stats     vocabulary and graph breakdown"
	@echo "  make token     generate an AUTH_TOKEN"
	@echo ""
	@echo "  --- use it ---"
	@echo "  make cli Q='sad slow piano for a rainy night'"
	@echo "  make interpret Q='music for the gym heavy and fast'"
	@echo "  make related T='dark ambient'"
	@echo "  make shell     python REPL with the index loaded"
	@echo ""
	@echo "  --- tidy ---"
	@echo "  make clean     remove containers and the image"

up:
	docker compose up -d --build
	@echo "-> http://localhost:8000  (bound to 127.0.0.1)"

keyless:
	docker compose -f docker-compose.keyless.yml up -d --build
	@echo "-> http://localhost:8000  (no API key needed)"

local:
	docker compose --profile local up -d --build
	@echo "-> http://localhost:8000  (pick provider 'own key / local', model llama3.2:1b)"

byok:
	@test -n "$$OPENAI_API_KEY" || (echo "set OPENAI_API_KEY first"; exit 1)
	AI_PROVIDER=compat docker compose up -d --build
	@echo "-> http://localhost:8000"

offline:
	MODE=offline AI_ENABLED=0 docker compose up -d --build
	@echo "-> http://localhost:8000  (no AI, no network calls)"

down:
	-docker compose down
	-docker compose -f docker-compose.keyless.yml down
	-docker rm -f treblo-off treblo-ssrf 2>/dev/null

logs:
	docker compose logs -f --tail=50

ps:
	@docker ps --filter name=treblo --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

test:
	python3 -m unittest discover -s tests -v

test-fast:
	python3 -m unittest discover -s tests 2>&1 | tail -5

sec:
	@echo "container hardening:"
	@echo -n "  user        : "; docker exec treblo-tag-explorer id 2>/dev/null || echo "(not running)"
	@echo -n "  read-only   : "; docker inspect treblo-tag-explorer --format '{{.HostConfig.ReadonlyRootfs}}' 2>/dev/null
	@echo -n "  cap drop    : "; docker inspect treblo-tag-explorer --format '{{.HostConfig.CapDrop}}' 2>/dev/null
	@echo -n "  no-new-priv : "; docker inspect treblo-tag-explorer --format '{{.HostConfig.SecurityOpt}}' 2>/dev/null
	@echo -n "  mem limit   : "; docker inspect treblo-tag-explorer --format '{{.HostConfig.Memory}}' 2>/dev/null
	@echo -n "  port bind   : "; docker inspect treblo-tag-explorer --format '{{json .NetworkSettings.Ports}}' 2>/dev/null
	@echo ""
	@echo "security headers:"
	@curl -sS -D- -o /dev/null 127.0.0.1:8000/ 2>/dev/null | grep -iE 'content-security|x-frame|x-content|referrer|permissions' | sed 's/^/  /' || echo "  (not running)"

stats:
	python3 app/cli.py --stats

token:
	@python3 app/cli.py --token

run:
	python3 app/server.py

cli:
	@test -n "$(Q)" || (echo "usage: make cli Q='sad slow piano'"; exit 1)
	python3 app/cli.py "$(Q)"

interpret:
	@test -n "$(Q)" || (echo "usage: make interpret Q='music for the gym'"; exit 1)
	python3 app/cli.py --interpret "$(Q)"

related:
	@test -n "$(T)" || (echo "usage: make related T='dark ambient'"; exit 1)
	python3 app/cli.py --related "$(T)"

shell:
	python3 -i -c "import sys; sys.path.insert(0,'app'); from tags import TagIndex; from pipeline import search; ix=TagIndex('data/treblo_tags_v3.txt'); print('index loaded:', len(ix), 'tags; derived:', ix.has_derived); print('try: search(ix, \"sad piano\")')"

clean:
	-docker compose down --rmi local 2>/dev/null
	-docker compose -f docker-compose.keyless.yml down --rmi local 2>/dev/null
	-docker rmi treblo-tag-explorer:latest 2>/dev/null
	@echo "cleaned"
