PORT ?= 8000

LAUNCHD_LABEL := dev.kotomka
LAUNCHD_DOMAIN := gui/$(shell id -u)
LAUNCHD_PLIST := launchd/$(LAUNCHD_LABEL).plist
LAUNCHD_PLIST_DEST := $(HOME)/Library/LaunchAgents/$(LAUNCHD_LABEL).plist

.DEFAULT_GOAL := serve
.PHONY: serve sync test launchd-install launchd-restart launchd-status launchd-uninstall

serve:
	uv run kotomka serve --port $(PORT)

sync:
	uv sync --extra dev

test:
	uv run pytest

launchd-install:
	cp $(LAUNCHD_PLIST) $(LAUNCHD_PLIST_DEST)
	-launchctl bootout $(LAUNCHD_DOMAIN)/$(LAUNCHD_LABEL) 2>/dev/null
	launchctl bootstrap $(LAUNCHD_DOMAIN) $(LAUNCHD_PLIST_DEST)

launchd-restart:
	launchctl kickstart -k $(LAUNCHD_DOMAIN)/$(LAUNCHD_LABEL)

launchd-status:
	launchctl print $(LAUNCHD_DOMAIN)/$(LAUNCHD_LABEL)

launchd-uninstall:
	-launchctl bootout $(LAUNCHD_DOMAIN)/$(LAUNCHD_LABEL) 2>/dev/null
	rm -f $(LAUNCHD_PLIST_DEST)
