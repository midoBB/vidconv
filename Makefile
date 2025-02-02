.PHONY: configure build install run clean all

PREFIX ?= ~/.local
all: configure build

configure: requirements.txt
	@python -m venv .venv
	@pip3 install -r requirements.txt > /dev/null
	@echo "Configured!"

generate-completions:
	@mkdir -p dist
	@_VIDCONV_COMPLETE=bash_source ./vidconv > dist/vidconv.bash
	@_VIDCONV_COMPLETE=zsh_source ./vidconv > dist/_vidconv.zsh
	@echo "Generated shell completions"

build: main.py
	@pyinstaller --onefile main.py --log-level=FATAL
	@cp dist/main vidconv
	@$(MAKE) generate-completions
	@echo "Built successfully!"

install: build
	@install -Dm755 vidconv $(PREFIX)/bin/vidconv
	@install -Dm644 dist/vidconv.bash $(PREFIX)/share/bash-completion/completions/vidconv
	@install -Dm644 dist/_vidconv.zsh ~/.zsh/completions/_vidconv
	@echo "Installed!"

run: build
	@./vidconv

clean:
	@rm -rf vidconv dist build main.spec
	@rm -rf .venv
	@rm -rf __pycache__
	@rm -rf *.pyc
	@echo "Cleaned!"
