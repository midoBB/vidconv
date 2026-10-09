.PHONY: build test lint install clean all completions

PREFIX ?= ~/.local
all: build

build:
	@cargo build --release --locked
	@echo "Built successfully!"

test:
	@cargo test --locked

lint:
	@cargo fmt --check
	@cargo clippy --all-targets --locked -- -D warnings

completions: build
	@mkdir -p dist
	@./target/release/vidconv --completions bash > dist/vidconv.bash
	@./target/release/vidconv --completions zsh > dist/_vidconv.zsh
	@echo "Generated shell completions"

install: completions
	@install -Dm755 target/release/vidconv $(PREFIX)/bin/vidconv
	@install -Dm644 dist/vidconv.bash $(PREFIX)/share/bash-completion/completions/vidconv
	@install -Dm644 dist/_vidconv.zsh $(PREFIX)/share/zsh/site-functions/_vidconv
	@echo "Installed to $(PREFIX)!"

clean:
	@cargo clean
	@rm -rf dist
	@echo "Cleaned!"
