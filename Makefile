release:
	cargo build --release

debug:
	cargo build

pgo-build:
	cargo pgo build

pgo-instrument: pgo-build
	./target/x86_64-unknown-linux-gnu/release/pytorch-tests

pgo-bolt-build: pgo-instrument
	cargo pgo bolt build --with-pgo

bolt-instrument: pgo-bolt-build
	./target/x86_64-unknown-linux-gnu/release/pytorch-tests-bolt-instrumented

pgo-bolt: bolt-instrument
	cargo pgo bolt optimize --with-pgo
