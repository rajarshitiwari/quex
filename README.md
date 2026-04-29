# Quex

Initially, these are just a lot of big words!

A high-performance, backend-agnostic dispatcher for quantum circuit execution.

Quex aims to be the "JAX for Quantum Computing." Currently, many quantum simulators are tightly coupled to their frontend SDKs (e.g., Qiskit, Cirq). Quex completely divorces circuit construction from circuit execution.

By ingesting standard intermediate representations (like OpenQASM 3.0), Quex acts as an intelligent dispatcher, routing the heavy linear algebra required for quantum simulation to the most optimal, hardware-specific classical solver available on your machine—whether that is a multi-core CPU, an NVIDIA GPU (via cuQuantum), or an Apple Silicon unified memory architecture.


## Core Architecture

1. **The Ingestion Layer:** Parses hardware-agnostic formats (OpenQASM 3.0, and eventually QIR) into a lightweight internal Abstract Syntax Tree (AST).

2. **The Dispatcher (WIP):** Analyzes circuit depth, qubit count, and host hardware to intelligently route execution.

3. **The Execution Engines (WIP):** Wrappers around optimized, low-level linear algebra libraries (NumPy, SciPy Sparse, cuQuantum, etc.) to crunch the matrices efficiently.


## Installation and Setup

Quex is managed using [uv](https://docs.astral.sh/uv/), the lightning-fast Python package manager written in Rust.


## Prerequisites

If you don't have uv installed, we recommend the standalone installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Local Development Setup

To clone the repository and set up the development environment instantly:

```bash
# 1. Clone the repo
git clone https://github.com/rajarshitiwari/quex.git
cd quex

# 2. Sync dependencies (uv will automatically create a .venv and install the lockfile)
uv sync

# 3. (Optional) Run tests to ensure everything is working
uv run pytest
```

### 💻 Quick Start

Currently, Quex features a highly efficient (hopefully) parsing layer for OpenQASM 3.0 strings. Here is how to ingest a standard quantum circuit:

```python
from quex.parser import parse_qasm_string

# A standard OpenQASM 3.0 string (e.g., a Bell State preparation)
bell_state_circuit = """
OPENQASM 3.0;
qubit[2] q;
h q[0];
cx q[0], q[1];
"""

# Ingest the circuit into Quex's internal representation
operations = parse_qasm_string(bell_state_circuit)

for op in operations:
    print(op)

# Output:
# {'gate': 'h', 'targets': [('q', 0)]}
# {'gate': 'cx', 'targets': [('q', 0), ('q', 1)]}
```

### Running Scripts Locally

Because uv handles virtual environments automatically, you do not need to manually activate your .venv. Simply prepend uv run to execute your scripts with all Quex dependencies loaded:

```bash
uv run your_script.py
```

## 🗺️ Roadmap

- [x] OpenQASM 3.0 Ingestion Layer
- [x] Internal Circuit Representation (DAG structure)
- [x] Baseline Classical Execution Engine (NumPy state-vector simulation)
- [ ] Hardware Dispatcher Logic
- [ ] Advanced Execution Engines (SciPy Sparse, cuQuantum integration)
- [ ] LLVM-based QIR Ingestion

