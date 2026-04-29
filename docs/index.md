---
jupytext:
    formats: md:myst
    text_representation:
        extension: .md
        format_name: myst
kernelspec:
    display_name: Python 3
    language: python
    name: python3
mystnb:
    render_markdown_format: myst
---

# Welcom to Quex

Quex is a lean, high-throughput quantum execution framework inspired
by the Atomic Simulation Environment (ASE). It decouples quantum circuit
geometry from the simulation physics, allowing researchers to seamlessly
build templates, bind parameters dynamically, and dispatch massive computational batches.

## Key Features
* **ASE-Inspired Architecture:** Circuits and Simulators are distinct, attachable objects.
* **Native OpenQASM 3.0:** Read, auto-format, and strictly export standard-compliant quantum code.
* **Late-Binding Engine:** Designed specifically for high-speed Quantum Machine Learning loops.
* **Hardware-Efficient Ansätze:** Generate mathematically rigorous (Haar-uniform) random parameterized circuits instantly.

Check out the [Quickstart](tutorials/quickstart.ipynb) to get right into the code!

## Core Architecture

1. **The Ingestion Layer:** Parses hardware-agnostic formats (OpenQASM 3.0, and eventually QIR) into a lightweight internal Abstract Syntax Tree (AST).

2. **The Dispatcher (WIP):** Analyzes circuit depth, qubit count, and host hardware to intelligently route execution.

3. **The Execution Engines (WIP):** Wrappers around optimized, low-level linear algebra libraries (NumPy, SciPy Sparse, cuQuantum, etc.) to crunch the matrices efficiently.

```{mermaid}
mindmap
  root((quex))
    quex_core((Core))
      Circuit Class
      Gates Registry
      QASM Parser
      ASCII Visualizer
    quex_execution((Execution))
      Simulator Class (Interface)
      Numpy Simulator Class
      Supported Gates
    quex_generative((Generative))
      Random Ansatz Function
    quex_utils((Utils))
      __init__.py
      py.typed (type safety)
```


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

```{code-cell}
import quex as qx

# 1. Define the geometry (The "Atoms" equivalent)
qc = qx.Circuit(num_qubits=2)

# 2. Add standard operations
qc.add_operation('h', 0)
qc.add_operation('cx', [0, 1])

# 3. Quex features a built-in topological visualizer
print("Circuit Topology:")
print(qc)

```

More detailed, quickstart tutorial is available at [Quickstart](./tutorials/quickstart.ipynb).

### Running Scripts Locally

Because uv handles virtual environments automatically, you do not need to manually activate your .venv. Simply prepend uv run to execute your scripts with all Quex dependencies loaded:

```bash
uv run your_script.py
```

## 🗺️ Roadmap

- [x] OpenQASM 3.0 Ingestion Layer
- [x] Internal Circuit Representation
- [x] Baseline Classical Execution Engine (NumPy state-vector simulation)
- [ ] Hardware Dispatcher Logic
- [ ] Advanced Execution Engines (SciPy Sparse, cuQuantum integration)
- [ ] LLVM-based QIR Ingestion

