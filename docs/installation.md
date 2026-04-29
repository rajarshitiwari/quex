# Installation

Quex is built with modern Python packaging in mind. We highly recommend using [uv](https://github.com/astral-sh/uv) for blazing-fast dependency management.

## Standard Installation
To install the core Quex library and its fast Numpy backend:
```bash
git clone https://github.com/rajarshitiwari/quex.git
cd quex
uv pip install quex
```

## Developer & Documentation Installation

If you are contributing to Quex or building these docs locally:

```bash
git clone https://github.com/rajarshitiwari/quex.git
cd quex
uv sync --all-groups
```
