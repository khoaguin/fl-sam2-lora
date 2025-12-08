set dotenv-load := true

# ---------------------------------------------------------------------------------------------------------------------
# Private vars

[private]
_red := '\033[1;31m'
[private]
_cyan := '\033[1;36m'
[private]
_green := '\033[1;32m'
[private]
_yellow := '\033[1;33m'
[private]
_nc := '\033[0m'


# ---------------------------------------------------------------------------------------------------------------------
@default:
    just --list

# ---------------------------------------------------------------------------------------------------------------------
[group('utils')]
setup:
    #!/bin/sh
    echo "{{ _cyan }}Setting up environment...{{ _nc }}"
    uv venv
    source .venv/bin/activate
    echo "{{ _cyan }}Installing dependencies from pyproject.toml...{{ _nc }}"
    uv sync
    echo "{{ _cyan }}Installing sam2 from git...{{ _nc }}"
    mkdir -p libs
    cd libs && git clone https://github.com/facebookresearch/sam2.git && cd sam2 && uv pip install -e . && cd ..
    # rm -rf sam2
    echo "{{ _green }}✓ Setup complete!{{ _nc }}"

clean:
    #!/bin/sh
    echo "{{ _cyan }}Cleaning up{{ _nc }}"

    # Remove __pycache__ directories
    pycache_count=$(find . -type d -name "__pycache__" 2>/dev/null | wc -l)
    if [ "$pycache_count" -gt 0 ]; then
        echo "  {{ _red }}✗{{ _nc }} Removing $pycache_count __pycache__ directories"
        find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    fi

    echo "{{ _green }}✓ Clean complete!{{ _nc }}"
