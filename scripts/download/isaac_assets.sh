#!/bin/bash
# Download Isaac Sim 5.1.0 Asset Packs

set -e

ASSETS_DIR="${ISAAC_ASSETS_DIR:-$HOME/.local/share/ov/pkg/isaac-sim-5.1.0/isaac-sim-assets}"
TEMP_DIR="/tmp/isaac-sim-assets"

echo "Isaac Sim Asset Downloader"
echo "=========================="
echo "Assets will be installed to: $ASSETS_DIR"
echo ""

mkdir -p "$TEMP_DIR"
mkdir -p "$ASSETS_DIR"

download_and_extract() {
    local name=$1
    local url=$2
    local filename=$(basename "$url")
    
    echo "[1/2] Downloading $name..."
    wget -q --show-progress -O "$TEMP_DIR/$filename" "$url"
    
    echo "[2/2] Extracting $name..."
    unzip -q -o "$TEMP_DIR/$filename" -d "$ASSETS_DIR"
    rm "$TEMP_DIR/$filename"
    
    echo "$name installed"
    echo ""
}

# Parse arguments
PACK=${1:-minimal}

case $PACK in
    minimal)
        echo "Installing MINIMAL pack (Robots + Sensors)"
        download_and_extract "Robots & Sensors" \
            "https://download.isaacsim.omniverse.nvidia.com/isaac-sim-assets-robots_and_sensors-5.1.0.zip"
        ;;
    
    standard)
        echo "Installing STANDARD pack (Robots + Materials + Environments)"
        download_and_extract "Robots & Sensors" \
            "https://download.isaacsim.omniverse.nvidia.com/isaac-sim-assets-robots_and_sensors-5.1.0.zip"
        download_and_extract "Materials & Props" \
            "https://download.isaacsim.omniverse.nvidia.com/isaac-sim-assets-materials_and_props-5.1.0.zip"
        download_and_extract "Environments" \
            "https://download.isaacsim.omniverse.nvidia.com/isaac-sim-assets-environments-5.1.0.zip"
        ;;
    
    complete)
        echo "Installing COMPLETE pack (All assets ~50GB+)"
        echo "[1/3] Downloading Complete Part 1..."
        wget -q --show-progress -O "$TEMP_DIR/isaac-sim-assets-complete-5.1.0.zip.001" \
            "https://download.isaacsim.omniverse.nvidia.com/isaac-sim-assets-complete-5.1.0.zip.001"
        echo "[2/3] Downloading Complete Part 2..."
        wget -q --show-progress -O "$TEMP_DIR/isaac-sim-assets-complete-5.1.0.zip.002" \
            "https://download.isaacsim.omniverse.nvidia.com/isaac-sim-assets-complete-5.1.0.zip.002"
        echo "[3/3] Downloading Complete Part 3..."
        wget -q --show-progress -O "$TEMP_DIR/isaac-sim-assets-complete-5.1.0.zip.003" \
            "https://download.isaacsim.omniverse.nvidia.com/isaac-sim-assets-complete-5.1.0.zip.003"

        echo "Merging split archives..."
        cat "$TEMP_DIR/isaac-sim-assets-complete-5.1.0.zip.001" \
            "$TEMP_DIR/isaac-sim-assets-complete-5.1.0.zip.002" \
            "$TEMP_DIR/isaac-sim-assets-complete-5.1.0.zip.003" \
            > "$TEMP_DIR/isaac-sim-assets-complete-5.1.0.zip"

        echo "Extracting complete pack..."
        unzip -q -o "$TEMP_DIR/isaac-sim-assets-complete-5.1.0.zip" -d "$ASSETS_DIR"

        rm -f "$TEMP_DIR/isaac-sim-assets-complete-5.1.0.zip."* \
              "$TEMP_DIR/isaac-sim-assets-complete-5.1.0.zip"
        ;;
    
    *)
        echo "Usage: $0 [minimal|standard|complete]"
        echo ""
        echo "Packs:"
        echo "  minimal   - Robots & Sensors only (~5GB)"
        echo "  standard  - Robots + Materials + Environments (~15GB)"
        echo "  complete  - Everything (~50GB+)"
        exit 1
        ;;
esac

echo "================================"
echo "Asset installation complete."
echo ""
echo "Assets location: $ASSETS_DIR"
echo ""
echo "To use in Python:"
echo "  export ISAAC_ASSETS_PATH=$ASSETS_DIR"
