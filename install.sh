#!/bin/bash
#
# Orion Data Analysis Agent - One-line installer
# Usage: curl -sSL https://raw.githubusercontent.com/gavrielhan/orion-data-analyst/main/install.sh | bash
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║                                                                   ║"
echo "║      ██████╗ ██████╗ ██╗ ██████╗ ███╗   ██╗                       ║"
echo "║     ██╔═══██╗██╔══██╗██║██╔═══██╗████╗  ██║                       ║"
echo "║     ██║   ██║██████╔╝██║██║   ██║██╔██╗ ██║                       ║"
echo "║     ██║   ██║██╔══██╗██║██║   ██║██║╚██╗██║                       ║"
echo "║     ╚██████╔╝██║  ██║██║╚██████╔╝██║ ╚████║                       ║"
echo "║      ╚═════╝ ╚═╝  ╚═╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝                       ║"
echo "║                                                                   ║"
echo "║                 Data Analysis Agent Installer                     ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check Python version
echo -e "${YELLOW}Checking Python installation...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed. Please install Python 3.8+ first.${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | grep -oE '[0-9]+\.[0-9]+' | head -1)
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo -e "${RED}❌ Python 3.8+ required. Found: $PYTHON_VERSION${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Python $PYTHON_VERSION found${NC}"

# Create installation directory
INSTALL_DIR="$HOME/.orion"
echo -e "${YELLOW}Creating installation directory: $INSTALL_DIR${NC}"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Download or clone repository
echo -e "${YELLOW}Downloading Orion...${NC}"
if [ -d "orion-data-analyst" ]; then
    echo -e "${YELLOW}Updating existing installation...${NC}"
    cd orion-data-analyst
    git pull
else
    git clone https://github.com/gavrielhan/orion-data-analyst.git
    cd orion-data-analyst
fi

# Create virtual environment
echo -e "${YELLOW}Creating virtual environment...${NC}"
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install --upgrade pip -q
pip install -e . -q

# Create wrapper script
echo -e "${YELLOW}Creating command-line wrapper...${NC}"
BIN_DIR="$HOME/.local/bin"
mkdir -p "$BIN_DIR"

cat > "$BIN_DIR/orion" << 'EOF'
#!/bin/bash
source "$HOME/.orion/orion-data-analyst/venv/bin/activate"
python -m src.cli "$@"
EOF

chmod +x "$BIN_DIR/orion"

# Check if .local/bin is in PATH
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    echo -e "${YELLOW}Adding ~/.local/bin to PATH...${NC}"
    
    # Detect shell and add to appropriate rc file
    if [ -n "$ZSH_VERSION" ]; then
        RC_FILE="$HOME/.zshrc"
    elif [ -n "$BASH_VERSION" ]; then
        RC_FILE="$HOME/.bashrc"
    else
        RC_FILE="$HOME/.profile"
    fi
    
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$RC_FILE"
    echo -e "${GREEN}✅ Added to $RC_FILE${NC}"
    echo -e "${YELLOW}Please run: source $RC_FILE${NC}"
fi

# Create .env template if it doesn't exist
if [ ! -f "$INSTALL_DIR/orion-data-analyst/.env" ]; then
    echo -e "${YELLOW}Creating .env template...${NC}"
    cat > "$INSTALL_DIR/orion-data-analyst/.env" << 'ENVEOF'
# Google Cloud Configuration
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json

# Gemini AI API Key
GEMINI_API_KEY=your-gemini-api-key

# Optional: BigQuery Settings
BIGQUERY_DATASET=bigquery-public-data.thelook_ecommerce
MAX_QUERY_ROWS=10000
QUERY_TIMEOUT=300
ENVEOF
    
    echo -e "${GREEN}✅ Created .env template${NC}"
    echo -e "${YELLOW}⚠️  Please edit ~/.orion/orion-data-analyst/.env with your API keys${NC}"
fi

# Success message
echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                                                                   ║${NC}"
echo -e "${GREEN}║                  ✅ Installation Complete!                         ║${NC}"
echo -e "${GREEN}║                                                                   ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo -e "  1. Configure your API keys:"
echo -e "     ${YELLOW}nano ~/.orion/orion-data-analyst/.env${NC}"
echo ""
echo -e "  2. Run Orion:"
echo -e "     ${GREEN}orion${NC}"
echo ""
echo -e "${BLUE}Documentation:${NC} https://github.com/gavrielhan/orion-data-analyst"
echo ""

