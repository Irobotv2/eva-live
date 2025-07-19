#!/bin/bash

# Eva Live Development Environment Setup Script
# This script sets up the complete development environment

set -e  # Exit on any error

echo "🚀 Setting up Eva Live development environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on supported OS
check_os() {
    print_status "Checking operating system..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        print_success "Linux detected"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        print_success "macOS detected"
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
        OS="windows"
        print_success "Windows detected"
    else
        print_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Python
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        print_success "Python $PYTHON_VERSION found"
    else
        print_error "Python 3.9+ is required but not found"
        exit 1
    fi
    
    # Check Node.js
    if command -v node &> /dev/null; then
        NODE_VERSION=$(node --version)
        print_success "Node.js $NODE_VERSION found"
    else
        print_warning "Node.js not found. Installing..."
        install_nodejs
    fi
    
    # Check Git
    if command -v git &> /dev/null; then
        print_success "Git found"
    else
        print_error "Git is required but not found"
        exit 1
    fi
    
    # Check Docker (optional)
    if command -v docker &> /dev/null; then
        print_success "Docker found"
    else
        print_warning "Docker not found. Some features may not work."
    fi
}

# Install Node.js if not present
install_nodejs() {
    if [[ "$OS" == "linux" ]]; then
        curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
        sudo apt-get install -y nodejs
    elif [[ "$OS" == "macos" ]]; then
        if command -v brew &> /dev/null; then
            brew install node
        else
            print_error "Homebrew not found. Please install Node.js manually."
            exit 1
        fi
    else
        print_error "Please install Node.js manually for Windows"
        exit 1
    fi
}

# Create Python virtual environment
setup_python_env() {
    print_status "Setting up Python virtual environment..."
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Created virtual environment"
    else
        print_warning "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install Python dependencies
    print_status "Installing Python dependencies..."
    pip install -r requirements.txt
    print_success "Python dependencies installed"
}

# Setup Node.js environment
setup_nodejs_env() {
    print_status "Setting up Node.js environment..."
    
    if [ -f "package.json" ]; then
        npm install
        print_success "Node.js dependencies installed"
    else
        print_warning "No package.json found, skipping Node.js setup"
    fi
}

# Setup environment configuration
setup_environment() {
    print_status "Setting up environment configuration..."
    
    if [ ! -f ".env" ]; then
        cp .env.example .env
        print_success "Created .env file from template"
        print_warning "Please edit .env file with your actual configuration values"
    else
        print_warning ".env file already exists"
    fi
    
    # Create necessary directories
    mkdir -p logs
    mkdir -p data
    mkdir -p uploads
    mkdir -p cache
    
    print_success "Created necessary directories"
}

# Setup database (optional - requires Docker)
setup_database() {
    print_status "Setting up development database..."
    
    if command -v docker &> /dev/null && command -v docker-compose &> /dev/null; then
        if [ -f "docker/docker-compose.dev.yml" ]; then
            docker-compose -f docker/docker-compose.dev.yml up -d postgres redis
            print_success "Development databases started"
            
            # Wait for databases to be ready
            print_status "Waiting for databases to be ready..."
            sleep 10
            
            # Run database migrations (when implemented)
            # python -m alembic upgrade head
            
        else
            print_warning "Docker compose file not found, skipping database setup"
        fi
    else
        print_warning "Docker not available, skipping database setup"
    fi
}

# Setup AI services configuration
setup_ai_services() {
    print_status "Setting up AI services..."
    
    # Download required models
    print_status "Downloading required AI models..."
    
    # Create models directory
    mkdir -p models
    
    # Download sentence transformer model (this will be cached locally)
    python3 -c "
from sentence_transformers import SentenceTransformer
print('Downloading sentence transformer model...')
model = SentenceTransformer('all-MiniLM-L6-v2')
print('Model downloaded successfully')
" || print_warning "Failed to download sentence transformer model"
    
    print_success "AI services setup complete"
}

# Create development scripts
create_dev_scripts() {
    print_status "Creating development scripts..."
    
    # Start development server script
    cat > scripts/start-dev.sh << 'EOF'
#!/bin/bash
# Start Eva Live development server

# Activate virtual environment
source venv/bin/activate

# Set development environment
export ENVIRONMENT=development
export DEBUG=true

# Start the application
python -m src.main
EOF
    
    # Make script executable
    chmod +x scripts/start-dev.sh
    
    # Create test script
    cat > scripts/run-tests.sh << 'EOF'
#!/bin/bash
# Run Eva Live tests

# Activate virtual environment
source venv/bin/activate

# Run tests
python -m pytest tests/ -v --cov=src --cov-report=html
EOF
    
    chmod +x scripts/run-tests.sh
    
    # Create format script
    cat > scripts/format-code.sh << 'EOF'
#!/bin/bash
# Format Eva Live code

# Activate virtual environment
source venv/bin/activate

# Format with black
black src/ tests/

# Sort imports with isort
isort src/ tests/

# Lint with flake8
flake8 src/ tests/
EOF
    
    chmod +x scripts/format-code.sh
    
    print_success "Development scripts created"
}

# Setup IDE configuration
setup_ide_config() {
    print_status "Setting up IDE configuration..."
    
    # VS Code settings
    mkdir -p .vscode
    
    cat > .vscode/settings.json << 'EOF'
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.args": ["--profile", "black"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        ".pytest_cache": true,
        ".coverage": true,
        "htmlcov": true
    }
}
EOF
    
    # VS Code launch configuration
    cat > .vscode/launch.json << 'EOF'
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Eva Live API",
            "type": "python",
            "request": "launch",
            "module": "src.main",
            "console": "integratedTerminal",
            "envFile": "${workspaceFolder}/.env",
            "cwd": "${workspaceFolder}"
        },
        {
            "name": "Eva Live Tests",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": ["tests/", "-v"],
            "console": "integratedTerminal",
            "envFile": "${workspaceFolder}/.env",
            "cwd": "${workspaceFolder}"
        }
    ]
}
EOF
    
    print_success "IDE configuration created"
}

# Setup git hooks
setup_git_hooks() {
    print_status "Setting up git hooks..."
    
    # Pre-commit hook
    cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Eva Live pre-commit hook

# Activate virtual environment
source venv/bin/activate

# Format code
black src/ tests/
isort src/ tests/

# Run linting
flake8 src/ tests/

# Run tests
python -m pytest tests/ --tb=short

# Add formatted files
git add src/ tests/
EOF
    
    chmod +x .git/hooks/pre-commit
    
    print_success "Git hooks setup complete"
}

# Main setup function
main() {
    print_status "Starting Eva Live development environment setup"
    
    # Change to project directory
    cd "$(dirname "$0")/.."
    
    check_os
    check_prerequisites
    setup_python_env
    setup_nodejs_env
    setup_environment
    setup_database
    setup_ai_services
    create_dev_scripts
    setup_ide_config
    setup_git_hooks
    
    print_success "🎉 Eva Live development environment setup complete!"
    echo ""
    echo "Next steps:"
    echo "1. Edit .env file with your API keys and configuration"
    echo "2. Start the development server: ./scripts/start-dev.sh"
    echo "3. Open http://localhost:8000/docs to view the API documentation"
    echo ""
    echo "Useful commands:"
    echo "- Run tests: ./scripts/run-tests.sh"
    echo "- Format code: ./scripts/format-code.sh"
    echo "- View logs: tail -f logs/eva-live.log"
}

# Run main function
main "$@"
