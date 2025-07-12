# 🤖 VIBE - Autonomous Coding Agent

<div align="center">

![VIBE Logo](https://img.shields.io/badge/VIBE-Autonomous%20Coding%20Agent-blue?style=for-the-badge)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)](https://python.org)
[![Claude](https://img.shields.io/badge/Claude-3.5%20Sonnet-orange?style=flat-square)](https://claude.ai)
[![Gemini](https://img.shields.io/badge/Gemini-2.0%20Flash-green?style=flat-square)](https://ai.google.dev)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

**Transform your project documentation into complete, production-ready software with AI**

[🚀 Quick Start](#-quick-start) • [📖 Documentation](#-documentation) • [🛠️ Features](#-features) • [🎯 Examples](#-examples)

</div>

---

## 📋 **Overview**

VIBE is an advanced autonomous coding agent that transforms project documentation into complete, production-ready software. Using a sophisticated **Master Claude + Code Claude** architecture with comprehensive defense systems, VIBE manages complex projects with intelligent supervision, context management, and automated quality assurance.

### 🌟 **Key Highlights**

- 🧠 **Intelligent Dual-AI Architecture** - Master Claude (Gemini) supervises Code Claude (Claude CLI) for optimal results
- 📚 **Multi-format Document Processing** - PDF, DOCX, Markdown, images, and UI mockups
- ⚡ **Context-Aware Execution** - 5 advanced compression strategies handle large-scale projects
- 🔄 **Dependency-Based Orchestration** - Smart task scheduling with parallel execution
- 🛡️ **Comprehensive Defense Systems** - Process management, deadlock detection, circuit breakers
- 💾 **Recovery & Persistence** - Checkpoint-based recovery with session continuity
- 📊 **Real-time Monitoring** - Live dashboard with execution insights and health monitoring

---

## 🚀 **Quick Start**

### Prerequisites

#### System Requirements
- **Python**: 3.9+ (tested with 3.12.3)
- **Operating System**: Linux, macOS, or Windows with WSL2
- **Memory**: 4GB+ RAM recommended
- **Storage**: 2GB+ free space

#### Required Tools
- **Claude CLI**: Must be installed and configured
- **Git**: For version control
- **Node.js/npm**: If using npx claude option

#### API Keys Required
- **Gemini API Key**: From [Google AI Studio](https://makersuite.google.com/)
- **Anthropic API Key**: For Claude access

### Installation

```bash
# Clone the repository
git clone https://github.com/Vibers-ai/auto-vibe.git
cd auto-vibe

# Create and activate virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Claude CLI Setup

```bash
# Check if Claude CLI is available
claude --version

# Or if using npx:
npx claude --version

# Configure Claude CLI if needed
claude configure
```

### Setup

```bash
# Initialize a new project using module syntax
python -m src.cli init my-awesome-project
cd my-awesome-project

# Configure API keys (create .env file)
# Copy from parent directory:
cp ../.env.example .env

# Edit .env with your API keys and configuration:
# GEMINI_API_KEY=your_gemini_key_here
# ANTHROPIC_API_KEY=your_claude_key_here
# 
# Optional configurations:
# PARALLEL_TASKS=4
# MAX_RETRIES=3
# TASK_TIMEOUT=600
# LOG_LEVEL=INFO
```

### Generate Your First Project

```bash
# Add your project documentation to docs/ folder
# (PDFs, Word docs, images, markdown files)

# Generate complete software project
python -m src.cli generate

# Monitor progress (optional)
python -m src.cli monitor --mode web --port 8080

# Check status
python -m src.cli status

# Validate tasks (if needed)
python -m src.cli validate tasks.json
```

---

## 🏗️ **Architecture**

### Master Claude + Code Claude System

```mermaid
graph TD
    A[📄 Project Docs] --> B[🔍 Document Ingestion]
    B --> C[🧠 Master Planner - Gemini]
    C --> D[📋 Task Generation]
    D --> E[👑 Master Claude Supervisor]
    E --> F[⚡ Code Claude CLI Executor]
    F --> G[📊 Context Manager]
    G --> E
    E --> H[✅ Quality Assurance]
    H --> I[🚀 Generated Project]
```

### Core Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Master Claude** | Gemini 2.0 Flash | Project supervision, context management, architectural decisions |
| **Code Claude** | Claude 3.5 Sonnet CLI | Code generation, implementation, testing |
| **Context Manager** | Custom Algorithm | Intelligent compression with 5 strategies (128K token management) |
| **Task Orchestrator** | DAG-based Engine | Dependency resolution, parallel execution |
| **Defense System** | Multi-layer Protection | Process management, deadlock detection, circuit breakers |
| **Recovery System** | SQLite + Checkpoints | Session persistence, failure recovery |
| **Monitor** | Rich + Web Dashboard | Real-time progress tracking and health monitoring |

---

## 🛠️ **Features**

### 📝 **Document Processing**
- **Multi-format Support**: PDF, DOCX, Markdown, Plain Text
- **Visual Analysis**: UI mockups and diagrams via Gemini Vision
- **Smart Synthesis**: Automatic ProjectBrief.md generation
- **Context Extraction**: Requirements, specifications, and constraints

### 🎯 **Intelligent Execution**
- **Architectural Planning**: Gemini-powered system design
- **Task Dependency Management**: Automated DAG-based scheduling
- **Parallel Processing**: Independent task execution
- **Context Compression**: 5 advanced strategies for large projects
  - `SUMMARIZE` - Smart completion summaries
  - `HIERARCHICAL` - Importance-based prioritization  
  - `SLIDING_WINDOW` - Recent context focus
  - `SEMANTIC_FILTERING` - Relevance-based filtering
  - `HYBRID` - Combined strategy optimization

### 🔧 **Code Generation**
- **Full-Stack Projects**: Frontend + Backend + Database
- **Framework Agnostic**: React, Vue, Node.js, Python, etc.
- **Best Practices**: Automated testing, linting, documentation
- **Quality Assurance**: Multi-layer validation and verification

### 🛡️ **Defense & Security Systems**
- **Process Management**: Central child process tracking and cleanup
- **Deadlock Detection**: Multi-type deadlock prevention and recovery
- **Circuit Breakers**: API failure protection with exponential backoff
- **File Operation Guards**: Concurrent file modification prevention
- **Health Monitoring**: Real-time system health checks and auto-recovery

### 🎛️ **Management & Monitoring**
- **Session Persistence**: Resume interrupted executions
- **Recovery System**: Automatic checkpoint-based recovery
- **Real-time Dashboard**: Web-based progress monitoring with health metrics
- **CLI Tools**: Comprehensive command-line interface

---

## 🎯 **Examples**

### Example 1: E-commerce Platform

```bash
# Create project from business requirements document
python -m src.cli init ecommerce-platform
cd ecommerce-platform

# Add your docs/
# - business_requirements.pdf
# - ui_mockups.png
# - api_specifications.md

python -m src.cli generate
# ✅ Generates: React frontend, Node.js backend, PostgreSQL database, tests
```

### Example 2: Mobile App Backend

```bash
# Generate API backend from mobile app specs
python -m src.cli init mobile-api
cd mobile-api

# Add docs/mobile_app_spec.docx with API requirements
python -m src.cli generate --output-path ./backend

# ✅ Generates: FastAPI backend, authentication, database models, documentation
```

---

## 📖 **Documentation**

### CLI Commands

```bash
# Project Management
python -m src.cli init <project-name>              # Initialize new project
python -m src.cli generate                         # Generate project from docs
python -m src.cli validate tasks.json              # Validate task configuration

# Context Management  
python -m src.cli context stats --project my-app   # Show context statistics
python -m src.cli context compress --strategy hybrid # Force context compression
python -m src.cli context export --file knowledge.json # Export project knowledge

# Monitoring & Status
python -m src.cli monitor --mode web --port 8080   # Start web dashboard
python -m src.cli monitor --mode terminal          # Start terminal dashboard  
python -m src.cli status --session session_id      # Check execution status

# Defense System Management
python -m src.cli defense status                   # Check defense system status
python -m src.cli defense diagnostics              # Run comprehensive diagnostics
python -m src.cli defense reset                    # Emergency system reset

# Sample and Demo
python -m src.cli sample --output sample-tasks.json # Generate sample tasks
```

### Configuration

Create a `.env` file with your API keys:

```env
# Required API Keys
GEMINI_API_KEY=your_gemini_api_key_here
ANTHROPIC_API_KEY=your_claude_api_key_here

# Optional Model Configuration
GEMINI_MODEL=gemini-2.0-flash-exp
CLAUDE_MODEL=claude-3-5-sonnet-20241022

# Execution Configuration
MAX_RETRIES=3
TASK_TIMEOUT=600
PARALLEL_TASKS=4
LOG_LEVEL=INFO

# Rate Limiting
CLAUDE_REQUESTS_PER_MINUTE=50
GEMINI_REQUESTS_PER_MINUTE=60

# Claude CLI Configuration  
CLAUDE_CLI_SKIP_PERMISSIONS=true
CLAUDE_CLI_USE_NPX=false

# Defense System Configuration
ENABLE_PROCESS_MANAGER=true
ENABLE_DEADLOCK_DETECTION=true
ENABLE_CIRCUIT_BREAKERS=true
ENABLE_HEALTH_MONITORING=true

# Monitoring
ENABLE_MONITORING=true
WEB_DASHBOARD_PORT=8080
```

---

## 📁 **Project Structure**

```
auto-vibe/
├── src/
│   ├── shared/                    # Core shared modules
│   │   ├── agents/               # AI agent implementations
│   │   │   ├── context_manager.py        # Context compression & management
│   │   │   ├── document_ingestion.py     # Multi-format document processing
│   │   │   └── master_planner.py         # Gemini-based architecture planning
│   │   ├── core/                 # Core engine components
│   │   │   ├── schema.py                 # Task schema & validation
│   │   │   ├── state_manager.py          # Execution state management
│   │   │   ├── parallel_executor.py      # Parallel task execution
│   │   │   ├── consistency_manager.py    # Code consistency validation
│   │   │   ├── defense_system.py         # Integrated defense system
│   │   │   ├── process_manager.py        # Central process management
│   │   │   ├── deadlock_detector.py      # Deadlock detection & recovery
│   │   │   ├── circuit_breaker.py        # API circuit breaker pattern
│   │   │   ├── file_operation_guard.py   # File concurrency control
│   │   │   ├── token_overflow_guard.py   # Token overflow prevention
│   │   │   └── system_health_monitor.py  # Health monitoring & recovery
│   │   ├── monitoring/           # Real-time monitoring
│   │   │   ├── master_claude_monitor.py  # Rich-based terminal dashboard
│   │   │   └── web_dashboard.py          # Web-based monitoring
│   │   ├── tools/                # Agent-Computer Interface
│   │   │   └── aci_interface.py          # Tool abstraction layer
│   │   └── utils/                # Utility functions
│   │       ├── api_manager.py            # Rate limiting & retry logic
│   │       ├── config.py                 # Configuration management
│   │       ├── file_utils.py             # File system operations
│   │       └── recovery_manager.py       # Checkpoint & recovery system
│   └── cli/                      # CLI-specific implementation
│       ├── agents/               # CLI agent implementations
│       │   ├── claude_cli_executor.py    # Claude CLI subprocess management
│       │   └── master_claude_cli_supervisor.py # Master supervision logic
│       └── core/                 # CLI execution engine
│           └── executor_cli.py           # Main CLI task executor
├── tests/                        # Comprehensive test suite
├── docs/                         # Documentation (auto-generated)
└── requirements.txt              # Python dependencies
```

---

## 🚨 **Troubleshooting**

### Common Issues and Solutions

#### 1. Installation Issues

```bash
# ModuleNotFoundError: Ensure virtual environment is activated
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install -e .

# Permission errors
chmod +x scripts/*
sudo chown -R $USER:$USER .
```

#### 2. API Configuration

```bash
# Invalid or missing API keys
cat .env  # Check if API keys are properly set

# Test API connectivity
python -c "import google.generativeai as genai; genai.configure(api_key='your_key'); print('Gemini OK')"
```

#### 3. Claude CLI Issues

```bash
# Claude CLI not found
claude --version

# If not installed, visit: https://claude.ai/docs/cli-installation
# Or use npx version:
export CLAUDE_CLI_USE_NPX=true
```

#### 4. Memory and Performance

```bash
# Reduce memory usage for large projects
export PARALLEL_TASKS=2
export CONTEXT_COMPRESSION_STRATEGY=hybrid

# Enable debug mode
export LOG_LEVEL=DEBUG
python -m src.cli generate --log-level DEBUG
```

#### 5. Process Management Issues

```bash
# Check process status
python -m src.cli defense status

# Emergency cleanup
python -m src.cli defense reset

# Check system health
python -m src.cli defense diagnostics
```

### Recovery from Failed Executions

```bash
# Check session status
python -m src.cli status --session session_id

# Resume from checkpoint (automatic)
python -m src.cli generate --skip-planning

# Manual recovery with context export/import
python -m src.cli context export --file backup.json
python -m src.cli context import --file backup.json
```

### Debug Mode

```bash
# Enable comprehensive debugging
export LOG_LEVEL=DEBUG
export ENABLE_MONITORING=true
python -m src.cli generate --debug

# Check logs
tail -f output/logs/session_*.md
```

---

## 🤝 **Contributing**

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/Vibers-ai/auto-vibe.git
cd auto-vibe
pip install -r requirements.txt
pip install -e .

# Run tests
pytest tests/

# Run linting
flake8 src/
black src/
```

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 **Acknowledgments**

- **Anthropic** for Claude 3.5 Sonnet and Claude CLI
- **Google** for Gemini 2.0 Flash and AI capabilities
- **Open Source Community** for NetworkX, Rich, FastAPI, and other excellent tools
- **Security Research Community** for defense system design patterns

---

<div align="center">

**Built with ❤️ by the VIBE Team**

[🌟 Star us on GitHub](https://github.com/Vibers-ai/auto-vibe) • [🐛 Report Issues](https://github.com/Vibers-ai/auto-vibe/issues) • [💬 Discussions](https://github.com/Vibers-ai/auto-vibe/discussions)

</div>