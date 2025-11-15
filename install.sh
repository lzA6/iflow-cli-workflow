#!/bin/bash

# 🌟 iFlow CLI 一键安装脚本
# 🎯 适用于 Linux/macOS 系统
# 📝 作者: AI架构师团队
# 📅 版本: 11.0.0

set -e  # 遇到错误立即退出

# 🌈 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# 🎯 打印带颜色的消息
print_message() {
    echo -e "${2}${1}${NC}"
}

# 🌟 打印标题
print_title() {
    echo -e "\n${PURPLE}================================${NC}"
    echo -e "${CYAN}🌟 iFlow CLI 工作流系统${NC}"
    echo -e "${CYAN}🧠 AGI级别的智能助手${NC}"
    echo -e "${PURPLE}================================${NC}\n"
}

# ✅ 成功消息
print_success() {
    print_message "✅ $1" "$GREEN"
}

# ⚠️ 警告消息
print_warning() {
    print_message "⚠️  $1" "$YELLOW"
}

# ❌ 错误消息
print_error() {
    print_message "❌ $1" "$RED"
}

# ℹ️ 信息消息
print_info() {
    print_message "ℹ️  $1" "$BLUE"
}

# 🚀 主函数
main() {
    print_title
    
    # 🎋 检查操作系统
    print_info "检查系统环境..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="Linux"
        print_success "检测到 Linux 系统"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macOS"
        print_success "检测到 macOS 系统"
    else
        print_error "不支持的操作系统: $OSTYPE"
        exit 1
    fi
    
    # 🐍 检查Python版本
    print_info "检查 Python 版本..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
        
        if [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -ge 8 ]]; then
            print_success "Python 版本检查通过: $PYTHON_VERSION"
            PYTHON_CMD="python3"
        else
            print_error "Python 版本过低: $PYTHON_VERSION (需要 3.8+)"
            print_info "请升级 Python 版本后重试"
            exit 1
        fi
    else
        print_error "未找到 Python3，请先安装 Python 3.8+"
        exit 1
    fi
    
    # 📦 检查pip
    print_info "检查 pip..."
    
    if command -v pip3 &> /dev/null; then
        print_success "pip3 检查通过"
        PIP_CMD="pip3"
    elif command -v pip &> /dev/null; then
        print_success "pip 检查通过"
        PIP_CMD="pip"
    else
        print_error "未找到 pip，请先安装 pip"
        exit 1
    fi
    
    # 🌐 检查网络连接
    print_info "检查网络连接..."
    
    if ping -c 1 google.com &> /dev/null; then
        print_success "网络连接正常"
    else
        print_warning "网络连接可能有问题，但继续安装..."
    fi
    
    # 📁 检查Git
    print_info "检查 Git..."
    
    if command -v git &> /dev/null; then
        print_success "Git 检查通过"
    else
        print_error "未找到 Git，请先安装 Git"
        if [[ "$OS" == "Linux" ]]; then
            print_info "Ubuntu/Debian: sudo apt install git"
            print_info "CentOS/RHEL: sudo yum install git"
        elif [[ "$OS" == "macOS" ]]; then
            print_info "macOS: brew install git"
        fi
        exit 1
    fi
    
    # 🎯 询问安装目录
    echo
    print_info "请选择安装目录:"
    print_info "1) 当前目录 (./iflow-cli-workflow)"
    print_info "2) 用户主目录 (~/iflow-cli-workflow)"
    print_info "3) 自定义目录"
    
    read -p "请输入选择 (1-3): " -n 1 -r
    echo
    
    case $REPLY in
        1)
            INSTALL_DIR="$(pwd)/iflow-cli-workflow"
            ;;
        2)
            INSTALL_DIR="$HOME/iflow-cli-workflow"
            ;;
        3)
            read -p "请输入自定义目录路径: " INSTALL_DIR
            ;;
        *)
            print_error "无效选择，使用默认目录"
            INSTALL_DIR="$(pwd)/iflow-cli-workflow"
            ;;
    esac
    
    # 📁 创建安装目录
    print_info "创建安装目录: $INSTALL_DIR"
    
    if [[ -d "$INSTALL_DIR" ]]; then
        print_warning "目录已存在，将进行更新..."
        cd "$INSTALL_DIR"
        git pull origin main
    else
        mkdir -p "$INSTALL_DIR"
        cd "$INSTALL_DIR"
    fi
    
    # 🌟 克隆项目
    if [[ ! -d ".git" ]]; then
        print_info "克隆项目仓库..."
        git clone https://github.com/lzA6/iflow-cli-workflow.git .
    else
        print_info "更新项目仓库..."
        git pull origin main
    fi
    
    print_success "项目下载完成"
    
    # 📦 安装Python依赖
    print_info "安装 Python 依赖包..."
    
    if [[ -f "requirements.txt" ]]; then
        $PIP_CMD install -r requirements.txt
        print_success "依赖包安装完成"
    else
        print_warning "未找到 requirements.txt，安装核心依赖..."
        $PIP_CMD install asyncio numpy psutil pathlib
        print_success "核心依赖安装完成"
    fi
    
    # 🔧 创建配置文件
    print_info "创建配置文件..."
    
    CONFIG_FILE=".iflow/settings.local.json"
    CONFIG_EXAMPLE=".iflow/settings.local.json.example"
    
    if [[ -f "$CONFIG_EXAMPLE" ]]; then
        if [[ ! -f "$CONFIG_FILE" ]]; then
            cp "$CONFIG_EXAMPLE" "$CONFIG_FILE"
            print_success "配置文件创建完成"
        else
            print_warning "配置文件已存在，跳过创建"
        fi
    else
        # 创建基础配置文件
        cat > "$CONFIG_FILE" << EOF
{
  "model_config": {
    "providers": {
      "openai": {
        "enabled": true,
        "models": ["gpt-4-turbo", "gpt-3.5-turbo"],
        "api_key_env": "OPENAI_API_KEY"
      },
      "anthropic": {
        "enabled": true,
        "models": ["claude-3-opus", "claude-3-sonnet"],
        "api_key_env": "ANTHROPIC_API_KEY"
      }
    }
  },
  "security_config": {
    "zero_trust_enabled": true,
    "sandbox_level": "strict"
  }
}
EOF
        print_success "基础配置文件创建完成"
    fi
    
    # 🧪 运行测试
    print_info "运行基础测试..."
    
    if $PYTHON_CMD -c "import asyncio, numpy, psutil; print('✅ 依赖测试通过')"; then
        print_success "基础测试通过"
    else
        print_error "基础测试失败，请检查依赖安装"
        exit 1
    fi
    
    # 🎯 创建启动脚本
    print_info "创建启动脚本..."
    
    cat > "iflow-cli" << EOF
#!/bin/bash
# 🌟 iFlow CLI 启动脚本

SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
cd "\$SCRIPT_DIR"

# 🧠 启动AGI核心
echo "🚀 启动 iFlow CLI..."
python3 .iflow/core/agi_core_v11.py "\$@"
EOF
    
    chmod +x "iflow-cli"
    print_success "启动脚本创建完成"
    
    # 📋 创建桌面快捷方式 (Linux)
    if [[ "$OS" == "Linux" ]] && [[ -d "$HOME/Desktop" ]]; then
        print_info "创建桌面快捷方式..."
        
        cat > "$HOME/Desktop/iFlow CLI.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=iFlow CLI
Comment=AGI级别的智能工作流系统
Exec=$INSTALL_DIR/iflow-cli
Icon=$INSTALL_DIR/.iflow/icons/iflow-icon.png
Terminal=true
Categories=Development;
EOF
        
        chmod +x "$HOME/Desktop/iFlow CLI.desktop"
        print_success "桌面快捷方式创建完成"
    fi
    
    # 🎉 安装完成
    print_title
    print_success "🎉 iFlow CLI 安装完成！"
    echo
    print_info "📁 安装目录: $INSTALL_DIR"
    print_info "🚀 启动方式:"
    echo "   方式1: cd $INSTALL_DIR && ./iflow-cli"
    echo "   方式2: cd $INSTALL_DIR && python3 .iflow/core/agi_core_v11.py"
    echo
    print_info "📚 更多信息:"
    echo "   📖 文档: https://github.com/lzA6/iflow-cli-workflow"
    echo "   💬 社区: https://discord.gg/iflow"
    echo "   🐛 问题: https://github.com/lzA6/iflow-cli-workflow/issues"
    echo
    print_warning "⚠️  下一步:"
    echo "   1. 配置API密钥: 编辑 $CONFIG_FILE"
    echo "   2. 设置环境变量: export OPENAI_API_KEY='your-key'"
    echo "   3. 运行测试: python3 .iflow/tests/comprehensive_test_framework_v11.py"
    echo
    
    # 🎯 询问是否立即运行
    read -p "🚀 是否立即运行 iFlow CLI? (y/n): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "启动 iFlow CLI..."
        cd "$INSTALL_DIR"
        $PYTHON_CMD .iflow/core/agi_core_v11.py --help
    fi
    
    print_success "安装脚本执行完成！"
}

# 🚀 运行主函数
main "$@"
