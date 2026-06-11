#!/bin/bash
# =============================================================================
# AI 能力评估系统 — 一键启动脚本
# 全程不需要 docker build，不需要 pip install，不需要下载任何东西
# =============================================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${BLUE}🔍 环境预检...${NC}"

# ---- 检查 Docker ----
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo -e "  ${RED}❌ docker: command not found${NC}"
        echo "     请先安装 Docker: sudo apt install docker.io -y"
        exit 1
    fi
    DOCKER_VERSION=$(docker --version 2>/dev/null | grep -oP '\d+\.\d+\.\d+' | head -1)
    if [ "$(printf '%s\n' "20.10" "$DOCKER_VERSION" | sort -V | head -1)" = "20.10" ]; then
        echo -e "  ${GREEN}✅${NC} Docker ${DOCKER_VERSION}"
    else
        echo -e "  ${YELLOW}⚠️${NC} Docker ${DOCKER_VERSION} (建议 ≥ 20.10)"
    fi
}

# ---- 检查存储驱动（必须是 overlay2）----
check_storage_driver() {
    DRIVER=$(docker info 2>/dev/null | grep "Storage Driver" | awk '{print $3}')
    case "$DRIVER" in
        overlay2)
            echo -e "  ${GREEN}✅${NC} 存储驱动: overlay2"
            ;;
        vfs)
            echo -e "  ${RED}❌${NC} 存储驱动为 vfs，移动硬盘上性能极差（I/O 慢 10-50 倍）"
            echo -e "     请执行以下命令切换到 overlay2："
            echo "       sudo systemctl stop docker"
            echo "       sudo mv /var/lib/docker /var/lib/docker.bak"
            echo "       sudo mkdir -p /etc/docker"
            echo '       echo '"'"'{"storage-driver":"overlay2"}'"'"' | sudo tee /etc/docker/daemon.json'
            echo "       sudo systemctl start docker"
            echo ""
            echo -e "     或使用 ${YELLOW}--force${NC} 参数强制启动（不推荐）"
            if [ "$1" != "--force" ]; then
                exit 1
            fi
            ;;
        *)
            echo -e "  ${YELLOW}⚠️${NC} 存储驱动: ${DRIVER}（建议迁移到 overlay2）"
            ;;
    esac
}

# ---- 检查磁盘空间 ----
check_disk() {
    AVAILABLE=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
    if [ "${AVAILABLE}" -ge 5 ]; then
        echo -e "  ${GREEN}✅${NC} 磁盘可用: ${AVAILABLE} GB"
    else
        echo -e "  ${RED}❌${NC} 磁盘可用: ${AVAILABLE} GB（需要 ≥ 5GB）"
        exit 1
    fi
}

# ---- 检查端口 ----
check_port() {
    local PORT=$1
    if ! docker info &>/dev/null; then
        return  # Docker 未运行，跳过端口检查
    fi
    # 检查宿主机端口
    if ss -tlnp 2>/dev/null | grep -q ":$PORT "; then
        echo -e "  ${RED}❌${NC} 端口 ${PORT} 被占用"
        echo "     请停止占用端口的程序，或修改 docker-compose.yml 中的端口映射"
        exit 1
    fi
    echo -e "  ${GREEN}✅${NC} 端口 ${PORT} 空闲"
}

# ---- 检查 .env ----
check_env() {
    if [ ! -f .env ]; then
        echo -e "  ${RED}❌${NC} .env 文件不存在"
        echo "     请执行: cp .env.example .env && nano .env"
        exit 1
    fi
    echo -e "  ${GREEN}✅${NC} .env 配置完成"

    # 读取并显示 API 地址
    API_BASE=$(grep -E '^EVAL_MODEL_API_BASE=' .env | head -1 | cut -d'=' -f2)
    echo -e "  ${BLUE}ℹ️${NC}  模型端点: ${API_BASE:-未配置}"
}

# ---- 检查 vLLM 连通性 ----
check_vllm() {
    local API_BASE=$(grep -E '^EVAL_MODEL_API_BASE=' .env 2>/dev/null | head -1 | cut -d'=' -f2)
    local API_KEY=$(grep -E '^EVAL_MODEL_API_KEY=' .env 2>/dev/null | head -1 | cut -d'=' -f2)

    if [ -z "$API_BASE" ]; then
        echo -e "  ${YELLOW}⚠️${NC} vLLM 端点未配置，跳过连通性测试"
        return
    fi

    local COMPLETIONS_URL="${API_BASE}/chat/completions"

    if curl -sf -o /dev/null --connect-timeout 5 "$COMPLETIONS_URL" 2>/dev/null \
        || curl -sf -o /dev/null --connect-timeout 5 "$(echo "$API_BASE" | sed 's|/v1$||')/v1/models" 2>/dev/null; then
        echo -e "  ${GREEN}✅${NC} vLLM 端点 ${API_BASE} 可达"
    else
        echo -e "  ${YELLOW}⚠️${NC} vLLM 端点 ${API_BASE} 不可达（不阻塞启动）"
        echo "     请确认 vLLM 已启动: curl ${API_BASE}/models"
    fi
}

# ---- 加载镜像 ----
load_image() {
    local COMPOSE_FILE="docker/docker-compose.yml"

    if [ ! -f "$COMPOSE_FILE" ]; then
        echo -e "  ${RED}❌${NC} docker-compose.yml 不存在"
        exit 1
    fi

    # 从 docker-compose.yml 读取版本号
    local IMAGE_TAG=$(grep 'image:' "$COMPOSE_FILE" | head -1 | sed 's/.*://' | tr -d ' ')
    local IMAGE_NAME="eval-app:${IMAGE_TAG}"

    echo -e "\n${BLUE}🚀 加载镜像...${NC}"

    # 检查镜像是否已存在
    if docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
        echo -e "  ${GREEN}✅${NC} ${IMAGE_NAME} (已存在)"
        return
    fi

    # 从 .tar 文件加载
    local TAR_FILE="images/eval-app-${IMAGE_TAG}.tar"
    if [ -f "$TAR_FILE" ]; then
        echo -e "  加载 ${TAR_FILE}..."
        docker load -i "$TAR_FILE"
        echo -e "  ${GREEN}✅${NC} ${IMAGE_NAME} (已加载)"
    else
        echo -e "  ${RED}❌${NC} 镜像文件不存在: ${TAR_FILE}"
        echo "     请将 images/eval-app-${IMAGE_TAG}.tar 放置到正确位置"
        exit 1
    fi
}

# ---- 启动服务 ----
start_service() {
    echo -e "\n${BLUE}🚀 启动服务...${NC}"

    # 确保结果目录存在
    mkdir -p results results/reports results/backups

    docker compose -f docker/docker-compose.yml up -d

    # 等待服务启动
    sleep 3

    if docker ps | grep -q "eval-app"; then
        echo -e "  ${GREEN}✅${NC} eval-app: started"
    else
        echo -e "  ${RED}❌${NC} 启动失败，查看日志: docker logs eval-app"
        exit 1
    fi
}

# ---- 主流程 ----
main() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE} AI 能力评估系统 - 一键启动${NC}"
    echo -e "${BLUE}========================================${NC}\n"

    check_docker
    check_storage_driver "$1"
    check_disk
    check_port 8000
    check_env
    check_vllm

    load_image

    start_service

    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN} ✅ 评估系统就绪！${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "   📊 运行评测:"
    echo -e "      ${BLUE}docker exec -it eval-app eval run llm --bench mmlu${NC}"
    echo ""
    echo -e "   📋 查看可用 benchmark:"
    echo -e "      ${BLUE}docker exec -it eval-app eval dataset list${NC}"
    echo ""
    echo -e "   🔌 测试模型连通性:"
    echo -e "      ${BLUE}docker exec -it eval-app eval model test${NC}"
    echo ""
    echo -e "   📄 查看帮助:"
    echo -e "      ${BLUE}docker exec -it eval-app eval --help${NC}"
    echo ""
}

main "$@"
