@echo off
REM =============================================================================
REM AI 能力评估系统 — Windows 一键启动脚本
REM Windows 系统需要 Docker Desktop (WSL2 后端)
REM =============================================================================

echo ========================================
echo  AI 能力评估系统 - 一键启动 (Windows)
echo ========================================

echo.
echo 🔍 环境预检...

REM 检查 Docker
where docker >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ❌ docker: command not found
    echo    请先安装 Docker Desktop: https://www.docker.com/products/docker-desktop/
    exit /b 1
)
echo   ✅ Docker

REM 检查 .env
if not exist .env (
    echo ❌ .env 文件不存在
    echo    请执行: copy .env.example .env
    exit /b 1
)
echo   ✅ .env 配置完成

REM 检查镜像
set COMPOSE_FILE=docker\docker-compose.yml
if not exist %COMPOSE_FILE% (
    echo ❌ docker-compose.yml 不存在
    exit /b 1
)

echo.
echo 🚀 启动服务...
mkdir results 2>nul
mkdir results\reports 2>nul
mkdir results\backups 2>nul

docker compose -f %COMPOSE_FILE% up -d

timeout /t 3 /nobreak >nul

docker ps | findstr "eval-app" >nul
if %ERRORLEVEL% EQU 0 (
    echo   ✅ eval-app: started
) else (
    echo   ❌ 启动失败，查看日志: docker logs eval-app
    exit /b 1
)

echo.
echo ========================================
echo  ✅ 评估系统就绪！
echo ========================================
echo.
echo   📊 运行评测:
echo      docker exec -it eval-app eval run llm --bench mmlu
echo.
echo   🔌 测试模型连通性:
echo      docker exec -it eval-app eval model test
echo.
