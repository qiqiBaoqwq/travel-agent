"""FastAPI主应用"""

from fastapi import FastAPI

from app.core.cors import set_cors
from app.core.config import get_settings, validate_config, print_config
from app.core.logging_config import setup_logging, get_logger, shutdown_logging
from app.core.logging_middleware import LoggingMiddleware
from app.api.endpoints import map as map_routes
from app.api.endpoints import trip, poi
from contextlib import asynccontextmanager

# 获取配置
settings = get_settings()

# 配置日志（应用启动时立即配置）
setup_logging(log_dir="logs", log_level=settings.log_level)
logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ==============================
    # 🟢 启动阶段 (Startup)
    # 运行在应用开始接收请求之前
    # ==============================
    logger.info("=" * 60 + f"\n🚀 {settings.app_name} v{settings.app_version}\n" + "=" * 60)
  
    # 打印配置信息
    print_config()

    # 验证配置
    try:
        validate_config()
        logger.info("✅ 配置验证通过")
    except ValueError as e:
        logger.error(f"❌ 配置验证失败: {e}")
        logger.error("请检查.env文件并确保所有必要的配置项都已设置")
        # 这里抛出异常会阻止应用启动，非常安全
        raise

    logger.info("=" * 60 + "\n📚 API文档: http://localhost:8000/docs" + "\n"
                + "📖 ReDoc文档: http://localhost:8000/redoc" + "\n" + "=" * 60)
    # 👉 关键点：yield 将启动和关闭逻辑分开
    # 如果你需要共享数据库连接等资源，可以 yield { "db": db_connection }
    yield

    # ==============================
    # 🔴 关闭阶段 (Shutdown)
    # 运行在应用停止接收请求之后
    # ==============================
    logger.info("=" * 60 + "\n👋 应用正在关闭... \n" + "=" * 60)
    # 关闭日志系统，确保所有日志被刷新
    shutdown_logging()

# 创建FastAPI应用
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="基于HelloAgents框架的智能旅行规划助手API",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# 添加中间件
app.add_middleware(LoggingMiddleware)

# 注册路由
app.include_router(trip.router, prefix="/api")
app.include_router(poi.router, prefix="/api")
app.include_router(map_routes.router, prefix="/api")

set_cors(app)
@app.get("/")
async def root():
    """根路径"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True
    )