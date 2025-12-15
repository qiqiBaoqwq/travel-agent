"""FastAPI主应用"""

from fastapi import FastAPI

from app.core.cors import set_cors
from app.core.config import get_settings, validate_config, print_config
from app.api.endpoints import map as map_routes
from app.api.endpoints import trip, poi
from contextlib import asynccontextmanager

# 获取配置
settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ==============================
    # 🟢 启动阶段 (Startup)
    # 运行在应用开始接收请求之前
    # ==============================
    print("\n" + "=" * 60)
    print(f"🚀 {settings.app_name} v{settings.app_version}")
    print("=" * 60)
  
    # 打印配置信息
    print_config()

    # 验证配置
    try:
        validate_config()
        print("\n✅ 配置验证通过")
    except ValueError as e:
        print(f"\n❌ 配置验证失败:\n{e}")
        print("\n请检查.env文件并确保所有必要的配置项都已设置")
        # 这里抛出异常会阻止应用启动，非常安全
        raise

    print("\n" + "=" * 60)
    print("📚 API文档: http://localhost:8000/docs")
    print("📖 ReDoc文档: http://localhost:8000/redoc")
    print("=" * 60 + "\n")

    # 👉 关键点：yield 将启动和关闭逻辑分开
    # 如果你需要共享数据库连接等资源，可以 yield { "db": db_connection }
    yield

    # ==============================
    # 🔴 关闭阶段 (Shutdown)
    # 运行在应用停止接收请求之后
    # ==============================
    print("\n" + "=" * 60)
    print("👋 应用正在关闭...")
    # 如果有数据库连接，在这里执行 db.close()
    print("=" * 60 + "\n")


# 创建FastAPI应用
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="基于HelloAgents框架的智能旅行规划助手API",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

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

