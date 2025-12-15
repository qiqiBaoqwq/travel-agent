from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from app.core.config import get_settings

def set_cors(app:FastAPI):
# 配置CORS - 跨域资源共享设置
# 允许前端应用(如React、Vue等)访问后端API
    app.add_middleware(
        CORSMiddleware,
        # 允许访问的源列表，从环境变量中获取
        allow_origins=get_settings().get_cors_origins_list(),
        # 是否允许携带认证信息(如cookies)
        allow_credentials=True,
        # 允许的HTTP方法(*表示所有方法)
        allow_methods=["*"],
        # 允许的请求头(*表示所有请求头)
        allow_headers=["*"],
    )