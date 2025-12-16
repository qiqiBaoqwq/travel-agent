"""请求日志中间件"""

import time
import uuid
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from app.core.logging_config import get_logger, set_request_id

logger = get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """HTTP请求日志中间件"""

    async def dispatch(self, request: Request, call_next):
        # 生成请求ID
        request_id = str(uuid.uuid4())[:8]

        # 将请求ID设置到上下文中（关键步骤！）
        set_request_id(request_id)

        # 记录请求开始
        logger.info(
            f"START | {request.method} {request.url.path} | "
            f"client={request.client.host if request.client else 'unknown'}"
        )

        # 记录开始时间
        start_time = time.time()

        # 处理请求
        try:
            response = await call_next(request)

            # 计算处理时间
            process_time = (time.time() - start_time) * 1000

            # 记录请求完成
            logger.info(
                f"END | {request.method} {request.url.path} | "
                f"status={response.status_code} | time={process_time:.2f}ms"
            )

            # 添加请求ID到响应头
            response.headers["X-Request-ID"] = request_id

            return response

        except Exception as e:
            # 记录错误
            process_time = (time.time() - start_time) * 1000
            logger.error(
                f"ERROR | {request.method} {request.url.path} | "
                f"time={process_time:.2f}ms | error={str(e)}",
                exc_info=True
            )
            raise
        finally:
            # 清理上下文
            set_request_id('')

