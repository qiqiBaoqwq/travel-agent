"""旅行规划API路由"""

from fastapi import APIRouter, HTTPException
from  app.schemas.travel_plan_related_schemas  import (
    TripRequest,
    TripPlan,
    AppResponse
)
from app.core.agents.trip_planner_agent import get_trip_planner_agent
from app.core.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/trip", tags=["旅行规划"])


@router.post(
    "/plan",
    response_model=AppResponse[TripPlan],
    summary="生成旅行计划",
    description="根据用户输入的旅行需求,生成详细的旅行计划"
)
async def plan_trip(request: TripRequest):
    """
    生成旅行计划

    Args:
        request: 旅行请求参数

    Returns:
        旅行计划响应
    """
    try:
        logger.info("="*60)
        logger.info("📥 收到旅行规划请求:")
        logger.info(f"   城市: {request.city}")
        logger.info(f"   日期: {request.start_date} - {request.end_date}")
        logger.info(f"   天数: {request.travel_days}")
        logger.info("="*60)

        # 获取Agent实例
        logger.info("🔄 获取LangGraph多智能体系统实例...")
        agent = get_trip_planner_agent()

        # 生成旅行计划
        logger.info("🚀 开始生成旅行计划...")
        trip_plan = agent.plan_trip(request)

        logger.info("✅ 旅行计划生成成功,准备返回响应")

        return AppResponse.success(data=trip_plan, message="旅行计划生成成功")

    except Exception as e:
        logger.error(f"❌ 生成旅行计划失败: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"生成旅行计划失败: {str(e)}"
        )


@router.get(
    "/health",
    summary="健康检查",
    description="检查旅行规划服务是否正常"
)
async def health_check():
    """健康检查"""
    try:
        # 检查Agent是否可用
        agent = get_trip_planner_agent()
        
        return AppResponse.success(data={
            "status": "healthy",
            "service": "trip-planner",
            "type": "LangGraph"
        })
    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=f"服务不可用: {str(e)}"
        )