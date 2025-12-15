"""POI相关API路由"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from app.services.amap_service import get_amap_service
from app.services.unsplash_service import get_unsplash_service
from app.schemas.travel_plan_related_schemas import AppResponse, POIInfo

router = APIRouter(prefix="/poi", tags=["POI"])


class BatchPhotoRequest(BaseModel):
    """批量获取图片请求"""
    names: List[str] = Field(..., description="景点名称列表")


@router.get(
    "/detail/{poi_id}",
    response_model=AppResponse[dict],
    summary="获取POI详情",
    description="根据POI ID获取详细信息,包括图片"
)
async def get_poi_detail(poi_id: str):
    """
    获取POI详情
    
    Args:
        poi_id: POI ID
        
    Returns:
        POI详情响应
    """
    try:
        amap_service = get_amap_service()
        
        # 调用高德地图POI详情API
        result = amap_service.get_poi_detail(poi_id)
        
        return AppResponse.success(data=result, message="获取POI详情成功")
        
    except Exception as e:
        print(f"❌ 获取POI详情失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"获取POI详情失败: {str(e)}"
        )


@router.get(
    "/search",
    response_model=AppResponse[List[POIInfo]],
    summary="搜索POI",
    description="根据关键词搜索POI"
)
async def search_poi(keywords: str, city: str = "北京"):
    """
    搜索POI

    Args:
        keywords: 搜索关键词
        city: 城市名称

    Returns:
        搜索结果
    """
    try:
        amap_service = get_amap_service()
        result = amap_service.search_poi(keywords, city)

        return AppResponse.success(data=result, message="搜索成功")

    except Exception as e:
        print(f"❌ 搜索POI失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"搜索POI失败: {str(e)}"
        )


@router.get(
    "/photo",
    response_model=AppResponse[dict],
    summary="获取景点图片",
    description="根据景点名称从Unsplash获取图片"
)
async def get_attraction_photo(name: str):
    """
    获取景点图片

    Args:
        name: 景点名称

    Returns:
        图片URL
    """
    try:
        unsplash_service = get_unsplash_service()

        # 搜索景点图片
        photo_url = unsplash_service.get_photo_url(f"{name} China landmark")

        if not photo_url:
            # 如果没找到,尝试只用景点名称搜索
            photo_url = unsplash_service.get_photo_url(name)

        return AppResponse.success(data={
            "name": name,
            "photo_url": photo_url
        }, message="获取图片成功")

    except Exception as e:
        print(f"❌ 获取景点图片失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"获取景点图片失败: {str(e)}"
        )


@router.post(
    "/photos/batch",
    response_model=AppResponse[Dict[str, str]],
    summary="批量获取景点图片",
    description="批量根据景点名称从Unsplash获取图片（并行处理，更快）"
)
async def get_attraction_photos_batch(request: BatchPhotoRequest):
    """
    批量获取景点图片
    
    Args:
        request: 包含景点名称列表的请求
        
    Returns:
        景点名称到图片URL的映射
    """
    try:
        unsplash_service = get_unsplash_service()
        
        # 使用批量获取方法（并行处理）
        photos = unsplash_service.batch_get_photo_urls(request.names)
        
        return AppResponse.success(data=photos, message=f"成功获取 {len(photos)} 张图片")
        
    except Exception as e:
        print(f"❌ 批量获取景点图片失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"批量获取景点图片失败: {str(e)}"
        )

