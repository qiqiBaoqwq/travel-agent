"""Unsplash图片服务"""

import requests
import hashlib
from typing import List, Optional, Dict
from app.core.config import get_settings


# 简单的内存缓存
_photo_cache: Dict[str, str] = {}


class UnsplashService:
    """Unsplash图片服务类"""
    
    def __init__(self):
        """初始化服务"""
        settings = get_settings()
        self.access_key = settings.unsplash_access_key
        self.base_url = "https://api.unsplash.com"
    
    def search_photos(self, query: str, per_page: int = 5) -> List[dict]:
        """
        搜索图片
        
        Args:
            query: 搜索关键词
            per_page: 每页数量
            
        Returns:
            图片列表
        """
        try:
            url = f"{self.base_url}/search/photos"
            params = {
                "query": query,
                "per_page": per_page,
                "client_id": self.access_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = data.get("results", [])
            
            # 提取图片URL
            photos = []
            for photo in results:
                photos.append({
                    "id": photo.get("id"),
                    "url": photo.get("urls", {}).get("regular"),
                    "thumb": photo.get("urls", {}).get("thumb"),
                    "small": photo.get("urls", {}).get("small"),
                    "description": photo.get("description") or photo.get("alt_description"),
                    "photographer": photo.get("user", {}).get("name")
                })
            
            return photos
            
        except Exception as e:
            print(f"❌ Unsplash搜索失败: {str(e)}")
            return []
    
    def get_photo_url(self, query: str, use_cache: bool = True) -> Optional[str]:
        """
        获取单张图片URL（带缓存）

        Args:
            query: 搜索关键词
            use_cache: 是否使用缓存

        Returns:
            图片URL
        """
        global _photo_cache
        
        # 生成缓存key
        cache_key = hashlib.md5(query.encode()).hexdigest()
        
        # 检查缓存
        if use_cache and cache_key in _photo_cache:
            return _photo_cache[cache_key]
        
        # 调用API
        photos = self.search_photos(query, per_page=1)
        if photos:
            # 使用small尺寸，加载更快
            url = photos[0].get("small") or photos[0].get("url")
            # 存入缓存
            _photo_cache[cache_key] = url
            return url
        return None
    
    def batch_get_photo_urls(self, names: List[str]) -> Dict[str, Optional[str]]:
        """
        批量获取图片URL
        
        Args:
            names: 景点名称列表
            
        Returns:
            名称到URL的映射
        """
        import concurrent.futures
        
        results = {}
        
        def fetch_photo(name: str) -> tuple:
            url = self.get_photo_url(f"{name} China landmark")
            if not url:
                url = self.get_photo_url(name)
            return (name, url)
        
        # 使用线程池并行获取
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(fetch_photo, name) for name in names]
            for future in concurrent.futures.as_completed(futures):
                try:
                    name, url = future.result()
                    results[name] = url
                except Exception as e:
                    print(f"❌ 批量获取图片失败: {str(e)}")
        
        return results


# 全局服务实例
_unsplash_service = None


def get_unsplash_service() -> UnsplashService:
    """获取Unsplash服务实例(单例模式)"""
    global _unsplash_service
    
    if _unsplash_service is None:
        _unsplash_service = UnsplashService()
    
    return _unsplash_service

