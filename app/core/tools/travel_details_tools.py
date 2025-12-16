import json

from langchain_core.tools import tool
import httpx

from app.core.config import get_settings


# ============ 工具定义 ============

@tool
def search_attractions(keywords: str, city: str) -> str:
    """
    搜索景点信息

    Args:
        keywords: 搜索关键词，如"历史文化"、"公园"、"美食"等
        city: 城市名称，如"北京"、"上海"等

    Returns:
        景点搜索结果的JSON字符串
    """
    settings = get_settings()
    api_key = settings.amap_api_key

    url = "https://restapi.amap.com/v3/place/text"
    params = {
        "key": api_key,
        "keywords": keywords,
        "city": city,
        "citylimit": "true",
        "offset": 20,
        "extensions": "all"
    }

    try:
        with httpx.Client(timeout=30) as client:
            response = client.get(url, params=params)
            data = response.json()

            if data.get("status") == "1" and data.get("pois"):
                pois = data["pois"][:10]
                results = []
                for poi in pois:
                    location = poi.get("location", "").split(",")
                    results.append({
                        "name": poi.get("name", ""),
                        "address": poi.get("address", ""),
                        "type": poi.get("type", ""),
                        "tel": poi.get("tel", ""),
                        "location": {
                            "longitude": float(location[0]) if len(
                                location) == 2 else 0,
                            "latitude": float(location[1]) if len(
                                location) == 2 else 0
                        }
                    })
                return json.dumps(results, ensure_ascii=False, indent=2)
            else:
                return json.dumps({"error": "未找到相关景点", "city": city,
                                   "keywords": keywords}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool
def search_weather(city: str) -> str:
    """
    查询城市天气信息

    Args:
        city: 城市名称，如"北京"、"上海"等

    Returns:
        天气信息的JSON字符串
    """
    settings = get_settings()
    api_key = settings.amap_api_key

    url = "https://restapi.amap.com/v3/weather/weatherInfo"
    params = {
        "key": api_key,
        "city": city,
        "extensions": "all"
    }

    try:
        with httpx.Client(timeout=30) as client:
            response = client.get(url, params=params)
            data = response.json()

            if data.get("status") == "1" and data.get("forecasts"):
                forecasts = data["forecasts"][0]
                casts = forecasts.get("casts", [])
                results = {
                    "city": forecasts.get("city", city),
                    "province": forecasts.get("province", ""),
                    "forecasts": [
                        {
                            "date": cast.get("date", ""),
                            "week": cast.get("week", ""),
                            "dayweather": cast.get("dayweather", ""),
                            "nightweather": cast.get("nightweather", ""),
                            "daytemp": cast.get("daytemp", ""),
                            "nighttemp": cast.get("nighttemp", ""),
                            "daywind": cast.get("daywind", ""),
                            "nightwind": cast.get("nightwind", ""),
                            "daypower": cast.get("daypower", ""),
                            "nightpower": cast.get("nightpower", "")
                        }
                        for cast in casts
                    ]
                }
                return json.dumps(results, ensure_ascii=False, indent=2)
            else:
                return json.dumps({"error": "未找到天气信息", "city": city},
                                  ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool
def search_hotels(city: str, hotel_type: str = "酒店") -> str:
    """
    搜索酒店信息

    Args:
        city: 城市名称，如"北京"、"上海"等
        hotel_type: 酒店类型，如"经济型酒店"、"豪华酒店"等

    Returns:
        酒店搜索结果的JSON字符串
    """
    settings = get_settings()
    api_key = settings.amap_api_key

    url = "https://restapi.amap.com/v3/place/text"
    params = {
        "key": api_key,
        "keywords": hotel_type,
        "city": city,
        "citylimit": "true",
        "types": "100000",
        "offset": 10,
        "extensions": "all"
    }

    try:
        with httpx.Client(timeout=30) as client:
            response = client.get(url, params=params)
            data = response.json()

            if data.get("status") == "1" and data.get("pois"):
                pois = data["pois"][:8]
                results = []
                for poi in pois:
                    location = poi.get("location", "").split(",")
                    results.append({
                        "name": poi.get("name", ""),
                        "address": poi.get("address", ""),
                        "type": poi.get("type", ""),
                        "tel": poi.get("tel", ""),
                        "location": {
                            "longitude": float(location[0]) if len(
                                location) == 2 else 0,
                            "latitude": float(location[1]) if len(
                                location) == 2 else 0
                        }
                    })
                return json.dumps(results, ensure_ascii=False, indent=2)
            else:
                return json.dumps({"error": "未找到相关酒店", "city": city},
                                  ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)
