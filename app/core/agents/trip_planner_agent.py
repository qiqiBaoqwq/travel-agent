"""基于LangChain和LangGraph的多智能体旅行规划系统"""

import os
import json
import httpx
import operator
from typing import Dict, Any, List, Annotated, TypedDict, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages

from  app.schemas.travel_plan_related_schemas  import (TripRequest, TripPlan, DayPlan, Attraction,
                               Meal,
                        WeatherInfo, Location, Hotel)
from  app.core.config import get_settings


# ============ 状态定义 ============

def merge_dicts(left: Dict[str, str], right: Dict[str, str]) -> Dict[str, str]:
    """合并字典，用于收集各个Agent的结果"""
    result = left.copy()
    result.update(right)
    return result

class TripPlannerState(TypedDict):
    """旅行规划器状态"""
    messages: Annotated[List[BaseMessage], add_messages]
    request: Dict[str, Any]
    # 任务规划
    task_plan: str
    # 各Agent收集的数据
    agent_results: Annotated[Dict[str, str], merge_dicts]
    # 完成的任务计数
    completed_tasks: Annotated[List[str], operator.add]
    # 最终计划
    final_plan: str
    current_step: str


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
                            "longitude": float(location[0]) if len(location) == 2 else 0,
                            "latitude": float(location[1]) if len(location) == 2 else 0
                        }
                    })
                return json.dumps(results, ensure_ascii=False, indent=2)
            else:
                return json.dumps({"error": "未找到相关景点", "city": city, "keywords": keywords}, ensure_ascii=False)
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
                return json.dumps({"error": "未找到天气信息", "city": city}, ensure_ascii=False)
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
                            "longitude": float(location[0]) if len(location) == 2 else 0,
                            "latitude": float(location[1]) if len(location) == 2 else 0
                        }
                    })
                return json.dumps(results, ensure_ascii=False, indent=2)
            else:
                return json.dumps({"error": "未找到相关酒店", "city": city}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# ============ Agent提示词 ============

SCHEDULER_AGENT_PROMPT = """你是旅行规划调度专家。你的职责是：
1. 分析用户的旅行需求
2. 制定任务计划，明确需要收集哪些信息
3. 协调其他专业Agent完成各自的任务

请根据用户需求，输出一个清晰的任务规划，说明：
- 需要搜索什么类型的景点
- 需要查询哪个城市的天气
- 需要搜索什么类型的酒店

格式：
```
任务规划:
1. 景点搜索: [关键词] - [城市]
2. 天气查询: [城市]
3. 酒店搜索: [酒店类型] - [城市]
```
"""

ATTRACTION_AGENT_PROMPT = """你是景点搜索专家。你的任务是根据城市和用户偏好搜索合适的景点。

请使用 search_attractions 工具来搜索景点信息。

**注意:**
1. 必须使用工具来获取真实的景点数据
2. 根据用户偏好选择合适的关键词进行搜索
3. 整理搜索结果并给出清晰的景点列表
"""

WEATHER_AGENT_PROMPT = """你是天气查询专家。你的任务是查询指定城市的天气信息。

请使用 search_weather 工具来查询天气。

**注意:**
1. 必须使用工具来获取真实的天气数据
2. 整理天气信息并给出未来几天的天气预报
"""

HOTEL_AGENT_PROMPT = """你是酒店推荐专家。你的任务是根据城市和用户需求搜索合适的酒店。

请使用 search_hotels 工具来搜索酒店。

**注意:**
1. 必须使用工具来获取真实的酒店数据
2. 根据用户的住宿偏好选择合适的酒店类型
"""

SUMMARIZER_AGENT_PROMPT = """你是行程规划专家。你的任务是根据景点信息、天气信息和酒店信息，生成详细的旅行计划。

**重要: 必须返回完整的JSON，不能截断！如果内容太长，请精简描述而不是省略结构。**

请严格按照以下JSON格式返回旅行计划:
```json
{
  "city": "城市名称",
  "start_date": "YYYY-MM-DD",
  "end_date": "YYYY-MM-DD",
  "days": [
    {
      "date": "YYYY-MM-DD",
      "day_index": 0,
      "description": "简短行程概述(20字内)",
      "transportation": "交通方式",
      "accommodation": "住宿类型",
      "hotel": {
        "name": "酒店名称",
        "address": "简短地址",
        "location": {"longitude": 116.39, "latitude": 39.91},
        "price_range": "300-500元",
        "rating": "4.5",
        "type": "酒店类型"
      },
      "attractions": [
        {
          "name": "景点名",
          "address": "简短地址",
          "location": {"longitude": 116.39, "latitude": 39.91},
          "visit_duration": 120,
          "description": "简短描述(30字内)",
          "category": "类别"
        }
      ],
      "meals": [
        {"type": "breakfast", "name": "早餐", "description": "简短描述"},
        {"type": "lunch", "name": "午餐", "description": "简短描述"},
        {"type": "dinner", "name": "晚餐", "description": "简短描述"}
      ]
    }
  ],
  "weather_info": [{"date": "YYYY-MM-DD", "day_weather": "晴", "night_weather": "多云", "day_temp": 25, "night_temp": 15}],
  "overall_suggestions": "简短建议(50字内)",
  "budget": {"total": 2000}
}
```

**要求:**
1. 所有描述尽量精简，避免长文本
2. 温度必须是纯数字
3. 每天安排2-3个景点
4. JSON必须完整闭合，确保所有括号配对
5. 不要添加注释或额外说明
"""


class MultiAgentTripPlanner:
    """基于LangGraph的多智能体旅行规划系统 - Scheduler协调模式
    
    架构:
    1. Scheduler Agent 分析任务并制定计划
    2. 并行分发任务给 Weather/Hotel/Attraction Agents
    3. 收集所有结果后，Scheduler Agent 进行总结
    """

    def __init__(self):
        """初始化多智能体系统"""
        print("🔄 开始初始化LangGraph多智能体旅行规划系统(Scheduler模式)...")

        try:
            # 从环境变量获取配置
            api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            model_id = os.getenv("LLM_MODEL_ID") or os.getenv("OPENAI_MODEL", "gpt-4")
            
            if not api_key:
                raise ValueError("未配置LLM API密钥")
            
            # 创建LLM实例 - 增加max_tokens确保输出完整
            self.llm = ChatOpenAI(
                model=model_id,
                api_key=api_key,
                base_url=base_url,
                temperature=0.7,
                timeout=300,
                max_tokens=8192,  # 确保有足够的输出空间
            )
            
            print(f"✅ LLM服务初始化成功")
            print(f"   模型: {model_id}")
            print(f"   Base URL: {base_url}")
            print(f"   Max Tokens: 8192")
            
            # 定义工具
            self.tools = [search_attractions, search_weather, search_hotels]
            self.tools_map = {t.name: t for t in self.tools}
            
            # 创建带工具的LLM
            self.llm_with_tools = self.llm.bind_tools(self.tools)
            
            # 创建工作流
            self._build_graph()
            
            print(f"✅ LangGraph多智能体系统初始化成功(Scheduler模式)")
            print(f"   可用工具: {[t.name for t in self.tools]}")
            print(f"   架构: Scheduler -> [Weather, Hotel, Attraction] -> Summarizer")

        except Exception as e:
            print(f"❌ 多智能体系统初始化失败: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def _build_graph(self):
        """构建LangGraph工作流 - Scheduler协调模式
        
        流程:
        START -> scheduler_plan -> [attraction_agent, weather_agent, hotel_agent] (并行)
              -> collector -> scheduler_summarize -> END
        """
        
        workflow = StateGraph(TripPlannerState)
        
        # 添加节点
        workflow.add_node("scheduler_plan", self._scheduler_plan_node)
        workflow.add_node("attraction_agent", self._attraction_agent_node)
        workflow.add_node("weather_agent", self._weather_agent_node)
        workflow.add_node("hotel_agent", self._hotel_agent_node)
        workflow.add_node("collector", self._collector_node)
        workflow.add_node("scheduler_summarize", self._scheduler_summarize_node)
        
        # 定义边 - Scheduler规划后并行执行
        workflow.add_edge(START, "scheduler_plan")
        
        # Scheduler规划完成后，并行分发到三个Agent
        workflow.add_edge("scheduler_plan", "attraction_agent")
        workflow.add_edge("scheduler_plan", "weather_agent")
        workflow.add_edge("scheduler_plan", "hotel_agent")
        
        # 三个Agent完成后汇集到collector
        workflow.add_edge("attraction_agent", "collector")
        workflow.add_edge("weather_agent", "collector")
        workflow.add_edge("hotel_agent", "collector")
        
        # Collector检查是否所有任务完成，然后到Summarizer
        workflow.add_conditional_edges(
            "collector",
            self._check_all_tasks_completed,
            {"wait": "collector", "summarize": "scheduler_summarize"}
        )
        
        workflow.add_edge("scheduler_summarize", END)
        
        self.graph = workflow.compile()

    def _scheduler_plan_node(self, state: TripPlannerState) -> Dict[str, Any]:
        """Scheduler Agent - 规划阶段: 分析任务并制定计划"""
        print("\n📋 Scheduler Agent: 分析任务并制定计划...")
        
        request = state["request"]
        
        query = f"""请分析以下旅行需求并制定任务计划:

**旅行需求:**
- 城市: {request['city']}
- 日期: {request['start_date']} 至 {request['end_date']}
- 天数: {request['travel_days']}天
- 交通方式: {request.get('transportation', '公共交通')}
- 住宿偏好: {request.get('accommodation', '经济型酒店')}
- 偏好标签: {', '.join(request.get('preferences', [])) or '无'}

请制定任务计划。"""
        
        messages = [
            SystemMessage(content=SCHEDULER_AGENT_PROMPT),
            HumanMessage(content=query)
        ]
        
        response = self.llm.invoke(messages)
        
        print(f"   任务计划: {response.content[:200]}...")
        
        return {
            "messages": messages + [response],
            "task_plan": response.content,
            "current_step": "planning",
            "completed_tasks": [],
            "agent_results": {}
        }

    def _attraction_agent_node(self, state: TripPlannerState) -> Dict[str, Any]:
        """景点搜索Agent - 直接调用工具获取数据"""
        print("📍 Attraction Agent: 搜索景点...")
        
        request = state["request"]
        preferences = request.get("preferences", [])
        keywords = preferences[0] if preferences else "景点"
        city = request["city"]
        
        # 直接调用工具
        result = search_attractions.invoke({"keywords": keywords, "city": city})
        
        print(f"   景点搜索完成: {result[:100]}...")
        
        return {
            "agent_results": {"attractions": result},
            "completed_tasks": ["attraction"]
        }

    def _weather_agent_node(self, state: TripPlannerState) -> Dict[str, Any]:
        """天气查询Agent - 直接调用工具获取数据"""
        print("🌤️  Weather Agent: 查询天气...")
        
        request = state["request"]
        city = request["city"]
        
        # 直接调用工具
        result = search_weather.invoke({"city": city})
        
        print(f"   天气查询完成: {result[:100]}...")
        
        return {
            "agent_results": {"weather": result},
            "completed_tasks": ["weather"]
        }

    def _hotel_agent_node(self, state: TripPlannerState) -> Dict[str, Any]:
        """酒店推荐Agent - 直接调用工具获取数据"""
        print("🏨 Hotel Agent: 搜索酒店...")
        
        request = state["request"]
        city = request["city"]
        accommodation = request.get("accommodation", "酒店")
        
        # 直接调用工具
        result = search_hotels.invoke({"city": city, "hotel_type": accommodation})
        
        print(f"   酒店搜索完成: {result[:100]}...")
        
        return {
            "agent_results": {"hotels": result},
            "completed_tasks": ["hotel"]
        }

    def _collector_node(self, state: TripPlannerState) -> Dict[str, Any]:
        """收集器节点 - 等待所有Agent完成"""
        completed = state.get("completed_tasks", [])
        print(f"📦 Collector: 已完成任务 {completed}")
        
        # 不做修改，只是一个同步点
        return {}

    def _check_all_tasks_completed(self, state: TripPlannerState) -> Literal["wait", "summarize"]:
        """检查是否所有任务都已完成"""
        completed = state.get("completed_tasks", [])
        required_tasks = {"attraction", "weather", "hotel"}
        
        if required_tasks.issubset(set(completed)):
            print("✅ 所有任务已完成，准备总结...")
            return "summarize"
        else:
            remaining = required_tasks - set(completed)
            print(f"⏳ 等待任务完成: {remaining}")
            return "wait"

    def _scheduler_summarize_node(self, state: TripPlannerState) -> Dict[str, Any]:
        """Scheduler Agent - 总结阶段: 整合所有信息生成最终计划"""
        print("\n📋 Scheduler Agent: 整合信息并生成最终计划...")
        
        request = state["request"]
        agent_results = state.get("agent_results", {})
        
        attractions = agent_results.get("attractions", "")
        weather = agent_results.get("weather", "")
        hotels = agent_results.get("hotels", "")
        
        query = f"""请根据以下收集到的信息，生成{request['city']}的{request['travel_days']}天旅行计划:

**基本信息:**
- 城市: {request['city']}
- 日期: {request['start_date']} 至 {request['end_date']}
- 天数: {request['travel_days']}天
- 交通方式: {request.get('transportation', '公共交通')}
- 住宿: {request.get('accommodation', '经济型酒店')}
- 偏好: {', '.join(request.get('preferences', [])) or '无'}

**景点信息:**
{attractions}

**天气信息:**
{weather}

**酒店信息:**
{hotels}

请返回完整的JSON格式旅行计划。"""
        
        messages = [
            SystemMessage(content=SUMMARIZER_AGENT_PROMPT),
            HumanMessage(content=query)
        ]
        
        response = self.llm.invoke(messages)
        
        print(f"   最终计划生成完成")
        
        return {
            "messages": messages + [response],
            "final_plan": response.content,
            "current_step": "done"
        }

    def plan_trip(self, request: TripRequest) -> TripPlan:
        """
        使用多智能体协作生成旅行计划

        Args:
            request: 旅行请求

        Returns:
            旅行计划
        """
        try:
            print(f"\n{'='*60}")
            print(f"🚀 开始Scheduler模式多智能体协作规划旅行...")
            print(f"目的地: {request.city}")
            print(f"日期: {request.start_date} 至 {request.end_date}")
            print(f"天数: {request.travel_days}天")
            print(f"偏好: {', '.join(request.preferences) if request.preferences else '无'}")
            print(f"{'='*60}\n")
            
            # 初始状态
            initial_state: TripPlannerState = {
                "messages": [],
                "request": request.model_dump(),
                "task_plan": "",
                "agent_results": {},
                "completed_tasks": [],
                "final_plan": "",
                "current_step": "start"
            }
            
            # 执行工作流
            final_state = self.graph.invoke(initial_state)
            
            # 解析最终计划
            final_plan = final_state.get("final_plan", "")
            print(f"\n行程规划结果: {final_plan[:300]}...\n")
            
            trip_plan = self._parse_response(final_plan, request)

            print(f"{'='*60}")
            print(f"✅ 旅行计划生成完成!")
            print(f"{'='*60}\n")

            return trip_plan

        except Exception as e:
            print(f"❌ 生成旅行计划失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._create_fallback_plan(request)
    
    def _parse_response(self, response: str, request: TripRequest) -> TripPlan:
        """
        解析Agent响应
        
        Args:
            response: Agent响应文本
            request: 原始请求
            
        Returns:
            旅行计划
        """
        try:
            # 尝试从响应中提取JSON
            json_str = self._extract_json(response)
            
            if not json_str:
                raise ValueError("响应中未找到JSON数据")
            
            # 尝试修复不完整的JSON
            json_str = self._fix_incomplete_json(json_str)
            
            # 解析JSON
            data = json.loads(json_str)
            
            # 确保必要字段存在
            data = self._ensure_required_fields(data, request)
            
            # 转换为TripPlan对象
            trip_plan = TripPlan(**data)
            
            return trip_plan
            
        except Exception as e:
            print(f"⚠️  解析响应失败: {str(e)}")
            print(f"   将使用备用方案生成计划")
            return self._create_fallback_plan(request)
    
    def _extract_json(self, response: str) -> str:
        """从响应中提取JSON字符串"""
        # 方法1: 查找 ```json 代码块
        if "```json" in response:
            json_start = response.find("```json") + 7
            json_end = response.find("```", json_start)
            if json_end > json_start:
                return response[json_start:json_end].strip()
        
        # 方法2: 查找 ``` 代码块
        if "```" in response:
            json_start = response.find("```") + 3
            # 跳过可能的语言标识符
            newline_pos = response.find("\n", json_start)
            if newline_pos > json_start and newline_pos - json_start < 20:
                json_start = newline_pos + 1
            json_end = response.find("```", json_start)
            if json_end > json_start:
                return response[json_start:json_end].strip()
        
        # 方法3: 直接查找JSON对象
        if "{" in response:
            json_start = response.find("{")
            json_end = response.rfind("}")
            if json_end > json_start:
                return response[json_start:json_end + 1]
        
        return ""
    
    def _fix_incomplete_json(self, json_str: str) -> str:
        """尝试修复不完整的JSON"""
        import re
        
        # 移除可能的尾部省略号
        json_str = re.sub(r'\.{3,}$', '', json_str.strip())
        
        # 统计括号
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        open_brackets = json_str.count('[')
        close_brackets = json_str.count(']')
        
        # 如果JSON被截断，尝试修复
        if open_braces != close_braces or open_brackets != close_brackets:
            print(f"   检测到JSON不完整 ({{ {open_braces}/{close_braces}, [ {open_brackets}/{close_brackets}), 尝试修复...")
            
            # 更智能的修复：逐字符分析找到最后一个完整的值
            json_str = self._truncate_to_valid_point(json_str)
            
            # 重新统计括号
            open_braces = json_str.count('{')
            close_braces = json_str.count('}')
            open_brackets = json_str.count('[')
            close_brackets = json_str.count(']')
            
            # 按正确顺序添加缺失的括号（需要正确的嵌套顺序）
            bracket_stack = []
            for char in json_str:
                if char == '{':
                    bracket_stack.append('}')
                elif char == '[':
                    bracket_stack.append(']')
                elif char in '}]':
                    if bracket_stack and bracket_stack[-1] == char:
                        bracket_stack.pop()
            
            # 反向添加缺失的括号
            closing = ''.join(reversed(bracket_stack))
            json_str += closing
            print(f"   添加了缺失的括号: {closing}")
        
        return json_str
    
    def _truncate_to_valid_point(self, json_str: str) -> str:
        """将JSON截断到最后一个有效点"""
        import re
        
        # 尝试多种截断策略
        strategies = [
            # 策略1: 找到最后一个完整的对象或数组结尾
            r'(.*[\}\]])\s*,?\s*"[^"]*"\s*:\s*[^\}\]]*$',
            # 策略2: 找到最后一个完整的值（字符串、数字、布尔等）
            r'(.*(?:true|false|null|\d+|"[^"]*"))\s*,\s*"[^"]*"\s*:?\s*[^\}\]]*$',
            # 策略3: 找到最后一个闭合括号
            r'(.*[\}\]])[^\}\]]*$',
        ]
        
        for pattern in strategies:
            match = re.match(pattern, json_str, re.DOTALL)
            if match:
                result = match.group(1).rstrip(' ,\n\t')
                if result:
                    print(f"   使用策略成功截断JSON")
                    return result
        
        # 如果所有策略都失败，使用原始的逐字符方法
        last_valid_pos = len(json_str)
        
        # 从末尾向前找，跳过不完整的部分
        i = len(json_str) - 1
        while i >= 0:
            char = json_str[i]
            if char in '}]':
                last_valid_pos = i + 1
                break
            elif char == '"':
                # 找到字符串开始
                j = i - 1
                while j >= 0 and json_str[j] != '"':
                    if json_str[j] == '\\':
                        j -= 1  # 跳过转义字符
                    j -= 1
                if j >= 0:
                    # 检查这是否是一个完整的键值对
                    before = json_str[:j].rstrip()
                    if before.endswith(':') or before.endswith(',') or before.endswith('[') or before.endswith('{'):
                        last_valid_pos = i + 1
                        break
                i = j
            i -= 1
        
        result = json_str[:last_valid_pos].rstrip(' ,\n\t')
        
        # 确保不以逗号结尾
        result = re.sub(r',\s*$', '', result)
        
        return result
    
    def _ensure_required_fields(self, data: dict, request: TripRequest) -> dict:
        """确保必要字段存在"""
        from datetime import datetime, timedelta
        
        # 确保基本字段
        if 'city' not in data:
            data['city'] = request.city
        if 'start_date' not in data:
            data['start_date'] = request.start_date
        if 'end_date' not in data:
            data['end_date'] = request.end_date
        if 'overall_suggestions' not in data:
            data['overall_suggestions'] = f"祝您在{request.city}旅途愉快！"
        
        # 确保days数组存在且完整
        if 'days' not in data or not data['days']:
            data['days'] = []
        
        # 检查天数是否足够
        start = datetime.strptime(request.start_date, "%Y-%m-%d")
        while len(data['days']) < request.travel_days:
            idx = len(data['days'])
            current_date = start + timedelta(days=idx)
            data['days'].append({
                "date": current_date.strftime("%Y-%m-%d"),
                "day_index": idx,
                "description": f"第{idx + 1}天行程",
                "transportation": request.transportation,
                "accommodation": request.accommodation,
                "attractions": [],
                "meals": []
            })
        
        # 确保每天的必要字段
        for i, day in enumerate(data['days']):
            if 'date' not in day:
                day['date'] = (start + timedelta(days=i)).strftime("%Y-%m-%d")
            if 'day_index' not in day:
                day['day_index'] = i
            if 'description' not in day:
                day['description'] = f"第{i + 1}天行程"
            if 'transportation' not in day:
                day['transportation'] = request.transportation
            if 'accommodation' not in day:
                day['accommodation'] = request.accommodation
            if 'attractions' not in day:
                day['attractions'] = []
            if 'meals' not in day:
                day['meals'] = []
        
        # 确保weather_info存在
        if 'weather_info' not in data:
            data['weather_info'] = []
        
        return data
    
    def _create_fallback_plan(self, request: TripRequest) -> TripPlan:
        """创建备用计划(当Agent失败时)"""
        from datetime import datetime, timedelta
        
        # 解析日期
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
        
        # 创建每日行程
        days = []
        for i in range(request.travel_days):
            current_date = start_date + timedelta(days=i)
            
            day_plan = DayPlan(
                date=current_date.strftime("%Y-%m-%d"),
                day_index=i,
                description=f"第{i+1}天行程",
                transportation=request.transportation,
                accommodation=request.accommodation,
                attractions=[
                    Attraction(
                        name=f"{request.city}景点{j+1}",
                        address=f"{request.city}市",
                        location=Location(longitude=116.4 + i*0.01 + j*0.005, latitude=39.9 + i*0.01 + j*0.005),
                        visit_duration=120,
                        description=f"这是{request.city}的著名景点",
                        category="景点"
                    )
                    for j in range(2)
                ],
                meals=[
                    Meal(type="breakfast", name=f"第{i+1}天早餐", description="当地特色早餐"),
                    Meal(type="lunch", name=f"第{i+1}天午餐", description="午餐推荐"),
                    Meal(type="dinner", name=f"第{i+1}天晚餐", description="晚餐推荐")
                ]
            )
            days.append(day_plan)
        
        return TripPlan(
            city=request.city,
            start_date=request.start_date,
            end_date=request.end_date,
            days=days,
            weather_info=[],
            overall_suggestions=f"这是为您规划的{request.city}{request.travel_days}日游行程,建议提前查看各景点的开放时间。"
        )


# 全局多智能体系统实例
_multi_agent_planner = None


def get_trip_planner_agent() -> MultiAgentTripPlanner:
    """获取多智能体旅行规划系统实例(单例模式)"""
    global _multi_agent_planner

    if _multi_agent_planner is None:
        _multi_agent_planner = MultiAgentTripPlanner()

    return _multi_agent_planner

