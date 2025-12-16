"""基于LangChain和LangGraph的多智能体旅行规划系统"""

import os
import json
import threading
from typing import Dict, Any, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END, START

from app.core.tools.travel_details_tools import (search_weather,
                                                 search_attractions,
                                                 search_hotels)

from app.core.prompts import SCHEDULER_AGENT_PROMPT, SUMMARIZER_AGENT_PROMPT
from app.schemas.travel_plan_related_schemas import (TripRequest, TripPlan,
                                                     DayPlan, Attraction,
                                                     Meal,
                                                     Location,
                                                     TripPlannerState)
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class MultiAgentTripPlanner:
    """基于LangGraph的多智能体旅行规划系统 - Scheduler协调模式
    
    架构:
    1. Scheduler Agent 分析任务并制定计划
    2. 并行分发任务给 Weather/Hotel/Attraction Agents
    3. 收集所有结果后，Scheduler Agent 进行总结
    """

    def __init__(self):
        """初始化多智能体系统"""
        logger.info("🔄 开始初始化LangGraph多智能体旅行规划系统(Scheduler模式)...")

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
            
            logger.info("✅ LLM服务初始化成功")
            logger.info(f"   模型: {model_id}")
            logger.info(f"   Base URL: {base_url}")
            logger.info(f"   Max Tokens: {8192}")
            
            # 定义工具
            self.tools = [search_attractions, search_weather, search_hotels]
            self.tools_map = {t.name: t for t in self.tools}
            
            # 创建带工具的LLM
            self.llm_with_tools = self.llm.bind_tools(self.tools)
            
            # 创建工作流
            self._build_graph()
            
            logger.info("✅ LangGraph多智能体系统初始化成功(Scheduler模式)")
            logger.info(f"   可用工具: {[t.name for t in self.tools]}")
            logger.info("   架构: Scheduler -> [Weather, Hotel, Attraction] -> Summarizer")

        except Exception as e:
            logger.error(f"❌ 多智能体系统初始化失败: {str(e)}", exc_info=True)
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
        logger.info("\n📋 Scheduler Agent: 分析任务并制定计划...")
        
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
        
        logger.info(f"   任务计划: {response.content[:200]}...")
        
        return {
            "messages": messages + [response],
            "task_plan": response.content,
            "current_step": "planning",
            "completed_tasks": [],
            "agent_results": {}
        }

    def _attraction_agent_node(self, state: TripPlannerState) -> Dict[str, Any]:
        """景点搜索Agent - 直接调用工具获取数据"""
        logger.info("📍 Attraction Agent: 搜索景点...")
        
        request = state["request"]
        preferences = request.get("preferences", [])
        keywords = preferences[0] if preferences else "景点"
        city = request["city"]
        
        # 直接调用工具
        result = search_attractions.invoke({"keywords": keywords, "city": city})
        
        logger.info(f"   景点搜索完成: {result[:100]}...")
        
        return {
            "agent_results": {"attractions": result},
            "completed_tasks": ["attraction"]
        }

    def _weather_agent_node(self, state: TripPlannerState) -> Dict[str, Any]:
        """天气查询Agent - 直接调用工具获取数据"""
        logger.info("🌤️  Weather Agent: 查询天气...")
        
        request = state["request"]
        city = request["city"]
        
        # 直接调用工具
        result = search_weather.invoke({"city": city})
        
        logger.info(f"   天气查询完成: {result[:100]}...")
        
        return {
            "agent_results": {"weather": result},
            "completed_tasks": ["weather"]
        }

    def _hotel_agent_node(self, state: TripPlannerState) -> Dict[str, Any]:
        """酒店推荐Agent - 直接调用工具获取数据"""
        logger.info("🏨 Hotel Agent: 搜索酒店...")
        
        request = state["request"]
        city = request["city"]
        accommodation = request.get("accommodation", "酒店")
        
        # 直接调用工具
        result = search_hotels.invoke({"city": city, "hotel_type": accommodation})
        
        logger.info(f"   酒店搜索完成: {result[:100]}...")
        
        return {
            "agent_results": {"hotels": result},
            "completed_tasks": ["hotel"]
        }

    def _collector_node(self, state: TripPlannerState) -> Dict[str, Any]:
        """收集器节点 - 等待所有Agent完成"""
        completed = state.get("completed_tasks", [])
        logger.info(f"📦 Collector: 已完成任务 {completed}")
        
        # 不做修改，只是一个同步点
        return {}

    def _check_all_tasks_completed(self, state: TripPlannerState) -> Literal["wait", "summarize"]:
        """检查是否所有任务都已完成"""
        completed = state.get("completed_tasks", [])
        required_tasks = {"attraction", "weather", "hotel"}
        
        if required_tasks.issubset(set(completed)):
            logger.info("✅ 所有任务已完成，准备总结...")
            return "summarize"
        else:
            remaining = required_tasks - set(completed)
            logger.info(f"⏳ 等待任务完成: {remaining}")
            return "wait"

    def _scheduler_summarize_node(self, state: TripPlannerState) -> Dict[str, Any]:
        """Scheduler Agent - 总结阶段: 整合所有信息生成最终计划"""
        logger.info("\n📋 Scheduler Agent: 整合信息并生成最终计划...")
        
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
        
        logger.info("   最终计划生成完成")
        
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
            logger.info(f"\n{'='*60}")
            logger.info("🚀 开始Scheduler模式多智能体协作规划旅行...")
            logger.info(f"目的地: {request.city}")
            logger.info(f"日期: {request.start_date} 至 {request.end_date}")
            logger.info(f"天数: {request.travel_days}天")
            logger.info(f"偏好: {', '.join(request.preferences) if request.preferences else '无'}")
            logger.info(f"{'='*60}\n")
            
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
            logger.info(f"\n行程规划结果: {final_plan[:300]}...\n")
            
            trip_plan = self._parse_response(final_plan, request)

            logger.info(f"{'='*60}")
            logger.info("✅ 旅行计划生成完成!")
            logger.info(f"{'='*60}\n")

            return trip_plan

        except Exception as e:
            logger.error(f"❌ 生成旅行计划失败: {str(e)}", exc_info=True)
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
            logger.warning(f"⚠️  解析响应失败: {str(e)}")
            logger.info("   将使用备用方案生成计划")
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
            logger.info(f"   检测到JSON不完整 ({{ {open_braces}/{close_braces}, [ {open_brackets}/{close_brackets}), 尝试修复...")
            
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
            logger.info(f"   添加了缺失的括号: {closing}")
        
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
                    logger.info("   使用策略成功截断JSON")
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
# =========== 饿汉式线程安全级别的单例模式（加载模块是安全加载，非并发环境） ===========
# _multi_agent_planner = MultiAgentTripPlanner()
#
# def get_trip_planner_agent() -> MultiAgentTripPlanner:
#     """获取多智能体旅行规划系统实例(单例模式)"""
#     return _multi_agent_planner


# 全局多智能体系统实例--实现方式-> 懒汉式线程不安全 （若要安全需双检➕锁）
_multi_agent_planner = None

# 全局可重入锁
_lock = threading.RLock()


def get_trip_planner_agent() -> MultiAgentTripPlanner:
    """获取多智能体旅行规划系统实例(单例模式)"""
    global _multi_agent_planner

    if _multi_agent_planner is None:
        with _lock:
            if _multi_agent_planner is None:
                _multi_agent_planner = MultiAgentTripPlanner()

    return _multi_agent_planner