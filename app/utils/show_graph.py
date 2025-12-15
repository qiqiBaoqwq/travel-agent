"""生成LangGraph工作流图"""
from IPython.display import Image

from app.core.agents.trip_planner_agent import get_trip_planner_agent

def show_graph(graph, xray=False):
    """Display a LangGraph mermaid diagram with fallback rendering.
    
    Handles timeout errors from mermaid.ink by falling back to pyppeteer.
    
    Args:
        graph: The LangGraph object that has a get_graph() method
        xray: Whether to show detailed internal structure
    
    Returns:
        IPython Image object or saves to file
    """
    try:
        # Try the default renderer first
        return Image(graph.get_graph(xray=xray).draw_mermaid_png())
    except Exception as e:
        print(f"⚠️  默认渲染器失败: {e}")
        # Fall back to pyppeteer if the default renderer fails
        import nest_asyncio
        nest_asyncio.apply()
        from langchain_core.runnables.graph import MermaidDrawMethod
        return Image(graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.PYPPETEER))


def generate_graph_image():
    """生成工作流图"""
    print("🔄 初始化多智能体系统...")
    planner = get_trip_planner_agent()
    
    # 获取图结构
    graph = planner.graph
    
    # 方法1: 生成 Mermaid 格式（文本）
    print("\n📊 Mermaid 流程图:")
    print("-" * 50)
    mermaid_code = graph.get_graph().draw_mermaid()
    print(mermaid_code)
    print("-" * 50)
    
    # 保存 Mermaid 代码到文件
    with open("workflow_graph.md", "w") as f:
        f.write("# LangGraph 工作流图\n\n")
        f.write("```mermaid\n")
        f.write(mermaid_code)
        f.write("\n```\n")
    print("✅ Mermaid 代码已保存到 workflow_graph.md")
    
    # 方法2: 生成 PNG 图片
    try:
        # 尝试使用默认渲染器
        png_data = graph.get_graph().draw_mermaid_png()
        with open("workflow_graph.png", "wb") as f:
            f.write(png_data)
        print("✅ PNG 图片已保存到 workflow_graph.png")
    except Exception as e:
        print(f"⚠️  默认渲染器失败: {e}")
        try:
            # 使用 pyppeteer 作为后备
            import nest_asyncio
            nest_asyncio.apply()
            from langchain_core.runnables.graph import MermaidDrawMethod
            png_data = graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.PYPPETEER)
            with open("workflow_graph.png", "wb") as f:
                f.write(png_data)
            print("✅ PNG 图片已保存到 workflow_graph.png (使用 pyppeteer)")
        except Exception as e2:
            print(f"⚠️  pyppeteer 渲染也失败: {e2}")
            print("   您可以将上面的 Mermaid 代码复制到 https://mermaid.live 在线查看")


if __name__ == "__main__":
    generate_graph_image()
