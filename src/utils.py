def show_graph(graph):
    # 生成并保存图表
    png_bytes = graph.get_graph().draw_mermaid_png()
    with open("langgraph_intro.png", "wb") as f:
        f.write(png_bytes)

    print("Graph image saved to langgraph_intro.png")

    # 自动用系统默认图片查看器打开
    import os
    os.startfile("langgraph_intro.png")  # Windows