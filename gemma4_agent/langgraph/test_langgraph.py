from langgraph.graph import StateGraph, MessagesState, START, END
from client_type import ollama_client
llm = ollama_client

workflow = StateGraph(MessagesState, llm=llm, name="Hello LangGraph")

def node1(str):
    return {"messages": [{"role": "ai", "content": "I reached Node1"}]}

def node2(str):
    return {"messages": [{"role": "ai", "content": "And Now at Node 2"}]}

# Add the nodes
workflow.add_node("node_1", node1)
workflow.add_node("node_2", node2)

workflow.add_edge(START, "node_1")
workflow.add_edge("node_1","node_2")
workflow.add_edge("node_2", END)

#Run the workflow
app = workflow.compile()
response =app.invoke({"messages": [{"role": "user", "content": "hi!"}]})
print(response)
# print(app.get_graph().draw_ascii())
print(app.get_graph().draw_mermaid())