"""
MCPCommander — MCP Agent 实验模式

LLM 自主决定调哪个 MCP 工具、传什么参数，不再硬编码 tool_name。

流程：
1. 连接 MCP Server → list_tools → 拿到工具描述
2. 把工具描述拼进 system prompt，发给 LLM
3. LLM 返回 tool_calls → 通过 MCP 协议执行
4. 工具结果再给 LLM → 生成最终回答
"""

import time
import json
import asyncio
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# 最大 ReAct 循环轮次（防止死循环）
MAX_ITERATIONS = 5


class MCPCommander:

    def __init__(
        self,
        llm,
        mcp_server_command: str = "python",
        mcp_server_args: Optional[List[str]] = None,
        mcp_server_env: Optional[Dict[str, str]] = None,
    ):
        self.llm = llm
        self.mcp_server_params = StdioServerParameters(
            command=mcp_server_command,
            args=mcp_server_args or ["mcp_server.py"],
            env=mcp_server_env,
        )
        self.last_mcp_result = None

    # ================================================================
    #  MCP 工具发现 & 执行
    # ================================================================

    async def _discover_mcp_tools(self, session: ClientSession) -> List[Dict[str, Any]]:
        """从 MCP Server 获取工具列表，转为 OpenAI function-calling 格式"""
        tools_result = await session.list_tools()
        openai_tools = []
        for t in tools_result.tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.inputSchema,
                }
            })
        return openai_tools

    async def _call_mcp_tool(self, session: ClientSession, tool_name: str, arguments: Dict[str, Any]) -> str:
        """通过 MCP 协议调用工具，返回文本结果"""
        result = await session.call_tool(tool_name, arguments=arguments)

        if result.isError:
            raise RuntimeError(f"MCP 工具调用失败: {result.content}")

        text_parts = []
        for block in result.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)
        return "\n".join(text_parts)

    # ================================================================
    #  LLM 交互（OpenAI 兼容接口）
    # ================================================================

    def _build_system_prompt(self, tools: List[Dict[str, Any]]) -> str:
        """构建包含工具描述的 system prompt"""
        tools_desc = "\n".join(
            f"- **{t['function']['name']}**: {t['function']['description']}"
            for t in tools
        )
        return f"""你是一个专业的劳动法知识助手。你可以使用以下工具来帮助回答用户问题：

{tools_desc}

使用规则：
1. 如果用户的问题与劳动法相关，选择合适的工具获取信息后再回答
2. 如果用户只是打招呼、感谢等闲聊，直接回答即可，不需要调用工具
3. 如果需要查看原文档片段进行分析，使用 search_knowledge_base
4. 如果只需要最终答案，使用 ask_knowledge_base
5. 根据工具描述和参数要求，自行决定调用哪个工具和传入什么参数
6. 基于工具返回的结果来组织你的回答，不要编造信息"""

    async def _llm_chat(self, messages: List[Dict], tools: Optional[List[Dict]] = None) -> Dict:
        """调用 LLM（OpenAI 兼容接口），支持 tool_use"""
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

        # 转换为 LangChain 消息格式
        lc_messages = []
        for m in messages:
            role = m.get("role")
            content = m.get("content", "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                tool_calls = m.get("tool_calls")
                if tool_calls:
                    # 带工具调用的 AI 消息
                    lc_tool_calls = []
                    for tc in tool_calls:
                        lc_tool_calls.append({
                            "name": tc["function"]["name"],
                            "args": json.loads(tc["function"]["arguments"]) if isinstance(tc["function"]["arguments"], str) else tc["function"]["arguments"],
                            "id": tc["id"],
                            "type": "tool_call",
                        })
                    lc_messages.append(AIMessage(content=content or "", tool_calls=lc_tool_calls))
                else:
                    lc_messages.append(AIMessage(content=content))
            elif role == "tool":
                lc_messages.append(ToolMessage(content=content, tool_call_id=m.get("tool_call_id", "")))

        # 绑定工具并调用
        if tools:
            lc_tools = []
            for t in tools:
                lc_tools.append({
                    "type": "function",
                    "function": t["function"],
                })
            llm_with_tools = self.llm.bind_tools(lc_tools)
            response = await llm_with_tools.ainvoke(lc_messages)
        else:
            response = await self.llm.ainvoke(lc_messages)

        # 解析响应
        result = {"content": response.content or ""}

        if hasattr(response, "tool_calls") and response.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["args"] if isinstance(tc["args"], str) else json.dumps(tc["args"], ensure_ascii=False),
                    }
                }
                for tc in response.tool_calls
            ]

        return result

    # ================================================================
    #  核心：Agent ReAct 循环
    # ================================================================

    async def run_via_mcp(self, question: str) -> Dict[str, Any]:
        """
        真正的 Agent 模式：LLM 自主决定调哪个工具

        ReAct 循环：
          user question → LLM(含工具描述) → tool_calls?
            → 有：执行工具 → 结果追加到 messages → 再问 LLM
            → 无：LLM 给出最终回答 → 结束
        """
        start_time = time.time()
        tool_calls_log = []

        async with stdio_client(self.mcp_server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                # 1. 发现工具
                tools = await self._discover_mcp_tools(session)
                print(f"[Agent] 可用工具: {[t['function']['name'] for t in tools]}")

                # 2. 构建 messages
                system_prompt = self._build_system_prompt(tools)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ]

                # 3. ReAct 循环
                for iteration in range(MAX_ITERATIONS):
                    print(f"[Agent] 迭代 {iteration + 1}/{MAX_ITERATIONS}")

                    llm_response = await self._llm_chat(messages, tools=tools)

                    # 没有工具调用 → LLM 给出最终回答
                    if "tool_calls" not in llm_response:
                        answer = llm_response["content"]
                        print(f"[Agent] 最终回答（迭代 {iteration + 1}）")
                        break

                    # 有工具调用 → 依次执行
                    messages.append({
                        "role": "assistant",
                        "content": llm_response.get("content") or "",
                        "tool_calls": llm_response["tool_calls"],
                    })

                    for tc in llm_response["tool_calls"]:
                        tool_name = tc["function"]["name"]
                        tool_args = json.loads(tc["function"]["arguments"]) if isinstance(tc["function"]["arguments"], str) else tc["function"]["arguments"]
                        tool_call_id = tc["id"]

                        print(f"[Agent] 调用工具: {tool_name}({tool_args})")

                        try:
                            tool_result = await self._call_mcp_tool(session, tool_name, tool_args)
                            tool_calls_log.append({"tool": tool_name, "args": tool_args, "status": "success"})
                        except Exception as e:
                            tool_result = f"工具调用失败: {str(e)}"
                            tool_calls_log.append({"tool": tool_name, "args": tool_args, "status": "error", "error": str(e)})

                        messages.append({
                            "role": "tool",
                            "content": tool_result,
                            "tool_call_id": tool_call_id,
                        })

                else:
                    # 超过最大迭代次数，强行让 LLM 做最终回答（不带工具）
                    final_response = await self._llm_chat(messages, tools=None)
                    answer = final_response["content"]
                    print(f"[Agent] 达到最大迭代次数，强制结束")

                elapsed_time = time.time() - start_time

        # 尝试从工具结果中提取结构化信息
        parsed = self._extract_structured_info(tool_calls_log)

        return {
            "answer": answer,
            "tool_calls": tool_calls_log,
            "retrieved": len(tool_calls_log) > 0,
            "elapsed_time": round(elapsed_time, 3),
            "confidence": parsed.get("confidence", 0.0),
            "attempt_count": parsed.get("attempt_count", 0),
            "query_history": parsed.get("query_history", []),
            "messages": [],
            "mode": "mcp_agent",
            "iterations": iteration + 1 if 'iteration' in dir() else MAX_ITERATIONS,
        }

    def _extract_structured_info(self, tool_calls_log: List[Dict]) -> Dict:
        """从工具调用日志中尝试提取结构化信息"""
        result = {"confidence": 0.0, "attempt_count": 0, "query_history": []}

        for tc in tool_calls_log:
            if tc.get("status") != "success":
                continue
            try:
                parsed = json.loads(tc.get("result", "{}"))
                if "confidence" in parsed:
                    result["confidence"] = parsed["confidence"]
                if "attempt_count" in parsed:
                    result["attempt_count"] = parsed["attempt_count"]
                if "query_history" in parsed:
                    result["query_history"] = parsed["query_history"]
            except (json.JSONDecodeError, TypeError):
                pass

        return result

    async def search_only(self, query: str, top_k: int = 8) -> Dict[str, Any]:
        """只检索，不生成答案（直接调 search_knowledge_base，不走 Agent 循环）"""
        async with stdio_client(self.mcp_server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                raw_result = await self._call_mcp_tool(
                    session, "search_knowledge_base", {"query": query, "top_k": top_k}
                )
                try:
                    return json.loads(raw_result)
                except json.JSONDecodeError:
                    return {"error": "结果解析失败", "raw": raw_result}