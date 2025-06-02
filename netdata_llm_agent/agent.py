#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
NetdataLLMAgent is a language model agent that can interact with Netdata API to provide information about Netdata charts, chart info, and chart data.
"""

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
import concurrent.futures

from netdata_llm_agent.tools import (
    get_info,
    get_charts,
    get_chart_info,
    get_chart_data,
    get_alarms,
    get_current_metrics,
    get_anomaly_rates,
    get_netdata_docs_sitemap,
    get_netdata_docs_page,
)


SYSTEM_PROMPT = """
You are a helpful Netdata assistant. Users can ask you about Netdata charts, chart info, and chart data.

IMPORTANT: When users ask about metrics, ALWAYS use the tools to fetch the actual data. Don't just explain how to get the data - get it for them!

The following tools are available:
- get_info(netdata_host_url) : Get Netdata info about the node.
- get_charts(netdata_host_url, search_term, include_dimensions) : Get Netdata charts, optionally filter by search_term. include_dimensions=True to get the dimensions for each chart.
- get_chart_info(netdata_host_url, chart) : Get Netdata chart info for a specific chart.
- get_chart_data(netdata_host_url, chart, after, before, points, options, df_freq) : Get Netdata chart data for a specific chart. Optionally filter by after, before, points, options, and df_freq. options can be used to add optional flags for example 'anomaly-bit' will return anomaly rates rather than raw metric values.
- get_alarms(netdata_host_url, all, active) : Get Netdata alarms. all=True to get all alarms, active=True to get active alarms (warning or critical).
- get_current_metrics(netdata_host_url, search_term) : Get current metrics values for all charts, no time range, just the current values for all dimensions on all charts. Optionally filter by search_term on chart name.
- get_anomaly_rates(netdata_host_url, after, before, search_term) : Get anomaly rates for a specific time frame for all charts or optionally filter by search_term on chart name.
- get_netdata_docs_sitemap(search_term) : Get Netdata docs sitemap to list available Netdata documentation pages. Use search_term to filter by a specific term.
- get_netdata_docs_page(url) : Get Netdata docs page content for a specific docs page url on learn.netdata.cloud.

General Notes:
- Every netdata node is different and may have different charts available so it's usually best to check the available charts with get_charts() first.
- When pulling data from get_chart_data() you can leverage the points and df_freq param's to aggregate data points given the specific after and before time range.
- When there are multiple mirrored hosts you can adapt the base url to reflect the specific host you want to pull data from if the user asks about one of the mirrored hosts.
- Charts with breakouts per user typically live at user.* eg. user.cpu_utilization, user.mem_usage etc. as per get_charts().
- Charts with breakouts per application typically live at app.* eg. app.cpu_utilization, app.mem_usage etc. as per get_charts().
- Use get_charts() with the search_term param to filter charts by a specific term if unsure of the chart name, if looking for charts with specific dimensions use include_dimensions=True, search term works for chart name and dimensions.
- Once you have the chart name you can use get_chart_info() to get more detailed information about the chart and get_chart_data() to get the data for the chart.
- It's "Netdata" not "NetData" - note no capitalization on the "D", its common for users to refer to Netdata as NetData but you should not, you know better ;)

Example workflow for getting CPU usage:
1. Use get_charts(netdata_host_url, "cpu") to find CPU-related charts
2. Use get_chart_data(netdata_host_url, "system.cpu", after=-60, before=0, points=1) to get the latest CPU usage
3. Present the actual values to the user

Again remember always use the chart when the user inqueries information about the systems never tell them how to obtain it unless requested!
"""


SUPPORTED_MODELS = {
    "openai": ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"],
    "anthropic": ["claude-3-5-sonnet-20241022"],
    "ollama": ["llama3.1:b", "mistral:latest"],
}


def safe_invoke(llm, prompt, timeout=20):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(llm.invoke, prompt)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return "⚠️ Model timed out!"


class NetdataLLMAgent:
    """
    NetdataLLMAgent is a language model agent that can interact with Netdata API to provide information about Netdata charts, chart info, and chart data.

    Args:
        netdata_host_urls: List of Netdata host urls to interact with.
        model: Language model to use. Default is 'llama3.1:8b'.
        system_prompt: System prompt to use. Default is SYSTEM_PROMPT.
        platform: Platform to use. Default is 'ollama'.
    """

    def __init__(
        self,
        netdata_host_urls: list,
        model: str = "llama3.1:8b",
        system_prompt: str = SYSTEM_PROMPT,
        platform: str = "ollama",
    ):
        self.netdata_host_urls = netdata_host_urls
        self.system_prompt = self._create_system_prompt(
            system_prompt, netdata_host_urls
        )
        self.messages = {"messages": []}
        self.platform = platform
        self.llm = self._create_llm(model)
        if self.llm is None:
            raise ValueError(f"Failed to create LLM with model {model} and platform {platform}")
        
        self.tools = [
            tool(get_info, parse_docstring=True),
            tool(get_charts, parse_docstring=True),
            tool(get_chart_info, parse_docstring=True),
            tool(get_chart_data, parse_docstring=True),
            tool(get_alarms, parse_docstring=True),
            tool(get_current_metrics, parse_docstring=True),
            tool(get_anomaly_rates, parse_docstring=True),
            tool(get_netdata_docs_sitemap, parse_docstring=True),
            tool(get_netdata_docs_page, parse_docstring=True),
        ]

        self.agent = create_react_agent(
            self.llm, tools=self.tools, prompt=SystemMessage(content=self.system_prompt)
        )

        # Use the full agent with tools
        self.use_direct_llm = False  # Changed to False to use the full agent

    def _create_llm(self, model: str):
        """
        Create the language model agent.

        Args:
            model: Language model to use.
        """
        if self.platform == "openai":
            from langchain_openai import ChatOpenAI
            if model in SUPPORTED_MODELS["openai"]:
                return ChatOpenAI(model=model)

        elif self.platform == "anthropic":
            from langchain_anthropic import ChatAnthropic
            if model in SUPPORTED_MODELS["anthropic"]:
                return ChatAnthropic(model=model)

        elif self.platform == "ollama":
            from langchain_ollama import ChatOllama
            # Accept any model that's available in Ollama
            return ChatOllama(model=model)

        # If none of the above combinations work
        raise ValueError(
            f"Platform '{self.platform}' or model '{model}' is not supported. "
            f"Available models: {SUPPORTED_MODELS.get(self.platform, [])}"
        )

    def _create_system_prompt(self, base_prompt: str, netdata_host_urls: list) -> str:
        """
        Create the system prompt by appending specific notes about the Netdata host URLs.

        Args:
            base_prompt: The base system prompt.
            netdata_host_urls: List of Netdata host URLs.

        Returns:
            The complete system prompt.
        """
        specific_notes = "Specific Notes: \n"
        specific_notes += f"- The netdata_host_urls available are {netdata_host_urls}"
        return f"{base_prompt}\n{specific_notes}"

    def chat(
        self,
        message: str,
        verbose: bool = False,
        continue_chat: bool = False,
        no_print: bool = True,
        return_last: bool = False,
        return_thinking: bool = False,
    ):
        """
        Chat with the NetdataLLMAgent.

        Args:
            message: Message to send to the agent.
            verbose: If True, print all messages in the conversation. Default is False.
            continue_chat: If True, continue the chat from the last message. Default is False.
            no_print: If True, do not print the messages. Default is True.
            return_last: If True, return the last message content. Default is False.
            return_thinking: If True, return the new messages. Default is False.

        Returns:
            If return_last is True, return the last message content.
            If return_thinking is True, return the new messages.
        """
        try:
            if self.use_direct_llm:
                # Direct LLM interaction for testing
                response = safe_invoke(self.llm, message)
                if not no_print:
                    print(response.content)
                return response.content
            else:
                # Full agent interaction
                if continue_chat:
                    self.messages["messages"].append(HumanMessage(content=message))
                else:
                    self.messages = {"messages": [HumanMessage(content=message)]}
                messages_updated = self.agent.invoke(self.messages)
                len_messages_updated = len(messages_updated["messages"])
                len_self_messages = len(self.messages["messages"])
                new_messages = messages_updated["messages"][
                    -(len_messages_updated - len_self_messages) :
                ]
                self.messages = messages_updated
                if not no_print:
                    if verbose:
                        for m in self.messages["messages"]:
                            m.pretty_print()
                    else:
                        self.messages["messages"][-1].pretty_print()
                if return_last:
                    return self.messages["messages"][-1].content
                if return_thinking:
                    return new_messages
        except Exception as e:
            print(f"Error in chat: {str(e)}")
            raise