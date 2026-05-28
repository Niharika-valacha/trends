from langchain_core.runnables import ConfigurableField, RunnableSerializable
from langchain_core.tools import tool
# from langchain_community.utilities import SerpAPIWrapper
from third_party_apps.adverse_media.service.google_search.google_search import GoogleSearchService
from typing import Optional, Dict, Union, List
from third_party_apps.adverse_media.service.adverse_service import AdverseService
from third_party_apps.ocs_agent.utils.search_utils import extract_link_from_question
from third_party_apps.ocs_agent.services.data_service.tenant_data_service import TenantDataService
from middleware import get_current_request
import logging

logger = logging.getLogger(__name__)

class WebLinkSummaryRunnable(RunnableSerializable):
    """
    Runnable for generating summaries from web links using an LLM.
    """
    
    llm: Optional[object] = None  # LLM will be passed at runtime

    def invoke(self, input, config=None):
        """Generate a summary from a web link in the query."""
        try:
            config = config or {}
            configurable = config.get("configurable", {})
            llm = configurable.get("llm", self.llm)
            prompt = configurable.get('summary_prompt')
            
            if not llm:
                raise ValueError("LLM not provided in config or initialization")

            query = input[0] if isinstance(input, list) else input
            get_link = extract_link_from_question(query)
            if not get_link:
                return "No valid weblink found in the query."
            
            data = {"link": get_link}
            # No need to initialize LLM here since it's passed via config
            get_summary = AdverseService.generate_summaries([data], llm, prompt)
            return get_summary
        except Exception as e:
            logger.exception(e)
            msg='Failed to generate summary!'
            return msg


# class SerpAPIRunnable(RunnableSerializable):
#     """
#     Runnable for performing web searches using SerpAPI.
#     """
    
#     serpapi_api_key: str  # Type-annotated field for the API key

#     def invoke(self, input, config=None):
#         """
#         Perform a web search and return structured results.
#         """
#         try:
#             query = input[0] if isinstance(input, list) else input
#             search = SerpAPIWrapper(serpapi_api_key=self.serpapi_api_key)
#             search_results = search.results(query)
#             sourced_information = []

#             for result_type in ['organic_results', 'news_results']:
#                 if result_type in search_results:
#                     for result in search_results[result_type]:
#                         snippet = result.get('snippet') or result.get('snippet_extended')
#                         link = result.get('link')
#                         if snippet and link:
#                             sourced_information.append({"text": snippet, "source_url": link})

#             if 'answer_box' in search_results and 'snippet' in search_results['answer_box'] and 'link' in search_results['answer_box']:
#                 sourced_information.append({
#                     "text": search_results['answer_box']['snippet'],
#                     "source_url": search_results['answer_box']['link']
#                 })

#             if 'related_questions' in search_results:
#                 for question_group in search_results['related_questions']:
#                     if 'snippet' in question_group and 'source' in question_group and 'link' in question_group['source']:
#                         sourced_information.append({
#                             "text": question_group['snippet'],
#                             "source_url": question_group['source']['link']
#                         })

#             return sourced_information
#         except Exception as e:
#             logger.exception(e)
#             msg='Failed to get data from websearch'
#             return msg

class GoogleCustomSearchRunnable(RunnableSerializable):
    """
    Runnable for performing web searches using Google Custom Search API.
    """
    google_api_key: str
    google_cse_id: str 
    google_search_url: str

    def invoke(self, input, config=None):
        """
        Perform a web search and return structured results.
        """
        try:
            query = input[0] if isinstance(input, list) else input
            grid_data, error_message = GoogleSearchService.get_google_search(
                name=query,
                address="",
                google_key=self.google_api_key,
                cx_id=self.google_cse_id,
                doc_limit=10,
                additional_keywords=None,
                google_search_url=self.google_search_url
            )

            if error_message:
                logger.error(f"Google search error: {error_message}")
                return "Failed to get data from websearch"
            
            # Transform the grid_data format to match expected format 
            sourced_information = []
            for item in grid_data:
                sourced_information.append({
                    "text": item.get("description"),
                    "source_url": item.get("link")
                })
            return sourced_information
        except Exception as e:
            logger.exception(e)
            msg = 'Failed to get data from websearch'
            return msg


class AgentWebTool:
    """
    A collection of web-related tools for search and summarization.
    """
    
    def generate_summary_from_weblink(query: str) -> str:
        """
        Generate a summary from a weblink. Use when the query contains a weblink.

        Args:
            query: The input query, expected to contain a weblink.
        Returns:
            A summary of the weblink content.
        """
        return (
            WebLinkSummaryRunnable()
            .configurable_fields(
                llm=ConfigurableField(
                    id="llm",
                    name="Language Model",
                    description="LLM instance for generating summaries, provided at runtime."
                )
            ).invoke(input=[query])
        )
    
    @tool
    def smart_web_search(query: str) -> Union[List[Dict[str, str]], str]:
        """
        Smart web search tool that chooses between weblink summary and Google Custom Search.

        Description:
            If the query contains a weblink, generates a summary using generate_summary_from_weblink.
            Otherwise, performs a web search using google_custom_search.

        Args:
            query: The input query to process.
        Returns:
            Either a list of search results or a summary string.
        """
        weblink = extract_link_from_question(query)
        if weblink:
            logger.info("Detected weblink, generating summary...")
            return AgentWebTool.generate_summary_from_weblink(query)
        logger.info("No weblink detected, performing web search...")
        return AgentWebTool.google_custom_search(query)



    def google_custom_search(query: str) -> list:
        """Searches the web for real-time, current information using Google Custom Search API."""
        request = get_current_request()
        db_setting = getattr(request, "db_setting", None)

        google_api_key = "default_key"
        google_cse_id = "default_cx"
        google_search_url = "default_url"

        if db_setting:
            tenant_cfg = TenantDataService.get_tenant_azure_info(db_setting)
            google_api_key = tenant_cfg.get("google_search_api_key") or google_api_key
            google_cse_id = tenant_cfg.get("google_cx_id") or google_cse_id
            google_search_url = tenant_cfg.get("google_search_url") or google_search_url

        return (
            GoogleCustomSearchRunnable(
                google_api_key=google_api_key,
                google_cse_id=google_cse_id,
                google_search_url=google_search_url
            )
            .configurable_fields(
                google_api_key=ConfigurableField(
                    id="google_api_key",
                    name="Google API Key",
                    description="API key for Google Custom Search, provided at runtime."
                ),
                google_cse_id=ConfigurableField(
                    id="google_cse_id",
                    name="Google CSE ID",
                    description="Custom Search Engine ID, provided at runtime."
                ),
                google_search_url=ConfigurableField(
                    id="google_search_url",
                    name="Google Search URL",
                    description="Google Search URL, provided at runtime."
                )
            ).invoke(input=[query])
        )
    
    # def serp_api_search(query: str) -> list:
    #     """Searches the web for real-time, current information using SerpAPI. Use for latest news, recent events, or up-to-date data."""
    #     return (
    #         SerpAPIRunnable(serpapi_api_key="default_key")
    #         .configurable_fields(
    #             serpapi_api_key=ConfigurableField(
    #                 id="serpapi_api_key",
    #                 name="SerpAPI Key",
    #                 description="API key for SerpAPI, provided at runtime."
    #             )
    #         ).invoke(input=[query])
    #     )
        


""" 
** File Name:        web_tools.py 
** Author:           arun 
** Creation Date:    2025-03-20
**
****************************************************************************** 
**                    COPYRIGHT                                             ** 
**                                                                          ** 
** (C) Copyright 2024                                                       ** 
** Cygnus Compliance Consulting, Inc.                                       ** 
**                                                                          ** 
** This software is furnished under a license for use only on a single      ** 
** computer system and may be copied only with the inclusion of the above   ** 
** copyright notice. This software or any other copies thereof, may not be  ** 
** provided or otherwise made available to any other person except for use  ** 
** on such system and to one who agrees to these license terms. Title and   ** 
** ownership of the software shall at all times remain in                   ** 
** Cygnus Compliance Consulting, Inc.                                       ** 
**                                                                          ** 
** The information in this software is subject to change without notice and ** 
** should not be construed as a commitment by                               ** 
** Cygnus Compliance Consulting, Inc.                                       ** 
****************************************************************************** 
            Maintenance History

-------------|----------|---------------------------------------------------- 
    Date     |  Person  |  Description of Modification 
-------------|----------|---------------------------------------------------- 
2025-03-20   |  arun  |  added agent web tool 
-------------|----------|---------------------------------------------------- 
"""
