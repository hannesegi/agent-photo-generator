import os
import sys
import json
from configparser import ConfigParser
import time
from typing import Dict, Any, List, Literal
import hashlib
import traceback

from openai import OpenAI, AsyncOpenAI
from openai.types import CompletionUsage
from loguru import logger

path_this = os.path.dirname(os.path.abspath(__file__))
path_project = os.path.dirname(os.path.join(path_this, '..'))
path_root = os.path.dirname(path_this)
sys.path.extend([path_root, path_project, path_this])

class BaseAgent:
    def __init__(
        self,
        system_prompt: str,
        human_prompt: str,
        provider:  "openai",
        agent_name: str = None,
        multiagent_name: str = None,
        stage: str = "test",
        max_retries: int = 3,
        **model_kwargs: Any,
    ):
        self.system_prompt = system_prompt
        self.human_prompt = human_prompt
        self.provider = provider
        self.agent_name = agent_name
        self.multiagent_name = multiagent_name
        self.stage = stage
        self.max_retries = max_retries
        
        self.timeout = model_kwargs.get("timeout", 180)
        self.model_name = model_kwargs.get("model_name")
        self.base_url = model_kwargs.get("base_url", "https://api.openai.com/v1")
        self.api_key = model_kwargs.get("api_key")
        
        self._validate_model_kwargs(model_kwargs)

        self._init_config()
        self.raw_system_prompt = system_prompt

    def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):
        model_kwargs.pop("model_name", None)
        model_kwargs.pop("base_url", None)
        model_kwargs.pop("api_key", None)
        
        if "max_new_token" in model_kwargs:
            model_kwargs.pop("max_new_token")
        if not "timeout" in model_kwargs:
            model_kwargs["timeout"] = 180
            
        self.model_kwargs = model_kwargs

    def _init_config(self):
        self.config = ConfigParser(allow_no_value=True)
        self.config.read(os.path.join(path_root, "config.ini"))

    def chat_prompt(self, **kwargs):
        """
        Initializes the chat prompt by combining the system and human prompts.
        """
        system_content = self.system_prompt.format(**kwargs) if kwargs else self.system_prompt
        user_content = self.human_prompt.format(**kwargs) if kwargs else self.human_prompt
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        
    def _llm(self):
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
    
    def _allm(self):
        return AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )
        
    def analyze(self, **kwargs: Any) -> str:
        start_time = time.time()
        try:
            tries = 0
            while tries < self.max_retries:
                with self._llm() as llm:
                    response=llm.chat.completions.create(
                        model=self.model_name,
                        messages=self.chat_prompt(**kwargs),
                        **self.model_kwargs
                    )
                if response.choices[0].finish_reason == "stop":
                    self._log_success(response.choices[0].message, start_time, response.usage, **kwargs)
                    return response.choices[0].message.content
                tries += 1
            
            raise Exception(f"Max retries exceeded on query input {self.human_prompt}")
        except Exception as e:
            self._handle_error(e, start_time, **kwargs)

    async def aanalyze(self, **kwargs: Any) -> str:
        start_time = time.time()
        try:
            tries = 0
            while tries < self.max_retries:
                async with self._allm() as llm:
                    response = await llm.chat.completions.create(
                        model=self.model_name,
                        messages=self.chat_prompt(**kwargs),
                        **self.model_kwargs
                    )
                    if response.choices[0].finish_reason == "stop":
                        self._log_success(response.choices[0].message, start_time, response.usage, **kwargs)
                        return response.choices[0].message.content
                    tries += 1
                    logger.warning(f"Retry {tries + 1} / {self.max_retries}")
                    
            raise Exception(f"Max retries exceeded on query input {self.chat_prompt(**kwargs)}")
        except Exception as e:
            self._handle_error(e, start_time, **kwargs)
            
    def _get_system_prompt(self, **kwargs) -> str:
        try:
            return self.system_prompt.format(**kwargs) if kwargs else self.system_prompt
        except Exception as e:
            logger.error(f"Error in _get_system_prompt: {str(e)}")
            return "Error occurred while extracting system prompt"

    def _get_human_prompt(self, **kwargs) -> str:
        try:
            return self.human_prompt.format(**kwargs) if kwargs else self.human_prompt
        except Exception as e:
            logger.error(f"Error in _get_human_prompt: {str(e)}")
            return "Error occurred while extracting human prompt"

    def _log_success(self, result: List[Any], start_time: float, usage: CompletionUsage, **kwargs):
        try:
            log_data = self._prepare_log_data(result, start_time, usage, **kwargs)
            logger.info("Successfully processed request", **log_data)
        except Exception as e:
            logger.warning(f"Failed to log success: {traceback.format_exc()}")

    def _handle_error(self, error: Exception, start_time: float, **kwargs):
        error_message = str(error)
        logger.error(f"Error: {error_message}")
        logger.error(f"Failed to generate result: {traceback.format_exc()}")
        try:
            log_data = self._prepare_log_data(error_message, start_time, is_error=True, **kwargs)
            logger.error("Error occurred during processing", **log_data)
        except Exception as e:
            logger.warning(f"Failed to log error: {traceback.format_exc()}")
        raise error

    def get_token_usage_from_metadata(self, response_metadata: Dict) -> Dict:
        """Extract token usage information from LLM response metadata."""
        if isinstance(response_metadata, CompletionUsage):
            return {
                "input_tokens":    response_metadata.prompt_tokens,
                "output_tokens":   response_metadata.completion_tokens,
                "total_tokens":    response_metadata.total_tokens,
                "model_name":      self.model_name,
            }
        token_info = response_metadata.get('token_usage', {})
        logger.debug(token_info)

        return {
            "input_tokens":  token_info.get('prompt_tokens', 0),
            "output_tokens": token_info.get('completion_tokens', 0),
            "total_tokens":  token_info.get('total_tokens', 0),
            "model_name":    self.model_name,
        }
    
    def _prepare_log_data(
        self, 
        result, 
        start_time: float,
        usage: CompletionUsage = None,
        **kwargs
    ) -> Dict[str, Any]:
        process_time = time.time() - start_time
        log_data = {
            "process_time": process_time,
            "timestamp": time.time(),
        }
        
        if isinstance(result, str):  # Error case
            log_data["error_message"] = result
        else:
            log_data["result"] = result.content if hasattr(result, 'content') else str(result)
            
        if usage is not None:
            log_data["token_usage"] = self.get_token_usage_from_metadata(usage)
        else:
            log_data["token_usage"] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "model_name": self.model_name,
            }
            
        log_data["id"] = hashlib.md5(f"{log_data.get('result', '')}{log_data['timestamp']}".encode()).hexdigest()
        return log_data
    

if __name__ == "__main__":
    
    system_prompt = """
    You are a skilled social media expert specializing in crafting engaging and potentially viral Instagram comments. Your task is to generate comments that meet the following criteria:

    Relevant to the caption content – The comment must directly relate to the subject or theme of the original post.

    Attention-grabbing – Use humor, curiosity, or relatable expressions to catch users’ attention.

    Encourage engagement – The comment should invite replies, likes, or further conversation.

    Appeal to general audiences – Use language and tone that most Instagram users can connect with.

    Natural and non-spammy – Avoid overly promotional, robotic, or generic comments.

    Output must be in Bahasa Indonesia. and must be shortly comment
    """
    human_prompt = "Buatkan komentar Instagram untuk caption tentang:{topic}."
    
    agent = BaseAgent(
        system_prompt=system_prompt,
        human_prompt=human_prompt,
        provider="openai",
        agent_name="JokeTeller",
        multiagent_name="ComedyTeam",
        stage="test",
        model_name="llama",
        api_key="api_key",  
        temperature=0.2,
        max_retries=3,
        base_url="http://192.168.18.162:8004/v1"
    )

    try:
        topics="""
        Issue Kebakaran Rumah
        """
        result = agent.analyze(topic=topics)
        print("Result:", result)
    except Exception as e:
        print("An error occurred:", str(e))