import os, sys
from loguru import logger

path_this = os.path.dirname(os.path.abspath(__file__))
path_project = os.path.dirname(os.path.join(path_this, '..'))
path_root = os.path.dirname(os.path.join(path_this, '../..'))
sys.path.extend([path_this, path_project, path_root])

from agents.base_agent import BaseAgent

class PromptGenAgent(BaseAgent):
    def __init__(
        self,
        agent_name: str = "CommentContentAgent",
        temperature: float = 0.1,
        max_retries: int = 3,
        timeout: int = 150,
        model_name: str = "",
        base_url: str = "",
        api_key: str = "test",
        stage: str = "prod",
        system_prompt: str = None,
        human_prompt: str = None,
        provider = "ai-hanes",
        **kwargs,
    ):
        super().__init__(
            agent_name=agent_name,
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            temperature=temperature,
            max_retries=max_retries,
            timeout=timeout,
            base_url=base_url,
            model_name=model_name,
            api_key=api_key,
            stage=stage,
            provider=provider,
            **kwargs
        )
        
    def analyze(self, data_input: str = None):
        try:
            result = super().analyze(data_input=data_input)
            logger.success(f"SUCCESS Generate Comment Agent")
            
            if result == data_input:
                logger.error("Agent returned input data instead of generated content")
                return None
                
            return result
        except Exception as e:
            logger.error(f"Error in CommentContentAgent: {str(e)}")
            logger.exception("traceback")
            return None

if __name__ == "__main__":
    system_prompt = """

    """
    human_prompt = """
    Analisis konten berikut:
    {data_input}
    Ringkas naratifnya dalam 1-2 kalimat saja, sesuai konteks dan tone postingannya.
    """
    agent = PromptGenAgent(
        system_prompt=system_prompt,
        human_prompt=human_prompt,
        model_name="",  #
        base_url="",
        api_key="EMPTY",  
        temperature=0.3,
        stage="test"
    )
    example_markdown = """
    """
    try:
        logger.info("Running NarrativeContentAgent test...")
        result = agent.analyze(data_input=example_markdown)
        logger.info("Result from analyze():")
        logger.info(result)
    except Exception as e:
        logger.error(f"An error occurred during testing: {str(e)}")