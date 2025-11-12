import os
import re
import sys
import uuid
import json
import srsly
from dateutil import parser
from loguru import logger
from urllib.parse import urlparse
from typing import Dict, Any
from configparser import ConfigParser   

path_this = os.path.dirname(os.path.abspath(__file__))
path_project = os.path.dirname(os.path.join(path_this, ".."))
path_root = os.path.dirname(os.path.join(path_this, "../.."))
sys.path.extend([path_this, path_project, path_root])

from agents.agent_prompt_generator import PromptGenAgent
from tools.tools_generate_i2i import SDClientT2I

class ImageGenAgent:
    def __init__(self):
        self.config = ConfigParser()
        self.config.read(os.path.join(path_this, "config.ini"))
        self._init_agent()
        self._init_tools()

    def _init_agent(self):
        self.system_prompts_path = os.path.join(path_project, self.config.get("default", "system_prompt_path_copy"))
        self.system_prompts = srsly.read_json(self.system_prompts_path)
        self.agentpromptgenerator = PromptGenAgent(
            system_prompt=self.system_prompts['agent_com']['system_prompt'],
            human_prompt = """
            Here is the prompt:
            {data_input}
            
            """,
            agent_name="Prompt Generator Agent",
            base_url=self.config.get('default','base_url'),
            provider="vl-model",
            model_name=self.config.get('default','model_name'),  
            api_key="api_key",
            max_retries=3,
        
        )

    def _init_tools(self):
        self.agent_text2img = SDClientT2I()

    
    def process_generate_image(self,prompt:str):
        logger.info(f"process generate photo with prompt: {prompt}")
        session_id = f"session_{uuid.uuid4().hex[:8]}"

        process_generate_prompt = self.agentpromptgenerator.analyze(data_input=prompt)
        

        if not process_generate_prompt:
            raise ValueError("Agent tidak mengembalikan komentar (respons kosong)")

        logger.success("Komentar berhasil digenerate")
            
        cleaned_text_prompt = (
            str(process_generate_prompt).strip('"')
                .replace('\\"', '"')
                .replace('"', '')
                .replace('{', '')
                .replace('}', '')
                .replace('response:', '')
                .strip()
            )
        logger.info(f"result generator prompt: {cleaned_text_prompt}")

        process_generate_photo = self.agent_text2img.generate(cleaned_text_prompt)

        get_base_64 = process_generate_photo.get("base64","")
        get_path = process_generate_photo.get("path","")
        
        metadata = {
            "id":session_id,
            "base64":get_base_64,
            "path_file":get_path
        }

        return metadata

if __name__ == "__main__":

    prompt=""
    logger.info("Initializing AgentCommentTask...")
    agent = ImageGenAgent()
    
    logger.info("Testing Prompt Image processing...")
    result_photo_gen = agent.process_generate_image(prompt=prompt)
    logger.info(result_photo_gen)

