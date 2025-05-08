"""
定义与大语言模型交互的客户端
"""

import os
from typing import Dict, Any, Optional

from config.settings import LLM_API_KEY, LLM_MODEL_NAME, LLM_PROVIDER

class LLMClient:
    """与大语言模型交互的客户端"""
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        provider: Optional[str] = None
    ):
        self.api_key = api_key or LLM_API_KEY
        self.model_name = model_name or LLM_MODEL_NAME
        self.provider = provider or LLM_PROVIDER
        
        if not self.api_key:
            raise ValueError("未设置LLM API密钥。请在配置文件或环境变量中设置LLM_API_KEY。")
            
        # 根据提供商初始化相应的客户端
        if self.provider.lower() == "openai":
            self._init_openai_client()
        elif self.provider.lower() == "anthropic":
            self._init_anthropic_client()
        else:
            raise ValueError(f"不支持的LLM提供商: {self.provider}")
    
    def _init_openai_client(self):
        """初始化OpenAI客户端"""
        try:
            import openai
            openai.api_key = self.api_key
            self.client = openai
        except ImportError:
            raise ImportError("使用OpenAI需要安装openai包。请运行: pip install openai")
    
    def _init_anthropic_client(self):
        """初始化Anthropic客户端"""
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("使用Anthropic需要安装anthropic包。请运行: pip install anthropic")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        生成文本响应
        
        Args:
            prompt: 输入提示
            **kwargs: 传递给底层API的额外参数
            
        Returns:
            模型生成的文本
        """
        if self.provider.lower() == "openai":
            return self._generate_openai(prompt, **kwargs)
        elif self.provider.lower() == "anthropic":
            return self._generate_anthropic(prompt, **kwargs)
        else:
            raise ValueError(f"不支持的LLM提供商: {self.provider}")
    
    def _generate_openai(self, prompt: str, **kwargs) -> str:
        """使用OpenAI生成文本"""
        default_params = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 500
        }
        
        # 合并默认参数和自定义参数
        params = {**default_params, **kwargs}
        
        # 调用OpenAI API
        response = self.client.ChatCompletion.create(**params)
        return response.choices[0].message.content
    
    def _generate_anthropic(self, prompt: str, **kwargs) -> str:
        """使用Anthropic生成文本"""
        default_params = {
            "model": self.model_name,
            "max_tokens_to_sample": 500,
            "temperature": 0.7
        }
        
        # 合并默认参数和自定义参数
        params = {**default_params, **kwargs}
        
        # 调用Anthropic API
        response = self.client.completions.create(
            prompt=f"\n\nHuman: {prompt}\n\nAssistant:",
            **params
        )
        return response.completion

def get_llm_client() -> LLMClient:
    """获取LLM客户端实例"""
    return LLMClient() 