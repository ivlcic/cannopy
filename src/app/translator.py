import re
import logging

from typing import Any, Dict, List, Callable

from .args.data import TranslateModelsConfig

logger = logging.getLogger('core.labels')


class Translator:
    fn = Callable[[List[str]], List[str]]

    _TEXT_RE = re.compile(r"<text>(.*?)</text>", re.DOTALL)

    _api_clients: Dict[str, Any] = {}

    @classmethod
    def encapsulate(cls, items: List[str]) -> str:
        return '\n'.join([f"<text>{s}</text>" for s in items])

    @classmethod
    def decapsulate(cls, s: str) -> List[str]:
        return cls._TEXT_RE.findall(s)

    @classmethod
    def translate(cls, payload: List[str], prompt: str, models: TranslateModelsConfig) -> List[str]:
        model = models.default
        if 'openai' in model.provider:
            if 'openai' in cls._api_clients:
                client = cls._api_clients['openai']
            else:
                from openai import OpenAI
                client = OpenAI()
                cls._api_clients['openai'] = client
                logger.debug('Calling OpenAI with model=%s', model.parameters['model'])

            texts = cls.encapsulate(payload)
            body = {
                'messages': [
                    {'role': 'system', 'content': prompt},
                    {'role': 'user', 'content': texts},
                ],
            }
            body = body | model.parameters
            response = client.chat.completions.create(**body)
            result = response.choices[0].message.content.strip()
            translated = cls.decapsulate(result)
            return translated
        return []
