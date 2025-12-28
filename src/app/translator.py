import re
import logging
from collections import defaultdict

from re import Pattern
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type, Callable, Tuple

from .args.data import TranslateModelConfig, TranslateConfig

logger = logging.getLogger('core.labels')


class Translator(ABC):
    fn = Callable[[List[str]], List[str]]

    _registry: Dict[str, Type["Translator"]] = {}

    @classmethod
    def register(cls, name: str):
        key = name.strip().lower()

        def decorator(subclass: Type["Translator"]) -> Type["Translator"]:
            cls._registry[key] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, config: TranslateConfig) -> "Translator":
        name = config.models.default.provider
        key = name.strip().lower()
        if key not in cls._registry:
            raise ValueError(f"Unknown translator '{name}'. Available: {sorted(cls._registry)}")

        return cls._registry[key](config)

    def __init__(self, config: TranslateConfig) -> None:
        self.config = config
        self.model: TranslateModelConfig = self.config.models.default
        self.sys_prompt = self.config.prompt

    # noinspection PyMethodMayBeStatic
    def _encapsulate(self, payload: Dict[str, Any], keys: List[str]) -> Tuple[List[str], Dict[str, Pattern]]:
        lines = []
        regs: Dict[str, Pattern] = {}
        for k in keys:
            v = payload.get(k, None)
            if v is None:
                continue
            if not v:
                continue
            if type(v) is list:
                k = 'l_' + k
                lines.extend([f'<{k}>{x}</{k}>' for x in v])
            elif type(v) is str:
                lines.append(f'<{k}>{v}</{k}>')
            else:
                raise ValueError(f'Invalid value for {k}')

            if k not in regs:
                regs[k] = re.compile(rf'<{k}>(.*?)</{k}>', re.DOTALL)
        return lines, regs

    # noinspection PyMethodMayBeStatic
    def _decapsulate(self, response_text: str, regs: Dict[str, Pattern]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for k, v in regs.items():
            trans: List[str] = v.findall(response_text)
            if 'l_' in k:
                k = k[2:]
                if k not in result:
                    result[k] = trans
                else:
                    result[k].extend(trans)
            elif len(trans) == 1:
                result[k] = trans[0]
            else:
                raise ValueError(f'Invalid translation for {k}')
        return result

    @abstractmethod
    def _translate(self, payload: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
        raise NotImplementedError

    def trans(self, payload: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
        result = self._translate(payload, keys)
        return result

    def trans_batch(self, payload: List[Dict[str, Any]], keys: List[str]) -> List[Dict[str, Any]]:
        batch: List[Dict[str, Any]] = []
        for p in payload:
            result = self._translate(p, keys)
            batch.append(result)
        return batch


@Translator.register("openai")
class OpenaiTranslator(Translator):

    def __init__(self, config: TranslateConfig) -> None:
        super().__init__(config)
        from openai import OpenAI
        self.client = OpenAI()
        logger.debug('Creating OpenAI client with model=%s', self.model.parameters['model'])

    def _translate(self, payload: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
        if not payload:
            return payload
        lines, regs = self._encapsulate(payload, keys)
        body = {
            'messages': [
                {'role': 'system', 'content': self.sys_prompt},
                {'role': 'user', 'content': '\n\n'.join(lines)},
            ],
        }

        body = body | self.model.parameters
        response = self.client.chat.completions.create(**body)
        response_text = response.choices[0].message.content.strip()
        result = self._decapsulate(response_text, regs)
        return result

