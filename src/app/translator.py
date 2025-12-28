import re
import logging

from re import Pattern
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        self.max_workers = 5
        self.max_batch_workers = 5
        self.max_chars_per_payload = 2000

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
    def _translate_payload(self, payload: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
        raise NotImplementedError

    # noinspection PyMethodMayBeStatic
    def _split_payload(self, payload: Dict[str, Any], keys: List[str], max_chars: int) -> List[Dict[str, Any]]:
        parts: List[Dict[str, Any]] = []
        current: Dict[str, Any] = {}
        current_len = 0

        def flush():
            nonlocal current, current_len
            if current:
                parts.append(current)
                current = {}
                current_len = 0

        for k in keys:
            val = payload.get(k)
            if val is None:
                continue
            if isinstance(val, str):
                val_len = len(val)
                if current_len + val_len > max_chars and current:
                    flush()
                current[k] = val
                current_len += val_len
            elif isinstance(val, list):
                for item in val:
                    item_len = len(item)
                    if current_len + item_len > max_chars and current:
                        flush()
                    current.setdefault(k, []).append(item)
                    current_len += item_len
            else:
                raise ValueError(f"Unsupported value type for key {k}: {type(val)}")
        flush()
        return parts or [payload]

    # noinspection PyMethodMayBeStatic
    def _merge_parts(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        for res in results:
            for k, v in res.items():
                if isinstance(v, list):
                    merged.setdefault(k, []).extend(v)
                else:
                    merged[k] = merged.get(k, "") + v
        return merged

    @staticmethod
    def _count_translatable_fields(payload: Dict[str, Any], keys: List[str]) -> int:
        total = 0
        for k in keys:
            v = payload.get(k)
            if isinstance(v, str):
                total += 1
            elif isinstance(v, list):
                total += len(v)
        return total

    def _translate_item(self, payload: Dict[str, Any], keys: List[str], max_chars: int = 2000) -> Dict[str, Any]:
        """
        Translate payload dict, splitting large payloads by character count while keeping fields together.
        """
        parts = self._split_payload(payload, keys, max_chars)
        num_payload = self._count_translatable_fields(payload, keys)

        results: List[Dict[str, Any]] = [{} for _ in parts]
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(parts))) as executor:
            future_map = {executor.submit(self._translate_payload, p, keys): idx for idx, p in enumerate(parts)}
            for future in as_completed(future_map):
                idx = future_map[future]
                results[idx] = future.result()

        merged = self._merge_parts(results)
        num_translated = self._count_translatable_fields(merged, keys)
        if num_translated != num_payload:
            # we already run at minimum i.e., safe mode
            if max_chars < 100:
                raise ValueError(
                    f"Translation result count mismatch after merge [{num_translated},{num_payload}] in safe mode"
                )
            # try to translate each field and field list item separately
            logger.warning(f"Translation result count mismatch after merge [{num_translated},{num_payload}]")
            logger.warning(f"Going in to a slow safe exec mode ...")
            merged = self._translate_item(merged, keys, 1)
            num_translated = self._count_translatable_fields(merged, keys)
            if num_translated != num_payload:
                raise ValueError(
                    f"Translation result count mismatch after merge [{num_translated},{num_payload}] in safe mode"
                )
        return merged

    def translate(self, item: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
        return self._translate_item(item, keys, max_chars=self.max_chars_per_payload)

    def translate_batch(self, items: List[Dict[str, Any]], keys: List[str]) -> List[Dict[str, Any]]:
        if not items:
            return []

        results: List[Dict[str, Any]] = [{} for _ in items]
        with ThreadPoolExecutor(max_workers=min(len(items), self.max_batch_workers)) as executor:
            future_map = {executor.submit(self.translate, p, keys): idx for idx, p in enumerate(items)}
            for future in as_completed(future_map):
                idx = future_map[future]
                results[idx] = future.result()
        return results


@Translator.register("openai")
class OpenaiTranslator(Translator):

    def __init__(self, config: TranslateConfig) -> None:
        super().__init__(config)
        from openai import OpenAI
        self.client = OpenAI()
        logger.debug('Creating OpenAI client with model=%s', self.model.parameters['model'])

    def _translate_payload(self, payload: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
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
