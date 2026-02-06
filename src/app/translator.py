import logging
import os
import re
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from re import Pattern
from typing import Any, Dict, List, Type, Callable, Tuple

import torch
import requests
from transformers import AutoProcessor, SeamlessM4TForTextToText

from .args.data import TranslateModelConfig, TranslateConfig
from .pip import Pip

logger = logging.getLogger('core.translator')


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
        name = config.model.provider
        key = name.strip().lower()
        if key not in cls._registry:
            raise ValueError(f"Unknown translator '{name}'. Available: {sorted(cls._registry)}")

        return cls._registry[key](config)

    def __init__(self, config: TranslateConfig) -> None:
        self.config = config
        self.model_cfg: TranslateModelConfig = config.model
        self.prompt = config.prompt
        self.max_payload_threads = config.max_payload_threads
        self.max_batch_threads = config.max_batch_threads
        self.max_chars_per_payload = config.max_chars_per_payload
        self.prompt = self.prompt.replace("{SOURCE_LANG}", self.config.src_lang)
        self.prompt = self.prompt.replace("{TARGET_LANG}", self.config.tgt_lang)
        self.prompt = self.prompt.replace("{SOURCE_CODE}", self.config.src_code)
        self.prompt = self.prompt.replace("{TARGET_CODE}", self.config.tgt_code)

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
                raise ValueError(f'Invalid translation [{trans}] for [{k}] from response text: [{response_text}]')
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
        with ThreadPoolExecutor(max_workers=min(self.max_payload_threads, len(parts))) as executor:
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
            merged = self._translate_item(payload, keys, 1)
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
        num_retries = 5
        results: List[Dict[str, Any]] = [{} for _ in items]
        while num_retries > 0:
            try:
                with ThreadPoolExecutor(max_workers=min(len(items), self.max_batch_threads)) as executor:
                    future_map = {executor.submit(self.translate, p, keys): idx for idx, p in enumerate(items)}
                    for future in as_completed(future_map):
                        idx = future_map[future]
                        results[idx] = future.result()
                return results
            except Exception as e:
                logger.error(e)
                num_retries -= 1
                if num_retries == 0:
                    raise e
                else:
                    logger.info(f'Retrying [{num_retries}]...')
        return results  # this never happens


# noinspection DuplicatedCode
@Translator.register("openai")
class OpenaiTranslator(Translator):

    def __init__(self, config: TranslateConfig) -> None:
        super().__init__(config)
        # intentional inline install and import
        Pip.install_packages("openai", "2.14.0")
        # noinspection PyUnresolvedReferences,PyPackageRequirements
        from openai import OpenAI
        self.client = OpenAI()
        logger.info('Creating OpenAI client with model=%s', self.model_cfg.parameters['model'])

    def _translate_payload(self, payload: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
        # if not payload:
        #     return payload
        # lines, regs = self._encapsulate(payload, keys)
        # body = {
        #     'messages': [
        #         {'role': 'system', 'content': self.sys_prompt},
        #         {'role': 'user', 'content': '\n'.join(lines)},
        #     ],
        # }
        #
        # body = body | self.model_cfg.parameters
        # response = self.client.chat.completions.create(**body)
        # response_text = response.choices[0].message.content.strip()
        # result = self._decapsulate(response_text, regs)
        if not payload:
            return payload

        lines, regs = self._encapsulate(payload, keys)

        # Responses API uses `input` (and supports role/content message items)
        body = {
            "input": [
                {"role": "system", "content": [{"type": "text", "text": self.prompt}]},
                {"role": "user", "content": [{"type": "text", "text": "\n".join(lines)}]},
            ],
        }

        # merge model parameters (e.g., model, temperature, max_output_tokens, etc.)
        body |= self.model_cfg.parameters

        response = self.client.responses.create(**body)

        # Most commonly you’ll want `output_text` (SDK convenience) if available:
        response_text = (getattr(response, "output_text", None) or "").strip()

        # Fallback if your SDK version doesn’t expose output_text:
        if not response_text:
            # Concatenate any text chunks from an output
            chunks = []
            for item in getattr(response, "output", []) or []:
                for part in getattr(item, "content", []) or []:
                    if getattr(part, "type", None) == "output_text":
                        chunks.append(getattr(part, "text", ""))
            response_text = "".join(chunks).strip()

        result = self._decapsulate(response_text, regs)
        return result


# noinspection DuplicatedCode
@Translator.register("groq")
class GroqTranslator(Translator):

    def __init__(self, config: TranslateConfig) -> None:
        super().__init__(config)
        # intentional inline install and import
        Pip.install_packages("groq", "1.0.0")
        # noinspection PyUnresolvedReferences,PyPackageRequirements
        from groq import Groq
        api_key = os.environ.get("GROQ_API_KEY")
        self.client = Groq()
        logger.info(
            'Creating Groq client with model=%s with api key=%s...',
            self.model_cfg.parameters['model'], api_key[0:7]
        )

    def _translate_payload(self, payload: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
        if not payload:
            return payload
        lines, regs = self._encapsulate(payload, keys)
        body = {
            'messages': [
                {'role': 'system', 'content': self.prompt},
                {'role': 'user', 'content': '\n'.join(lines)},
            ],
        }

        body = body | self.model_cfg.parameters
        response = self.client.chat.completions.create(**body)
        response_text = response.choices[0].message.content.strip()
        result = self._decapsulate(response_text, regs)
        return result


class OllamaTranslator(Translator):

    def __init__(self, config: TranslateConfig) -> None:
        super().__init__(config)
        self.base_url = self.model_cfg.parameters.get("base_url", "http://localhost:11434")
        self.model_name = self.model_cfg.parameters.get("model", "")
        if not self.model_name:
            raise ValueError("Ollama translator requires model.parameters.model to be set.")
        logger.info('Creating Ollama client with model=%s at %s', self.model_name, self.base_url)

    def _make_body(self, text: str) -> Dict[str, Any]:
        body = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": text},
            ],
            "stream": False,
        }
        return body

    def _make_request(self, body: Dict[str, Any]) -> str:
        extra = dict(self.model_cfg.parameters)
        extra.pop("model", None)
        extra.pop("base_url", None)
        body = body | extra

        resp = requests.post(f"{self.base_url}/api/chat", json=body, timeout=600)
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"].strip()

    def _translate_payload(self, payload: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
        if not payload:
            return payload
        # lines, regs = self._encapsulate(payload, keys)
        result = {}
        for key in keys:
            text = payload.get(key, None)
            if isinstance(text, list):
                items = []
                for item in text:
                    body = self._make_body(item)
                    items.append(self._make_request(body))
                result[key] = items
            else:
                body = self._make_body(text)
                response_text = self._make_request(body)
                result[key] = response_text
        return result


@Translator.register("ollama-eurollm-9b-it")
class OllamaEuroLLMTranslator(OllamaTranslator):

    def __init__(self, config: TranslateConfig) -> None:
        super().__init__(config)

    def _make_body(self, text: str) -> Dict[str, Any]:
        body = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": self.prompt + "\n" + text},
            ],
            "stream": False,
        }
        return body


@Translator.register("ollama-translate-gemma")
class OllamaTranslateGemmaTranslator(OllamaTranslator):

    def __init__(self, config: TranslateConfig) -> None:
        super().__init__(config)

    def _make_body(self, text: str) -> Dict[str, Any]:
        body = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": self.prompt + "\n" + text},
            ],
            "stream": False,
        }
        return body


@Translator.register("local-seamless-m4t")
class SeamlessM4t(Translator):

    # noinspection SpellCheckingInspection
    _LANG_MAP = {
        "en": "eng",
        "sl": "slv",
        "sr": "srp",
        "sr-cyrl": "srp",
        "hr": "hrv",
        "cs": "ces",
        "sk": "slk",
        "mk": "mkd",
        "bg": "bul",
        "pl": "pol",
        "uk": "ukr",
        "de": "deu",
        "fr": "fra",
        "es": "spa",
        "it": "ita",
        "pt": "por",
        "ru": "rus",
    }

    @classmethod
    def _map_lang(cls, lang: str) -> str:
        if not lang:
            return lang
        key = lang.strip().lower()
        return cls._LANG_MAP.get(key, key)

    @staticmethod
    def _resolve_dtype(value: Any) -> torch.dtype | None:
        if not value:
            return None
        if isinstance(value, torch.dtype):
            return value
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"fp16", "float16", "torch.float16"}:
                return torch.float16
            if v in {"bf16", "bfloat16", "torch.bfloat16"}:
                return torch.bfloat16
            if v in {"fp32", "float32", "torch.float32"}:
                return torch.float32
        return None

    def __init__(self, config: TranslateConfig) -> None:
        super().__init__(config)
        model_name = self.model_cfg.parameters.get("model", "facebook/seamless-m4t-v2-large")
        torch_dtype = self._resolve_dtype(self.model_cfg.parameters.get("torch_dtype"))
        device = self.model_cfg.parameters.get("device", None)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = SeamlessM4TForTextToText.from_pretrained(model_name, torch_dtype=torch_dtype)
        self.model.to(self.device)
        self.model.eval()
        self.src_code = self._map_lang(self.config.src_code)
        self.tgt_code = self._map_lang(self.config.tgt_code)
        logger.info('Loaded SeamlessM4T model=%s on %s', model_name, self.device)

    def _translate_payload(self, payload: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
        if not payload:
            return payload

        gen_kwargs = dict(self.model_cfg.parameters)
        gen_kwargs.pop("model", None)
        gen_kwargs.pop("torch_dtype", None)
        gen_kwargs.pop("device", None)

        result = {}
        for key in keys:
            text = payload.get(key, None)
            inputs = self.processor(text=text, src_lang=self.src_code, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.inference_mode():
                # noinspection PyUnresolvedReferences
                output = self.model.generate(
                    **inputs,
                    tgt_lang=self.tgt_code,
                    **gen_kwargs,
                )
            decoded = self.processor.decode(output[0], skip_special_tokens=True)
            result[key] = decoded
        # if self.device == "cuda":
        #     torch.cuda.empty_cache()
        return result
