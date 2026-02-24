import logging
import os
import copy
import torch
import requests
import threading

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Type, Callable, Tuple, Optional

from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, SeamlessM4TForTextToText

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
        self.config = config
        self.model_cfg: TranslateModelConfig = config.model
        self.prompt = config.prompt
        self.max_payload_threads = config.max_payload_threads
        self.max_batch_threads = config.max_batch_threads
        self.prompt = self.prompt.replace("{SOURCE_LANG}", self.config.src_lang)
        self.prompt = self.prompt.replace("{TARGET_LANG}", self.config.tgt_lang)
        self.prompt = self.prompt.replace("{SOURCE_CODE}", self.config.src_code)
        self.prompt = self.prompt.replace("{TARGET_CODE}", self.config.tgt_code)
        self.strip_nl = self.config.attributes.get("strip_nl", False)
        self.strip_nbsp = self.config.attributes.get("strip_nbsp", False)

        self.device = self.model_cfg.parameters.get("device")
        if not self.device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = self._resolve_dtype(self.model_cfg.parameters.get("torch_dtype"))


    @abstractmethod
    def _translate_payload(self, key: str, payload: str) -> Tuple[str, Optional[str]]:
        raise NotImplementedError

    def _translate_item(self, payload: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
        """
        Translate payload dict, splitting large payloads by character count while keeping fields together.
        """
        flat_payload: Dict[str, str] = {}
        delim = '::::'
        for k in keys:
            idx = 0
            v = payload.get(k, None)
            if v is None:
                continue
            if not v:
                continue
            if type(v) is list:
                for idx, x in enumerate(v):
                    flat_payload[f'{k}{delim}{idx}'] = x
            elif type(v) is str:
                flat_payload[f'{k}{delim}{idx}'] = v
            else:
                raise ValueError(f'Invalid value for {k} it has to be a list or a string!')

        results = copy.deepcopy(payload)

        with ThreadPoolExecutor(max_workers=min(self.max_payload_threads, len(flat_payload))) as executor:
            future_map = {executor.submit(self._translate_payload, k, flat_payload[k]): k for k in flat_payload}
            for future in as_completed(future_map):
                k_indexed, translated_value = future.result()
                if self.strip_nl and translated_value is not None:
                    translated_value = translated_value.replace('\n', ' ')
                if self.strip_nbsp and translated_value is not None:
                    translated_value = translated_value.replace(' ', ' ')

                key, idx = k_indexed.split(delim, 1)
                idx = int(idx)
                element = results.get(key)
                if type(element) is list:
                    element[idx] = translated_value
                else:
                    results[key] = translated_value

        return results

    def translate(self, item: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
        return self._translate_item(item, keys)

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

    def _translate_payload(self, key: str, payload: str) -> Tuple[str, Optional[str]]:
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
            return key, None

        # Responses API uses `input` (and supports role/content message items)
        body = {
            "instructions": self.prompt,
            "input": payload
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

        return key, response_text


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

    def _translate_payload(self, key: str, payload: str) -> Tuple[str, str]:
        if not payload:
            return key, payload
        body = {
            'messages': [
                {'role': 'system', 'content': self.prompt},
                {'role': 'user', 'content': payload},
            ],
        }

        body = body | self.model_cfg.parameters
        response = self.client.chat.completions.create(**body)
        response_text = response.choices[0].message.content.strip()
        return key, response_text


class OllamaTranslator(Translator):

    def __init__(self, config: TranslateConfig) -> None:
        super().__init__(config)
        self.base_url = self.model_cfg.parameters.get("base_url", "http://localhost:11434")
        self.model_name = self.model_cfg.parameters.get("model", "")
        self.api_key = os.environ.get("OLLAMA_API_KEY")
        if not self.model_name:
            raise ValueError("Ollama translator requires model.parameters.model to be set.")
        logger.info('Creating Ollama client with model=%s at %s', self.model_name, self.base_url)

    def _make_body(self, text: str) -> Dict[str, Any]:
        body = {
            "messages": [
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": text},
            ],
            "stream": False,
        }
        body |= self.model_cfg.parameters
        body.pop("base_url", None)
        return body

    def _make_request(self, body: Dict[str, Any]) -> str:
        url = f"{self.base_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        resp = requests.post(url, json=body, headers=headers, timeout=600)
        resp.raise_for_status()
        data = resp.json()

        # OpenAI chat completions format
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"Unexpected /v1/chat/completions response shape: {data!r}") from e

        if not isinstance(content, str):
            raise ValueError(f"Non-text completion content: {content!r}")

        return content.strip()

    def _translate_payload(self, key: str, payload: str) -> Tuple[str, Optional[str]]:
        if not payload:
            return key, None
        body = self._make_body(payload)
        response_text = self._make_request(body)
        return key, response_text


class OllamaInContentTranslator(OllamaTranslator):

    def _make_body(self, text: str) -> Dict[str, Any]:
        body = {
            "messages": [
                {"role": "user", "content": self.prompt + "\n\n" + text},
            ],
            "stream": False,
        }
        body |= self.model_cfg.parameters
        body.pop("base_url", None)
        return body


@Translator.register("ollama-eurollm-9b-it")
class OllamaEuroLLMTranslator(OllamaInContentTranslator):
    pass


@Translator.register("ollama-translate-gemma")
class OllamaTranslateGemmaTranslator(OllamaInContentTranslator):
    pass


@Translator.register("ollama-gams-trans")
class OllamaGamsTranslator(OllamaInContentTranslator):
    pass


@Translator.register("google-translate-v3")
class GoogleTranslator(Translator):

    def __init__(self, config: TranslateConfig) -> None:
        super().__init__(config)
        self.api_key = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

        # intentional inline install and import
        Pip.install_packages("google-cloud-translate", "2.0.1")
        # noinspection PyUnresolvedReferences,PyPackageRequirements
        from google.cloud import translate

        self.client = translate.TranslationServiceClient()
        location = "global"
        project_id = self.model_cfg.parameters.pop("project_id", None)
        self.parent = f"projects/{project_id}/locations/{location}"
        logger.info("Creating Google Translate client (v3)")

    def _translate_payload(self, key: str, payload: str) -> Tuple[str, Optional[str]]:
        if not payload:
            return key, None

        response = self.client.translate_text(
            contents=[payload],
            source_language_code=self.config.src_code,
            target_language_code=self.config.tgt_code,
            mime_type="text/plain",
            parent=self.parent
        )

        response_text = response.translations[0].translated_text
        return key, response_text


class LocalModelTranslator(Translator, ABC):

    def __init__(self, config: TranslateConfig) -> None:
        super().__init__(config)
        self.model_name = self.model_cfg.parameters.get("model", None)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.model_args = dict(self.model_cfg.parameters)
        self.model_args.pop("model", None)
        self.model_args.pop("torch_dtype", None)
        self.model_args.pop("device", None)

        self._lock = threading.Lock()


@Translator.register("local-seamless-m4t")
class SeamlessM4t(LocalModelTranslator):

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

    def __init__(self, config: TranslateConfig) -> None:
        super().__init__(config)
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = SeamlessM4TForTextToText.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        self.src_code = self._map_lang(self.config.src_code)
        self.tgt_code = self._map_lang(self.config.tgt_code)
        logger.info('Loaded SeamlessM4T model=%s on %s', self.model_name, self.device)

    def _translate_payload(self, key: str, payload: str) -> Tuple[str, Optional[str]]:
        if not payload:
            return key, None

        inputs = self.processor(text=payload, src_lang=self.src_code, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self._lock, torch.inference_mode():
            # noinspection PyUnresolvedReferences
            output = self.model.generate(
                **inputs,
                tgt_lang=self.tgt_code,
                **self.model_args,
            )
        result = self.processor.decode(output[0], skip_special_tokens=True)
        return key, result


@Translator.register("eurollm-9b-instruct")
class EuroLLM9BInstructTranslator(LocalModelTranslator):

    def __init__(self, config: TranslateConfig) -> None:
        super().__init__(config)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        self._lock = threading.Lock()
        logger.info("Loaded EuroLLM translator model=%s on %s", self.model_name, self.device)

    def _translate_payload(self, key: str, payload: str) -> Tuple[str, Optional[str]]:
        if not payload:
            return key, None

        messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": payload},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)

        with self._lock, torch.inference_mode():
            output = self.model.generate(
                input_ids,
                **self.model_args,
            )

        # Decoder-only generation returns prompt + continuation; keep only newly generated tokens.
        generated = output[0][input_ids.shape[-1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return key, text
