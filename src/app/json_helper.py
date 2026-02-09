import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger('json-helper')

class JsonHelper:

    @classmethod
    def read_ir_sample(cls, line: str, line_no: int, source: Path) -> Optional[Dict[str, Any]]:
        line = line.strip()
        if not line:
            return None
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            logger.warning('Skipping malformed JSON in %s line %d.', source.name, line_no)
            return None
        if 'query' not in obj:
            logger.warning(
                'Skipping malformed JSON in %s line %d, missing query.', source.name, line_no
            )
            return None
        if 'pos' not in obj:
            logger.warning(
                'Skipping malformed JSON in %s line %d, missing positive samples.', source.name, line_no
            )
            return None
        if 'neg' not in obj:
            logger.warning(
                'Skipping malformed JSON in %s line %d, missing negative samples.', source.name, line_no
            )
            return None
        return obj