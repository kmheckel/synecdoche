# Running with Ollama

No cloud API key required. pydantic-ai's `OpenAIChatModel` accepts an
arbitrary base URL, so pointing it at Ollama's OpenAI-compatible endpoint
is enough.

## Setup

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:7b-instruct    # any model works
ollama serve                        # if it isn't already
```

## Runnable example

```python
from typing import Literal
from pydantic import BaseModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from synecdoche import Runtime

class Sentiment(BaseModel):
    label: Literal["positive", "negative", "neutral"]
    confidence: float

ollama = OpenAIChatModel(
    "qwen2.5:7b-instruct",
    provider=OpenAIProvider(base_url="http://localhost:11434/v1", api_key="ollama"),
)

rt = Runtime(model=ollama, trace="tree")

@rt.infer
def classify_sentiment(text: str) -> Sentiment:
    """Classify sentiment as positive, negative, or neutral."""

print(classify_sentiment("I love this."))
```

See also: [`examples/ollama_infer.py`](https://github.com/kmheckel/synecdoche/blob/main/examples/ollama_infer.py).
