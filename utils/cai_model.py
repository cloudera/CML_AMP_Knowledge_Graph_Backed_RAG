from typing import Any, Dict

from langchain_openai import OpenAI


class CAIHostedOpenAI(OpenAI):
    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        normal_params: Dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "n": self.n,
        }

        if self.max_tokens is not None:
            normal_params["max_tokens"] = self.max_tokens

        # Azure gpt-35-turbo doesn't support best_of
        # don't specify best_of if it is 1
        if self.best_of > 1:
            normal_params["best_of"] = self.best_of

        return {**normal_params, **self.model_kwargs}


def getCAIHostedOpenAIModels(
    base_url: str, model: str, api_key: str, max_tokens: int = 1024, **kwargs
) -> CAIHostedOpenAI:
    m = CAIHostedOpenAI(
        base_url=base_url,
        model=model,
        api_key=api_key,
        max_tokens=max_tokens,
        **kwargs,
    )
    old_create = m.client.create
    m.client.create = lambda prompt, **params: old_create(
        prompt=prompt[0] if isinstance(prompt, list) else prompt, **params
    )

    return m
