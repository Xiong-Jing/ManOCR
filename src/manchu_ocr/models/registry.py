from typing import Callable, Dict, Type


class Registry:
    def __init__(self, name: str):
        self.name = name
        self._modules: Dict[str, Type] = {}

    def register(self, name: str | None = None) -> Callable:
        def decorator(cls):
            module_name = name or cls.__name__

            if module_name in self._modules:
                raise KeyError(f"{module_name} is already registered in {self.name}")

            self._modules[module_name] = cls
            return cls

        return decorator

    def get(self, name: str):
        if name not in self._modules:
            available = ", ".join(self._modules.keys())
            raise KeyError(
                f"{name} is not registered in {self.name}. "
                f"Available: {available}"
            )

        return self._modules[name]

    def build(self, cfg: dict):
        cfg = cfg.copy()
        name = cfg.pop("name")
        cls = self.get(name)
        return cls(**cfg)


BACKBONES = Registry("backbone")
NECKS = Registry("neck")
HEADS = Registry("head")
DETECTORS = Registry("detector")
RECOGNIZERS = Registry("recognizer")
