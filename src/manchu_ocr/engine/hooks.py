class Hook:
    """Minimal training hook interface."""

    def on_train_begin(self, trainer) -> None:
        pass

    def on_epoch_begin(self, trainer, epoch: int) -> None:
        pass

    def on_epoch_end(self, trainer, epoch: int, metrics: dict) -> None:
        pass

    def on_train_end(self, trainer) -> None:
        pass


class HookList:
    def __init__(self, hooks=None):
        self.hooks = list(hooks or [])

    def call(self, name: str, *args, **kwargs) -> None:
        for hook in self.hooks:
            method = getattr(hook, name, None)
            if method is not None:
                method(*args, **kwargs)
