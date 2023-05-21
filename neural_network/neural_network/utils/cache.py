from typing import Any


class Cache(dict):
    def __setitem__(self, key: Any, value: Any) -> None:
        if key not in self:
            raise KeyError(f"Cache don't contain key: {key}")
        return super().__setitem__(key, value)

    def __getitem__(self, key: Any) -> None:
        if key not in self:
            raise KeyError(f"Cache don't contain key: {key}")
        return super().__getitem__(key)

    def update(self, **kwargs):
        for name, value in kwargs.items():
            self[name] = value

    def __getattribute__(self, __name: str) -> Any:
        try:
            return self[__name]
        except KeyError:
            return super().__getattribute__(__name)



