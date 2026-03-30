"""Deprecated — use raven.providers.gitea instead."""
from raven.providers.gitea import GiteaProvider as GiteaClient, _split_repo

__all__ = ["GiteaClient", "_split_repo"]
