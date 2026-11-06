"""Load EasyTSAD/Harness/config.toml and derive subprocess environments."""
from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parents[1]          # EasyTSAD/Harness
ROOT = HARNESS_DIR.parents[1]                               # Live-EasyTSAD repo root
CONFIG_PATH = Path(os.environ.get("TSAD_HARNESS_CONFIG", HARNESS_DIR / "config.toml"))
PROMPTS_DIR = HARNESS_DIR / "prompts"


class HarnessError(RuntimeError):
    pass


@dataclass(frozen=True)
class Config:
    raw: dict

    def __getitem__(self, key: str) -> dict:
        return self.raw[key]

    def path(self, section: str, key: str) -> Path:
        p = Path(os.path.expanduser(self.raw[section][key]))
        return p if p.is_absolute() else ROOT / p

    @property
    def tasks_dir(self) -> Path:
        return self.path("paths", "tasks_dir")

    @property
    def dsh_home(self) -> Path:
        return self.path("paths", "dsh_home")

    @property
    def methods_dir(self) -> Path:
        return self.path("paths", "methods_dir")

    @property
    def dsh_bin(self) -> Path:
        return ROOT / "envs" / "harness" / "venv" / "bin" / "dsh"

    @property
    def python_bin(self) -> Path:
        return ROOT / "envs" / "harness" / "venv" / "bin" / "python"

    # ------------------------------------------------------------------ secrets
    def api_key(self) -> str:
        """Key from $api_key_env (the var the user exports), $dsh_key_env, or api_key_file."""
        llm = self.raw["llm"]
        key = os.environ.get(llm["api_key_env"], "") or os.environ.get(self.dsh_key_env, "")
        if not key and llm.get("api_key_file"):
            f = Path(os.path.expanduser(llm["api_key_file"]))
            if f.exists():
                key = json.loads(f.read_text()).get(llm.get("api_key_file_field", "OPENAI_API_KEY"), "")
        if not key:
            raise HarnessError(f"no API key: export ${llm['api_key_env']} (the harness hands it to dsh as "
                               f"${self.dsh_key_env}) or provide {llm.get('api_key_file')}")
        return key

    @property
    def dsh_key_env(self) -> str:
        """Var name used inside dsh; must match DSH's scrub regex so children never see it."""
        return self.raw["llm"].get("dsh_key_env", "YYT_API_KEY")

    def _drop_secrets(self, env: dict) -> dict:
        """Remove the API key (source var, dsh var, and any var holding the same value)."""
        names = (self.raw["llm"]["api_key_env"], self.dsh_key_env)
        values = {os.environ.get(n) for n in names} - {None, ""}
        return {k: v for k, v in env.items() if k not in names and v not in values}

    # -------------------------------------------------------------- environment
    def network_env(self, base: dict | None = None) -> dict:
        """Environment with NO_PROXY extended by the configured direct hosts; never carries the API key."""
        env = dict(os.environ if base is None else base)
        hosts = list(self.raw["network"].get("direct_hosts", []))
        for var in ("NO_PROXY", "no_proxy"):
            cur = [h for h in env.get(var, "").split(",") if h]
            env[var] = ",".join(dict.fromkeys(cur + hosts))
        if self.raw["network"].get("drop_socks_proxy", True):
            for var in ("ALL_PROXY", "all_proxy"):
                if env.get(var, "").startswith("socks"):
                    env.pop(var)
        env["TSAD_ROOT"] = str(ROOT)
        return self._drop_secrets(env)

    def agent_env(self) -> dict:
        """Environment for the `dsh` process: network + API key + DSH paths."""
        key = self.api_key()
        env = {k: v for k, v in self.network_env().items() if v != key}
        env[self.dsh_key_env] = key
        env["DSH_HOME"] = str(self.dsh_home)
        env["DSH_PERMISSION_MODE"] = self.raw["dsh"]["permission_mode"]
        env["TQDM_DISABLE"] = env.get("TQDM_DISABLE", "1")
        return env


@lru_cache(maxsize=1)
def load() -> Config:
    with open(CONFIG_PATH, "rb") as f:
        return Config(tomllib.load(f))
