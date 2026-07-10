__all__ = ["PalletPackingEnv", "PalletPackingEnvCfg", "MaskedPalletMLP"]


def __getattr__(name):
    if name in {"PalletPackingEnv", "PalletPackingEnvCfg"}:
        from .pallet_packing_env import PalletPackingEnv, PalletPackingEnvCfg

        return {"PalletPackingEnv": PalletPackingEnv, "PalletPackingEnvCfg": PalletPackingEnvCfg}[name]
    if name == "MaskedPalletMLP":
        from .policies import MaskedPalletMLP

        return MaskedPalletMLP
    raise AttributeError(name)
