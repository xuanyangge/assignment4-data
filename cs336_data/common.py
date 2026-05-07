from pathlib import Path

import modal

MODAL_SHARED_PATH = Path("/shared-data")


def get_shared_assets_path() -> Path:
    if modal.is_local():
        return Path("local-shared-data").resolve()
    else:
        return MODAL_SHARED_PATH
