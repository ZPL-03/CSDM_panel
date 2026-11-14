from core import paths
from core.paths import CASES_DIR, CHROMA_DIR


def test_runtime_paths_and_imported_aliases_are_isolated(tmp_path):
    assert paths.DATA_DIR == tmp_path / "data"
    assert CASES_DIR == paths.CASES_DIR == tmp_path / "data/cases"
    assert CHROMA_DIR == paths.CHROMA_DIR == tmp_path / "knowledge/chroma_db"
    assert not paths.MODELS_DIR.is_relative_to(paths.ROOT_DIR)
    assert paths.EXTERNAL_KNOWLEDGE_DIR == paths.ROOT_DIR / "knowledge/external"
