from core.config_loader import load_app_config


def test_abaqus_command_uses_machine_environment(monkeypatch) -> None:
    load_app_config.cache_clear()
    try:
        monkeypatch.setenv("ABAQUS_COMMAND", "D:/SIMULIA/Commands/abaqus.bat")
        assert load_app_config()["abaqus"]["command"] == "D:/SIMULIA/Commands/abaqus.bat"
    finally:
        load_app_config.cache_clear()
