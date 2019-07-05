import importlib.util
spec = importlib.util.spec_from_file_location("config", "/home/oliverkn/cloud/eth/2019_FS/pro/pycharm/alad_mod/config_server.py")
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

print(config.result_path)