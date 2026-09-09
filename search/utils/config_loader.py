# Config loader for environment-specific settings
import yaml
import os


class ConfigLoader:
    """環境別設定を読み込むユーティリティクラス"""
    
    @staticmethod
    def load_config(config_path):
        """YAMLファイルから設定を読み込む"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    @staticmethod
    def get_default_config_path(env_name=None):
        """環境名からデフォルト設定ファイルパスを取得"""
        configs_dir = os.path.join(os.path.dirname(__file__), 'configs')
        
        if env_name is None:
            # 環境変数から環境名を判定
            env_name = os.environ.get('PROXYLESS_ENV', '5080')
        
        config_file = f"config_{env_name}.yaml"
        config_path = os.path.join(configs_dir, config_file)
        
        return config_path
    
    @staticmethod
    def get_latency_model_url(config, target_hardware):
        """ハードウェアターゲットに応じたレイテンシモデルURLを取得"""
        if target_hardware is None:
            return None
        
        key_map = {
            'mobile': 'mobile_latency_model',
            'cpu': 'cpu_latency_model',
            'gpu8': 'gpu8_latency_model',
            'flops': None,
        }
        
        key = key_map.get(target_hardware)
        if key and key in config:
            return config[key]
        
        # デフォルトはmobile
        return config.get('mobile_latency_model')
