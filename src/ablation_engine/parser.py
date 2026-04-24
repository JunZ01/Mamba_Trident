
import yaml
from pathlib import Path
from typing import Dict, Any, List


class AblationConfigParser:

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.base_config: Dict[str, Any] = self.config['base_config']
        self.ablations: Dict[str, Any] = self.config['ablations']
        self.execution_config: Dict[str, Any] = self.config['execution']

    def get_experiment_configs(self) -> List[Dict[str, Any]]:

        run_priorities = self.execution_config.get('run_priorities', [0, 1, 2, 3])

        enabled_experiments = []
        for exp_id, exp_config in self.ablations.items():
            if exp_config.get('enabled', False):
                if exp_config.get('priority', 99) in run_priorities:
                    exp_config['experiment_id'] = exp_id
                    enabled_experiments.append(exp_config)

        enabled_experiments.sort(key=lambda x: (x.get('priority', 99), x['experiment_id']))

        print(f"Found {len(enabled_experiments)} enabled experiments to run.")
        return enabled_experiments