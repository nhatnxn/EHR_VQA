import yaml
import pprint
import os 

class Cfg(dict):
    def __init__(self, config_dict):
        super(Cfg, self).__init__(**config_dict)
        self.__dict__ = self

    @staticmethod
    def load_config_from_file(fname, download_base=False):
        if not os.path.exists(fname):
            raise FileNotFoundError("Not found config at {}".format(fname))
        with open(fname, encoding='utf-8') as f:
            base_config = yaml.safe_load(f)
        return Cfg(base_config)

    def save(self, fname):
        with open(fname, 'w', encoding='utf-8') as outfile:
            yaml.dump(dict(self), outfile, default_flow_style=False, allow_unicode=True)

    # @property
    def pretty_text(self):
        return pprint.PrettyPrinter().pprint(self)

