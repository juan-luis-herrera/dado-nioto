class ConfigLoader:

    def __init__(self, filename: str):
        self.__file = filename

    def load_json_config(self) -> dict:
        import json
        with open(self.__file, 'r') as in_config:
            config = json.load(in_config)
        return config

class TimingReport:

    def __init__(self, filename: str):
        self.__file = filename
    
    def export_as_csv(self, report: dict):
        import csv
        with open(self.__file, 'w', newline='') as out_csv:
            writer = csv.writer(out_csv)
            writer.writerow(('Event', 'Time (s)'))
            for k, v in report.items():
                writer.writerow((k, v))