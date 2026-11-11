__version__ = "0.1.0"
ISSUES = "https://github.com/KatsuKumi/rwg2-migrator/issues"


class Refusal(Exception):
    def __init__(self, reason: str, action: str):
        super().__init__(f"{reason}. {action}")
        self.reason = reason
        self.action = action
