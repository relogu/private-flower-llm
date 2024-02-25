import json
from flwr.common import Parameters
from flwr.server.history import History

class ServerState(object):

    def __init__(
        self,
        id: str,
        round: int,
        global_model: Parameters,
        momentum: Parameters | None,
        elapsed_time_in_seconds: float,
        history: History
    ) -> None:

        if (not isinstance(id, str)) or len(id) < 1:
            raise TypeError("id is not a non-empty string")
        if not isinstance(round, int) or round < 1:
            raise TypeError("server_round is not a positive non-zero integer")
        if not isinstance(global_model, Parameters):
            raise TypeError("global_model is not an instance of Parameters")
        if not isinstance(momentum, Parameters) and momentum != None:
            raise TypeError("momentum is not an instance of Parameters or None")
        if not isinstance(elapsed_time_in_seconds, float) and not isinstance(elapsed_time_in_seconds, int):
            raise TypeError("elapsed_time_in_seconds is not a numeric value")
        if not isinstance(history, History):
            raise TypeError("history is not an instance of History")

        self.id: str = id
        self.round: int = round
        self.global_model: Parameters = global_model
        self.momentum: Parameters | None = momentum
        self.elapsed_time_in_seconds: float = elapsed_time_in_seconds if isinstance(elapsed_time_in_seconds, float) else float(elapsed_time_in_seconds)
        self.history: History = history

    def toJson(self) -> str:
        return json.dumps({
            "id": self.id,
            "round": self.round,
            "momentum": self.momentum != None,
            "elapsed_time_in_seconds": self.elapsed_time_in_seconds
        })
