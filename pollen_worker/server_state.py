import json
from flwr.common import Parameters
from flwr.server.history import History
from flwr.common import NDArrays

class ServerState(object):

    def __init__(
        self,
        id: str,
        round: int,
        momentum: NDArrays | None,
        elapsed_time_in_seconds: float,
        history: History
    ) -> None:
        if (not isinstance(id, str)) or len(id) < 1:
            raise TypeError("id is not a non-empty string")
        if not isinstance(round, int) or round < 1:
            raise TypeError("server_round is not a positive non-zero integer")
        if (not isinstance(momentum, list) and momentum is not None) or isinstance(momentum, Parameters):
            raise TypeError("momentum is not an instance of NDArrays or None")
        if not isinstance(elapsed_time_in_seconds, float) and not isinstance(elapsed_time_in_seconds, int):
            raise TypeError("elapsed_time_in_seconds is not a numeric value")
        if not isinstance(history, History):
            raise TypeError("history is not an instance of History")

        self.id: str = id
        self.round: int = round
        self.momentum: NDArrays | None = momentum
        self.elapsed_time_in_seconds: float = elapsed_time_in_seconds if isinstance(elapsed_time_in_seconds, float) else float(elapsed_time_in_seconds)
        self.history: History = history

    def toJson(self) -> str:
        return json.dumps({
            "id": self.id,
            "round": self.round,
            "contains_momentum": self.momentum is not None,
            "elapsed_time_in_seconds": self.elapsed_time_in_seconds
        })

class ServerStateWithGlobalModel(ServerState):
    def __init__(
        self,
        id: str,
        round: int,
        global_model: NDArrays,
        momentum: NDArrays | None,
        elapsed_time_in_seconds: float,
        history: History
    ) -> None:
        super().__init__(id, round, momentum, elapsed_time_in_seconds, history)
        if not isinstance(global_model, list) or isinstance(global_model, Parameters):
            raise TypeError("global_model is not an instance of NDArrays")
        self.global_model: NDArrays = global_model
