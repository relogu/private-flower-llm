"""A ray actor pool node-manager interface."""

from flwr.simulation.ray_transport.ray_actor import (
    VirtualClientEngineActor,
    ClientAppFn,
)


from flwr.client.client_app import ClientApp, ClientAppException, LoadClientAppError

import ray

from flwr.common.context import Context


from collections.abc import Callable

from flwr.common import (
    Message,
)
from flower_llm.conf.base_schema import S3CommConfig

from flwr.client.typing import ClientFnExt, Mod
from flower_llm.clients.virtual_llm_client import VirtualLLMClient
from flower_llm.node_manager.node_manager import NodeManager
from flower_llm.node_manager.node_manager_app import NodeManagerApp
from flower_llm.node_manager.utils import ModelParametersMetadata


@ray.remote
class VirtualClientEngineActorPollen(VirtualClientEngineActor):
    """Abstract base class for VirtualClientEngine Actors."""

    def __init__(
        self,
        client_fn: ClientFnExt | None,
        mods: list[Mod] | None = None,
    ) -> None:
        super().__init__()
        self.node_manager = NodeManagerApp(client_fn=client_fn, mods=mods)

    def run(
        self,
        client_app_fn: ClientAppFn,
        message: Message,
        cid: str,
        context: Context,
    ) -> tuple[str, Message, Context]:
        """Run a client run."""
        # Execute tasks and return result
        # return also cid which is needed to ensure results
        # from the pool are correctly assigned to each ClientProxy
        try:
            # Load app
            app: ClientApp = client_app_fn()

            # Handle task message
            out_message = app(message=message, context=(context, self.node_manager))  # type: ignore[reportArgumentType]

        except LoadClientAppError:
            raise
        except Exception as ex:
            raise ClientAppException(str(ex)) from ex

        return cid, out_message, context
