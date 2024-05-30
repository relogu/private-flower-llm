"""A ray actor pool node-manager interface."""

from typing import cast

from flwr.simulation.ray_transport.ray_actor import (
    VirtualClientEngineActor,
    ClientException,
    ClientRes,
    JobFn,
)
import traceback

import ray

from flwr.client import ClientFn
from flwr.common.context import Context
from flwr.simulation.ray_transport.utils import check_clientfn_returns_client


from collections.abc import Callable

from flwr.common import (
    NDArrays,
)
from flower_llm.conf.base_schema import S3CommConfig

from flower_llm.clients.virtual_llm_client import VirtualLLMClient
from flower_llm.node_manager.node_manager import NodeManager


@ray.remote
class VirtualClientEngineActorPollen(VirtualClientEngineActor):
    """Abstract base class for VirtualClientEngine Actors."""

    def __init__(
        self,
        client_fn: Callable[[int], VirtualLLMClient],
        run_uuid: str,
        parameters: NDArrays,
        refresh_period: int,
        cpu_only: bool,
        cpu_concurrency: int,
        use_s3_comm: bool = False,
        s3_comm_config: S3CommConfig | None = None,
    ) -> None:
        super().__init__()
        self.node_manager = NodeManager(
            client_fn=client_fn,
            run_uuid=run_uuid,
            parameters=parameters,
            refresh_period=refresh_period,
            cpu_only=cpu_only,
            cpu_concurrency=cpu_concurrency,
            use_s3_comm=use_s3_comm,
            s3_comm_config=s3_comm_config,
        )

    def run(
        self,
        client_fn: ClientFn,
        job_fn: JobFn,
        cid: str,
        context: Context,
    ) -> tuple[str, ClientRes, Context]:
        """Run a client run."""
        # Execute tasks and return result
        # return also cid which is needed to ensure results
        # from the pool are correctly assigned to each ClientProxy
        try:
            # Instantiate client (check 'Client' type is returned)
            client = check_clientfn_returns_client(client_fn(cid))
            # Inject context
            client.set_context((context, self.node_manager))  # type: ignore[reportArgumentType,arg-type]
            # Run client job
            job_results = job_fn(client)
            # Retrieve context (potentially updated)
            updated_context, _ = cast(tuple[Context, NodeManager], client.get_context())  # type: ignore[reportGeneralTypeIssues]
            client.set_context(updated_context)
        except Exception as ex:
            client_trace = traceback.format_exc()
            message = (
                "\n\tSomething went wrong when running your client run.\n\tClient "
                + cid
                + " crashed when the "
                + self.__class__.__name__
                + " was running its run.\n\tException triggered on the client side: "
                + client_trace,
            )
            raise ClientException(str(message)) from ex

        return cid, job_results, updated_context
