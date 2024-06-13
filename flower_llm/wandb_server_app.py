# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Flower WandbServerApp."""

from collections.abc import Callable
from typing import Any
from flwr.common import Context, RecordSet
from flwr.common.logger import warn_preview_feature
from flwr.server import ServerApp
from flwr.server.strategy import Strategy

from flwr.server.client_manager import ClientManager
from flwr.server.compat import start_driver
from flwr.server.driver import Driver
from flwr.server.server import Server
from flwr.server.server_config import ServerConfig
from flwr.server.typing import ServerAppCallable

import wandb

from flower_llm.conf.base_schema import WandbSetup
from flower_llm.utils import wandb_init


class WandbServerApp(ServerApp):
    """Flower WandbServerApp.

    Examples
    --------
    Use the `WandbServerApp` with a `PollenServer`:

    >>> config = ServerConfig(num_rounds=3)
    >>> server = PollenServer(...)
    >>> wandb_kwargs: WandbSetup = ...
    >>>
    >>> app = WandbServerApp(
    >>>     config=config,
    >>>     server=server,
    >>>     wandb_kwargs=wandb_kwargs,
    >>> )

    Use the `WandbServerApp` with a custom main function:

    >>> app = WandbServerApp()
    >>>
    >>> @app.main()
    >>> def main(driver: Driver, context: Context) -> None:
    >>>    print("WandbServerApp running")
    """

    def __init__(
        self,
        server: Server | None = None,
        config: ServerConfig | None = None,
        strategy: Strategy | None = None,
        client_manager: ClientManager | None = None,
        wandb_kwargs: WandbSetup | None = None,
        wandb_config: dict[Any, Any] | None = None,
    ) -> None:
        self._server = server
        self._config = config
        self._strategy = strategy
        self._client_manager = client_manager
        self._wandb_kwargs = wandb_kwargs
        self._wandb_config = wandb_config
        self._main: ServerAppCallable | None = None

    def __call__(self, driver: Driver, context: Context) -> None:
        """Execute `WandbServerApp`."""
        # NOTE: We need a dict type for the wandb_kwargs even though we are not using it
        passed_wandb_kwargs = self._wandb_kwargs or {}  # type: ignore[var-annotated]
        with wandb_init(  # type: ignore[union-attr,misc]
            self._wandb_kwargs is not None,
            **passed_wandb_kwargs,  # type: ignore[reportCallIssue]
            config=self._wandb_config or {},
            settings=wandb.Settings(start_method="thread"),  # type: ignore[arg-type]
        ) as _:
            # Compatibility mode
            if not self._main:
                start_driver(
                    server=self._server,
                    config=self._config,
                    strategy=self._strategy,
                    client_manager=self._client_manager,
                    driver=driver,
                )
                return

            # New execution mode
            context = Context(state=RecordSet())
            self._main(driver, context)

    def main(self) -> Callable[[ServerAppCallable], ServerAppCallable]:
        """Return a decorator that registers the main fn with the server app.

        Examples
        --------
        >>> app = WandbServerApp()
        >>>
        >>> @app.main()
        >>> def main(driver: Driver, context: Context) -> None:
        >>>    print("WandbServerApp running")
        """

        def main_decorator(main_fn: ServerAppCallable) -> ServerAppCallable:
            """Register the main fn with the WandbServerApp object."""
            if self._server or self._config or self._strategy or self._client_manager:
                raise ValueError(
                    """Use either a custom main function or a `PollenServer`, not both.

                    Use the `WandbServerApp` with a `PollenServer`:

                    >>> config = ServerConfig(num_rounds=3)
                    >>> server = PollenServer(...)
                    >>> wandb_kwargs: WandbSetup = ...
                    >>>
                    >>> app = WandbServerApp(
                    >>>     config=config,
                    >>>     server=server,
                    >>>     wandb_kwargs=wandb_kwargs,
                    >>> )

                    Use the `WandbServerApp` with a custom main function:

                    >>> app = WandbServerApp()
                    >>>
                    >>> @app.main()
                    >>> def main(driver: Driver, context: Context) -> None:
                    >>>    print("WandbServerApp running")""",
                )

            warn_preview_feature("WandbServerApp-register-main-function")

            # Register provided function with the WandbServerApp object
            self._main = main_fn

            # Return provided function unmodified
            return main_fn

        return main_decorator


class LoadWandbServerAppError(Exception):
    """Error when trying to load `WandbServerApp`."""
