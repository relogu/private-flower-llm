"""TODO."""

from logging import DEBUG
import warnings
from flwr.common import (
    Message,
    Context,
    RecordSet,
    ConfigsRecord,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
    FitRes,
    Code,
    Status,
    EvaluateRes,
)
from flwr.common.recordset_compat import (
    recordset_to_fitins,
    fitres_to_recordset,
    evaluateres_to_recordset,
    recordset_to_evaluateins,
)
from flwr.common.logger import update_console_handler
from flwr.common.recordset_compat import parametersrecord_to_parameters

from flower_llm.node_manager.node_manager_app import NodeManagerApp
from flower_llm.node_manager.utils import (
    POLLEN_PARAMETERS_SHM,
    get_parameters_shm,
    set_parameters_shm,
)
from flower_llm.server.s3_utils import (
    replace_parameters_in_recordset_with_remote,
    replace_remote_with_parameters_in_recordset,
)

# Fix the logger
update_console_handler(level=DEBUG, colored=False, timestamps=True)

# Filter user warning from configuration of MPT
warnings.filterwarnings(
    action="ignore",
    category=UserWarning,
    message=("If not using a Prefix Language Model*"),
    append=True,
)
# TODO: These don't work -- not sure why
# Filter deprecation warning from pkg_resources
warnings.filterwarnings(
    action="ignore",
    category=DeprecationWarning,
    message=("Deprecated call to *"),
    append=True,
)
warnings.filterwarnings(
    action="ignore",
    category=DeprecationWarning,
    message=("pkg_resources is deprecated*"),
    append=True,
)

# Flower ClientApp
app = NodeManagerApp()


def set_parameters(msg: Message, ctx: Context) -> Message:
    """Set the parameters received into shared memory for the client application.

    This function is responsible for handling a message that contains new parameters for
    the client application, typically as part of a federated learning cycle. It performs
    several key operations: downloading parameters from S3 if required, closing and
    unlinking any existing shared memory for parameters, creating new shared memory for
    the received parameters, and setting these parameters in the shared memory. Finally,
    it prepares a reply message indicating the successful setting of parameters.

    Parameters
    ----------
    msg : Message
        The incoming message containing the new parameters to be set. This message may
        include instructions to download the parameters from S3.
    ctx : Context
        The context in which the function operates, providing access to the
        application's state and configuration.

    Returns
    -------
    Message
        A reply message indicating the status of setting the parameters.

    Notes
    -----
    - The function first checks if the parameters need to be downloaded from S3, based
        on the application's configuration and the incoming message. If so, it downloads
        the parameters using the application's remote uploader/downloader.
    - It then attempts to close and unlink any existing shared memory for parameters. If
        no such shared memory exists (indicated by a FileNotFoundError), it simply
        proceeds.
    - New shared memory for the round's parameters is created, and the parameters from
        the message are set into this shared memory.
    - The parameters are converted from their record format in the message to the
        application's internal parameter format before being set in shared memory.
    - A reply message is prepared with a basic configuration, indicating the completion
        of the parameter setting process.
    - This function is part of a client application in a federated learning system,
        assuming the existence of `Message`, `Context`, `RecordSet`, `Configs`,
        `replace_parameters_in_recordset_with_remote`, `get_parameters_shm`,
        `parametersrecord_to_parameters`, `set_parameters_shm`, and related utilities
        and configurations.
    """
    # Download from the S3 if asked to
    msg = replace_parameters_in_recordset_with_remote(
        remote_uploader_downloader=app.remote_up_down,
        incoming_message=msg,
        use_s3_comm=app.cfg.use_s3_comm,
        msg_str="broadcastins",
    )
    # Close the shared memory, if exists
    try:
        parameters, parameters_sh = get_parameters_shm(
            parameters_metadata=app.parameters_metadata,
            create=False,
            name=app.node_manager_uuid + POLLEN_PARAMETERS_SHM,
        )
        parameters_sh.close()
        parameters_sh.unlink()
    except FileNotFoundError:
        pass
    # Create the parameters shared memory
    round_parameters, _round_parameters_sh = get_parameters_shm(
        parameters_metadata=app.parameters_metadata,
        create=True,
        name=app.node_manager_uuid + POLLEN_PARAMETERS_SHM,
    )
    # Get the parameters from the recordset
    parameters = parameters_to_ndarrays(
        parametersrecord_to_parameters(
            msg.content.parameters_records["broadcastins.parameters"], keep_input=False
        )
    )
    # Set the parameters in the shared memory
    set_parameters_shm(round_parameters, parameters)
    # Prepare reply message
    recordset = RecordSet()
    recordset.configs_records["broadcast"] = ConfigsRecord({"status": "OK"})
    return msg.create_reply(content=recordset)


def get_properties(msg: Message, ctx: Context) -> Message:
    msg.content = RecordSet(
        configs_records={"get_properties": ConfigsRecord(app.properties)}
    )
    return msg.create_reply(content=msg.content)


def free_resources(msg: Message, ctx: Context) -> Message:
    return msg.create_reply(content=msg.content)


# When you decorate a function with, for example, @app.train(), the train method of the
# app instance is called with the decorated function as an argument.
@app.train()
def train(msg: Message, ctx: Context) -> Message:
    """Handle a training request, performs training, and returns the training results.

    This function processes a training request encapsulated in a `Message` object. It
    extracts the training instructions, performs the training using the application's
    training mechanism, and compiles the results into a response message. The function
    supports dynamic management of worker processes based on the server round,
    restarting all workers at intervals defined by `app.refresh_period` to ensure fresh
    application state.

    Parameters
    ----------
    msg : Message
        The incoming message containing training instructions.
    ctx : Context
        The context in which the training is being performed, providing access to
        application state and utilities.

    Returns
    -------
    Message
        A reply message containing the training results, including the trained
        parameters, metrics, and the number of examples trained on.

    Raises
    ------
    AssertionError
        If the `server_round` is missing from the training configuration.

    Notes
    -----
    - The function first converts the incoming message's content into training
        instructions (`FitIns`).
    - It checks if the current server round is a multiple of `app.refresh_period`. If
        so, it restarts all worker processes to refresh the application state.
    - The training is then performed, and the results are compiled into a `FitRes`
        object, which includes the training status, trained parameters, metrics, and
        the number of examples trained on.
    - Finally, the `FitRes` object is translated back into a `RecordSet` for the reply
        message, with additional S3 communication configuration added to the
        `configs_records`.
    - This function is part of a client application in a federated learning system,
        assuming the existence of `Message`, `Context`, `FitIns`, `FitRes`, `Status`,
        `Code`, `ConfigsRecord`, and `ndarrays_to_parameters` functions or classes.
    """
    msg_str = "fitres"
    fitins = recordset_to_fitins(msg.content, False)
    config = fitins.config
    assert "server_round" in config, "Server round must be in the config"
    # Restart all the worker every `app.refresh_period` rounds
    if config["server_round"] % app.refresh_period == 0:
        # Close and remove the workers
        app._close_workers()
        # Re-create and start the workers
        app._create_and_start_workers()
    # Launch the actual training
    trained_parameters, num_examples, metrics = app.fit(
        configs=msg.content.configs_records
    )
    # Compile FitRes
    status = Status(code=Code.OK, message="chiappe sode")
    fitres = FitRes(
        status=status,
        parameters=ndarrays_to_parameters(trained_parameters),
        metrics=metrics,
        num_examples=num_examples,
    )
    # Translate FitRes to RecordSet
    recordset = fitres_to_recordset(fitres, keep_input=False)
    recordset.configs_records[f"{msg_str}.s3_comm_config"] = ConfigsRecord(
        {
            "endpoint_id": app.node_manager_uuid,
            "file_name": "parameters",
            "current_round": str(config["server_round"]),
        }
    )
    msg_str = "fitres"
    return replace_remote_with_parameters_in_recordset(
        remote_uploader_downloader=app.remote_up_down,
        outgoing_message=msg.create_reply(recordset),
        use_s3_comm=app.cfg.use_s3_comm,
        msg_str=msg_str,
    )


@app.evaluate()
def evaluate(msg: Message, ctx: Context) -> Message:
    """Process an evaluation request, performs evaluation, and returns the results.

    This function handles an evaluation request encapsulated in a `Message` object. It
    extracts evaluation instructions, performs the evaluation using the application's
    evaluation mechanism, and compiles the results into a response message. The function
    also supports the dynamic management of worker processes based on the server round,
    restarting all workers at intervals defined by `app.refresh_period`.

    Parameters
    ----------
    msg : Message
        The incoming message containing evaluation instructions.
    ctx : Context
        The context in which the evaluation is being performed, providing access to
        application state and utilities.

    Returns
    -------
    Message
        A reply message containing the evaluation results, including loss, metrics, and
        the number of examples evaluated.

    Raises
    ------
    AssertionError
        If the `server_round` is missing from the evaluation configuration.

    Notes
    -----
    - The function first converts the incoming message's content into evaluation
        instructions (`EvaluateIns`).
    - It checks if the current server round is a multiple of `app.refresh_period`. If
        so, it restarts all worker processes to refresh the application state.
    - The evaluation is then performed, and the results are compiled into an
        `EvaluateRes` object, which includes the evaluation status, loss, metrics, and
        the number of examples evaluated.
    - Finally, the `EvaluateRes` object is translated back into a `RecordSet` for the
        reply message, with additional S3 communication configuration added to the
        `configs_records`.
    - This function is part of a server application in a federated learning system,
        assuming the existence of `Message`, `Context`, `EvaluateIns`, `EvaluateRes`,
        `Status`, `Code`, and `ConfigsRecord` classes or interfaces.
    """
    msg_str = "evaluateres"
    evaluateins = recordset_to_evaluateins(msg.content, False)
    config = evaluateins.config
    assert "server_round" in config, "Server round must be in the config"
    # Restart all the worker every `app.refresh_period` rounds
    if config["server_round"] % app.refresh_period == 0:
        # Close and remove the workers
        app._close_workers()
        # Re-create and start the workers
        app._create_and_start_workers()
    # Launch the actual training
    loss, num_examples, metrics = app.eval(configs=msg.content.configs_records)
    # Compile EvaluateRes
    status = Status(code=Code.OK, message="chiappe sode")
    evaluateres = EvaluateRes(
        status=status,
        loss=loss,
        metrics=metrics,
        num_examples=num_examples,
    )
    # Translate EvaluateRes to RecordSet
    recordset = evaluateres_to_recordset(evaluateres)
    recordset.configs_records[f"{msg_str}.s3_comm_config"] = ConfigsRecord(
        {
            "endpoint_id": app.node_manager_uuid,
            "file_name": "parameters",
            "current_round": str(config["server_round"]),
        }
    )
    return msg.create_reply(recordset)


@app.query()
def query(msg: Message, ctx: Context) -> Message:
    # This method serves as dispatcher to perform those tasks that are not train or
    # eval. It will dispatch to the appropriate method based on the contents of the
    # message.
    # Extract the RecordSet for this type of message
    content = msg.content
    assert "query" in content.configs_records, "Query message must contain 'query' key"
    query_type = content.configs_records["query"]["type"]
    match query_type:
        case "broadcast_parameters":
            # The server has sent the new parameters to the client app
            return set_parameters(msg=msg, ctx=ctx)
        case "get_properties":
            return get_properties(msg=msg, ctx=ctx)
        case "free_resources":
            return free_resources(msg=msg, ctx=ctx)
    raise ValueError(f"Unknown query_type: {query_type!s}.")
