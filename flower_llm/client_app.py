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
    EvaluateRes
)
from flwr.common.recordset_compat import recordset_to_fitins, fitres_to_recordset, evaluateres_to_recordset, recordset_to_evaluateins
from flwr.common.logger import log, update_console_handler
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


def hello_world_mod(msg, ctx, call_next) -> Message:
    print("Hello, ...[pause for dramatic effect]...")
    out = call_next(msg, ctx)
    print("...[pause was long enough]... World!")
    return out


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
app = NodeManagerApp(
    mods=[
        hello_world_mod,
    ],
)


def set_parameters(msg: Message, ctx: Context) -> Message:
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
    round_parameters, round_parameters_sh = get_parameters_shm(
        parameters_metadata=app.parameters_metadata,
        create=True,
        name=app.node_manager_uuid + POLLEN_PARAMETERS_SHM,
    )
    # Get the parameters from the recordset
    parameters = parametersrecord_to_parameters(
        msg.content.parameters_records["broadcastins.parameters"], keep_input=False
    )
    # Set the parameters in the shared memory
    set_parameters_shm(round_parameters, parameters_to_ndarrays(parameters))
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
    loss, num_examples, metrics = app.eval(
        configs=msg.content.configs_records
    )
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
    # eval. It will dispatch to the appropriate method based on the contents of the message.
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
    raise ValueError(f"Unknown query_type: {query_type}. Message content: {content}")
