"""TODO."""

from logging import DEBUG
import warnings
from flwr.common import (
    Message,
    Context,
    RecordSet,
    ConfigsRecord,
    parameters_to_ndarrays,
)
from flwr.common.logger import log, update_console_handler
from flwr.common.recordset_compat import parametersrecord_to_parameters

from flower_llm.node_manager.node_manager_app import NodeManagerApp
from flower_llm.node_manager.utils import (
    POLLEN_PARAMETERS_SHM,
    get_parameters_shm,
    set_parameters_shm,
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
    # Get NodeManager UUID
    assert (
        "node_manager" in ctx.state.configs_records
    ), "NodeManager key must be in the context"
    node_manager_dict = ctx.state.configs_records["node_manager"]
    assert "uuid" in node_manager_dict, "UUID key must be in the node_manager_dict"
    node_manager_uuid = node_manager_dict["uuid"]
    assert type(node_manager_uuid) is str, "UUID must be of type str"
    # Create the parameters shared memory
    round_parameters, round_parameters_sh = get_parameters_shm(
        parameters_metadata=app.parameters_metadata,
        create=True,
        name=node_manager_uuid + POLLEN_PARAMETERS_SHM,
    )
    parameters = parametersrecord_to_parameters(
        msg.content.parameters_records["broadcastins.parameters"], keep_input=False
    )
    set_parameters_shm(round_parameters, parameters_to_ndarrays(parameters))
    recordset = RecordSet()
    recordset.configs_records["broadcast"] = ConfigsRecord({"status": "OK"})
    return msg.create_reply(content=recordset)


def get_properties(msg: Message, ctx: Context) -> Message:
    msg.content = RecordSet(configs_records={"": ConfigsRecord(app.properties)})
    return msg.create_reply(content=msg.content)


def free_resources(msg: Message, ctx: Context) -> Message:
    return msg.create_reply(content=msg.content)


# When you decorate a function with, for example, @app.train(), the train method of the
# app instance is called with the decorated function as an argument.
@app.train()
def train(msg: Message, ctx: Context) -> Message:
    log(DEBUG, "`train` is not implemented, echoing original message")
    return msg.create_reply(msg.content)


@app.evaluate()
def eval(msg: Message, ctx: Context) -> Message:
    log(DEBUG, "`evaluate` is not implemented, echoing original message")
    return msg.create_reply(msg.content)


@app.query()
def query(msg: Message, ctx: Context) -> Message:
    # This method serves as dispatcher to perform those tasks that are not train or
    # eval. It will dispatch to the appropriate method based on the contents of the message.
    # Extract the RecordSet for this type of message
    content = msg.content
    assert "query" in content.configs_records, "Query message must contain 'query' key"
    query_type = content.configs_records["query"]["type"]
    match query_type:
        case "ping":
            return msg.create_reply(content=msg.content)
        case "set_parameters":
            return set_parameters(msg=msg, ctx=ctx)
        case "get_properties":
            return get_properties(msg=msg, ctx=ctx)
        case "free_resources":
            return free_resources(msg=msg, ctx=ctx)
    raise ValueError(f"Unknown query_type: {query_type}")
