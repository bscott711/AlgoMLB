from algomlb.core.logger import logger
from algomlb.core.agent_io import AgentResult, emit_agent_result
import io
import contextlib


def test_logger_initialization():
    logger.info("Core logger test")
    assert logger is not None


def test_agent_result():
    res = AgentResult(status="success", command="test")
    assert res.status == "success"
    assert res.duration_ms == 0


def test_emit_agent_result():
    res = AgentResult(status="success", command="test")
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        emit_agent_result(res)
    output = f.getvalue()
    assert '"status":"success"' in output
