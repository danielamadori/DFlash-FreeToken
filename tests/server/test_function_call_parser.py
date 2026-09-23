from __future__ import annotations

import pathlib
import json

import pytest

from freetoken.server.function_call_parser import FunctionCallParser, SUPPORTED_TOOL_CALL_PARSERS


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }
]

OPENCODE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "parameters": {
                "type": "object",
                "properties": {"filePath": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                },
            },
        },
    },
]


@pytest.mark.parametrize("parser_name", SUPPORTED_TOOL_CALL_PARSERS)
def test_supported_parser_names_instantiate(parser_name):
    parser = FunctionCallParser(TOOLS, tool_call_parser=parser_name)

    result = parser.parse_non_stream("plain text")

    assert result.normal_text == "plain text"
    assert result.calls == []


def test_mistral_parser_accepts_tool_calls_tag():
    parser = FunctionCallParser(TOOLS, tool_call_parser="mistral")

    result = parser.parse_non_stream(
        '[TOOL_CALLS] [{"name":"get_weather","arguments":{"city":"Paris"}}]'
    )

    assert result.normal_text == ""
    assert len(result.calls) == 1
    assert result.calls[0].name == "get_weather"
    assert json.loads(result.calls[0].parameters) == {"city": "Paris"}


@pytest.mark.parametrize(
    ("parser_name", "text"),
    [
        (
            "mistral",
            '[TOOL_CALLS] [{"name":"get_weather","arguments":{"city":"Paris"}}]',
        ),
        (
            "qwen25",
            '<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>',
        ),
        (
            "llama3",
            '<|python_tag|>{"name":"get_weather","arguments":{"city":"Paris"}}',
        ),
    ],
)
def test_base_json_parsers_use_declared_tool_index(parser_name, text):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "other_tool",
                "parameters": {"type": "object"},
            },
        },
        TOOLS[0],
    ]
    parser = FunctionCallParser(tools, tool_call_parser=parser_name)

    result = parser.parse_non_stream(text)

    assert len(result.calls) == 1
    assert result.calls[0].name == "get_weather"
    assert result.calls[0].tool_index == 1


def test_base_json_parsers_forward_unknown_tools_by_default():
    parser = FunctionCallParser(TOOLS, tool_call_parser="mistral")

    result = parser.parse_non_stream(
        '[TOOL_CALLS] [{"name":"not_declared","arguments":{"city":"Paris"}}]'
    )

    assert len(result.calls) == 1
    assert result.calls[0].tool_index == -1
    assert result.calls[0].name == "not_declared"
    assert json.loads(result.calls[0].parameters) == {"city": "Paris"}


@pytest.mark.parametrize(
    "text",
    [
        '<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>',
    ],
)
def test_parser_accepts_common_tagged_tool_call_shapes(text):
    parser = FunctionCallParser(TOOLS, tool_call_parser="qwen25")

    result = parser.parse_non_stream(text)

    assert len(result.calls) == 1
    assert result.calls[0].name == "get_weather"
    assert json.loads(result.calls[0].parameters) == {"city": "Paris"}


def test_gemma4_parser_accepts_compact_tool_call_shape():
    parser = FunctionCallParser(TOOLS, tool_call_parser="gemma4")

    result = parser.parse_non_stream('<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>')

    assert len(result.calls) == 1
    assert result.calls[0].name == "get_weather"
    assert json.loads(result.calls[0].parameters) == {"city": "Paris"}


def test_gemma4_parser_accepts_namespaced_tool_name():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "superpowers:using_superpowers",
                "parameters": {"type": "object"},
            },
        }
    ]
    parser = FunctionCallParser(tools, tool_call_parser="gemma4")

    result = parser.parse_non_stream(
        "<|tool_call>call:superpowers:using_superpowers{}<tool_call|>"
    )

    assert result.normal_text == ""
    assert len(result.calls) == 1
    assert result.calls[0].name == "superpowers:using_superpowers"
    assert json.loads(result.calls[0].parameters) == {}


def test_gemma4_parser_forwards_namespaced_skill_without_declared_tool():
    parser = FunctionCallParser(TOOLS, tool_call_parser="gemma4")

    result = parser.parse_non_stream(
        "<|tool_call>call:superpowers:using_superpowers{}<tool_call|>"
    )

    assert result.normal_text == ""
    assert len(result.calls) == 1
    assert result.calls[0].name == "superpowers:using_superpowers"
    assert json.loads(result.calls[0].parameters) == {}


def test_gpt_oss_parser_accepts_namespaced_tool_name():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "superpowers:using_superpowers",
                "parameters": {"type": "object"},
            },
        }
    ]
    parser = FunctionCallParser(tools, tool_call_parser="gpt_oss")

    result = parser.parse_non_stream(
        "<|start|>assistant<|channel|>commentary "
        "to=functions.superpowers:using_superpowers <|constrain|>json<|message|>{}<|end|>"
    )

    assert result.normal_text == ""
    assert len(result.calls) == 1
    assert result.calls[0].name == "superpowers:using_superpowers"
    assert json.loads(result.calls[0].parameters) == {}


@pytest.mark.parametrize(
    ("parser_name", "text", "expected_name", "expected_args"),
    [
        (
            "qwen3_coder",
            "<tool_call><function=read><parameter=filePath>/tmp/test_calc.py</parameter></function></tool_call>",
            "read",
            {"filePath": "/tmp/test_calc.py"},
        ),
        (
            "glm47",
            "<tool_call>read<arg_key>filePath</arg_key><arg_value>/tmp/test_calc.py</arg_value></tool_call>",
            "read",
            {"filePath": "/tmp/test_calc.py"},
        ),
        (
            "minimax",
            "<minimax:tool_call><invoke name=\"read\"><parameter name=\"filePath\">"
            "/tmp/test_calc.py</parameter></invoke></minimax:tool_call>",
            "read",
            {"filePath": "/tmp/test_calc.py"},
        ),
        (
            "gpt_oss",
            "<|channel|>analysis<|message|>Need files.<|end|><|start|>assistant"
            "<|channel|>commentary to=functions.glob <|constrain|>json<|message|>"
            "{\"pattern\":\"**/*.py\",\"path\":\"/tmp/ws\"}",
            "glob",
            {"pattern": "**/*.py", "path": "/tmp/ws"},
        ),
        (
            "deepseekv32",
            "<｜DSML｜function_calls><｜DSML｜invoke name=\"read\">"
            "<｜DSML｜parameter name=\"filePath\" string=\"true\">/tmp/test_calc.py</｜DSML｜parameter>"
            "</｜DSML｜invoke></｜DSML｜function_calls>",
            "read",
            {"filePath": "/tmp/test_calc.py"},
        ),
        (
            "deepseekv32",
            "<｜DSML｜tool_calls><｜DSML｜invoke name=\"read\">"
            "<｜DSML｜parameter name=\"filePath\">/tmp/test_calc.py</｜DSML｜parameter>"
            "</｜DSML｜invoke></｜DSML｜tool_calls>",
            "read",
            {"filePath": "/tmp/test_calc.py"},
        ),
    ],
)
def test_parser_accepts_family_specific_tool_call_shapes(
    parser_name, text, expected_name, expected_args
):
    parser = FunctionCallParser(OPENCODE_TOOLS, tool_call_parser=parser_name)

    result = parser.parse_non_stream(text)

    assert result.normal_text in ("", "Need files.")
    assert len(result.calls) == 1
    assert result.calls[0].name == expected_name
    assert json.loads(result.calls[0].parameters) == expected_args


# --------------------------------------------------------------------------- #
# Streaming incremental parsing (parse_stream_chunk)
# --------------------------------------------------------------------------- #
def _feed(parser, chunks):
    """Feed chunks through the streaming parser; return (per-chunk normal texts, calls)."""
    texts, calls = [], []
    for chunk in chunks:
        normal, chunk_calls = parser.parse_stream_chunk(chunk)
        texts.append(normal)
        calls.extend(chunk_calls)
    return texts, calls


@pytest.mark.parametrize("parser_name", ["qwen25", "glm47", "gemma4", "minimax", "deepseekv32", "qwen3_coder"])
def test_streaming_plain_text_releases_per_chunk(parser_name):
    # A pure-text response must stream out chunk by chunk, not buffer to the end.
    parser = FunctionCallParser(TOOLS, tool_call_parser=parser_name)
    texts, calls = _feed(parser, ["Hello ", "world."])
    assert texts == ["Hello ", "world."]
    assert calls == []
    assert parser.finish_stream() == ""


def test_streaming_partial_tag_holdback_then_release():
    # Text held back as a suspected tag prefix must be released once disambiguated.
    parser = FunctionCallParser(TOOLS, tool_call_parser="qwen25")
    texts, _ = _feed(parser, ["Hi <", "there"])
    assert "".join(texts) + parser.finish_stream() == "Hi <there"


def test_dsv32_streaming_partial_prefix_not_dropped():
    # Regression: the non-tool branch used to clear the whole buffer but return only
    # the newest chunk, dropping text held back as a suspected partial bot_token.
    parser = FunctionCallParser(TOOLS, tool_call_parser="deepseekv32")
    texts, calls = _feed(parser, ["a<", "b"])
    assert "".join(texts) + parser.finish_stream() == "a<b"
    assert calls == []


def test_streaming_finish_stream_releases_held_tail():
    parser = FunctionCallParser(TOOLS, tool_call_parser="qwen25")
    texts, calls = _feed(parser, ["text <tool"])
    assert calls == []
    assert "".join(texts) + parser.finish_stream() == "text <tool"


def test_dsv32_streaming_multi_param_args_prefix_stable():
    # The DSML streaming state machine emits prefix-stable fragments (vLLM-style):
    # '{"key":"' at parameter open, escaped value chars, '"' at close, '}' at
    # invoke close — concatenation IS the final arguments JSON.
    parser = FunctionCallParser(OPENCODE_TOOLS, tool_call_parser="deepseekv32")
    assert parser.args_fragments_prefix_stable() is True
    block = (
        "<｜DSML｜function_calls>\n"
        '<｜DSML｜invoke name="glob">\n'
        '<｜DSML｜parameter name="pattern" string="true">*.py</｜DSML｜parameter>\n'
        '<｜DSML｜parameter name="path" string="true">/src</｜DSML｜parameter>\n'
        "</｜DSML｜invoke>\n"
        "</｜DSML｜function_calls>"
    )
    chunks = [block[i : i + 7] for i in range(0, len(block), 7)]
    _, calls = _feed(parser, chunks)
    named = [c for c in calls if c.name]
    assert len(named) == 1 and named[0].name == "glob"
    joined = "".join(c.parameters for c in calls if c.name is None)
    assert json.loads(joined) == {"pattern": "*.py", "path": "/src"}
    # multiple argument fragments streamed mid-call, not one blob at close
    assert sum(1 for c in calls if c.name is None) >= 4
    # detector parse state agrees (used as the truncation fallback)
    assert json.loads(parser.unstreamed_arguments(named[0].tool_index)) == {
        "pattern": "*.py",
        "path": "/src",
    }


def test_streaming_support_flags():
    # Every registered detector is incremental-safe (buffered fallback remains as
    # the escape hatch for future formats, exercised via monkeypatch in
    # test_streaming_model_matrix.py::test_non_streaming_detector_falls_back_to_buffered_parse).
    for name in SUPPORTED_TOOL_CALL_PARSERS:
        assert FunctionCallParser(TOOLS, tool_call_parser=name).supports_streaming() is True


def _infer_from(module) -> "callable":
    """The inference function out of an args module, without importing the CLI.

    ``_infer_tool_call_parser`` is defined in the class body, so it is not an
    attribute of ServerArgs and cannot be called directly; the argument parser it
    belongs to wants a full command line. Lifting the source is ugly and honest:
    the alternative is not testing the rule that decides which dialect a model
    speaks.
    """
    import io
    import textwrap

    righe = io.open(module.__file__, encoding="utf-8").read().split("\n")
    inizio = next(
        i for i, r in enumerate(righe) if r.strip().startswith("def _infer_tool_call_parser")
    )
    fine = inizio + 1
    while fine < len(righe) and (righe[fine].strip() == "" or righe[fine].startswith("        ")):
        fine += 1
    spazio: dict = {}
    exec(textwrap.dedent("\n".join(righe[inizio:fine])), spazio)  # noqa: S102
    return spazio["_infer_tool_call_parser"]


#: I moduli che definiscono la regola del dialetto. Ne resta UNO, ed e' quello
#: che il motore importa davvero; `test_la_regola_vive_in_un_modulo_solo` fa
#: fallire la suite se ne ricompare un secondo. La parametrizzazione resta
#: perche' l'elenco e' l'unico posto da cambiare se un giorno tornassero a
#: essere due -- e allora quel test lo dira' per primo.
_MODULI_ARGS = ["freetoken.server.args"]

@pytest.mark.parametrize("modulo", _MODULI_ARGS)
def test_the_whole_qwen3_8_family_gets_the_xml_dialect(modulo: str, senza_rete):
    """Qwen3.8-27B emits the qwen3_coder XML, and only "-Flash" was enumerated.

    Measured through the cluster gateway: the 27B answered a tools request with

        <tool_call><function=read><parameter=filePath>segreto.txt</parameter>
        </function></tool_call>

    -- a correct call -- and the response carried no tool_calls at all. The name
    fell past the "-Flash" test into the generic "qwen" branch and got qwen25,
    which looks for JSON inside the tags, finds none, logs "Failed to parse JSON
    part", and drops it. The caller is then handed a tool call as prose.

    Both args modules carry a copy of the rule, and a fix in one of them is a
    node that behaves differently depending on how it was started.
    """
    import importlib

    dedurre = _infer_from(importlib.import_module(modulo))

    assert dedurre("Qwen3.8-27B") == "qwen3_coder"
    assert dedurre("Qwen3.8-Flash") == "qwen3_coder"
    # The families on either side must not move.
    assert dedurre("Qwen2.5-Coder-1.5B-Instruct") == "qwen25"
    assert dedurre("Qwen2-7B") == "qwen25"


def test_the_xml_dialect_needs_its_own_detector():
    """The same text to both detectors: qwen25 yields nothing, and says so only
    in a log line nobody reads on the caller's side."""
    uscita = (
        "<tool_call>\n<function=read>\n<parameter=filePath>\nsegreto.txt\n"
        "</parameter>\n</function>\n</tool_call>"
    )
    giusto = FunctionCallParser(TOOLS, tool_call_parser="qwen3_coder").parse_non_stream(uscita)
    assert [c.name for c in giusto.calls] == ["read"]
    assert json.loads(giusto.calls[0].parameters) == {"filePath": "segreto.txt"}

    sbagliato = FunctionCallParser(TOOLS, tool_call_parser="qwen25").parse_non_stream(uscita)
    assert sbagliato.calls == []


@pytest.fixture
def senza_rete(monkeypatch):
    """No hub lookup while the substring chain is under test.

    ``_infer_tool_call_parser`` asks ``cached_load_hf_config`` first, and for a name that is
    not a local folder that goes to huggingface.co. Measured from this suite: a HEAD for a
    made-up name came back 429 and the client waited 61 s, five retries deep -- one test name
    cost more than the whole file. The rule being pinned here is the one built from the
    marker, so the lookup is made to fail the way it fails for an unreachable hub.
    """

    def _niente_rete(percorso):
        raise FileNotFoundError(f"nessuna rete nei test: {percorso}")

    monkeypatch.setattr("freetoken.utils.cached_load_hf_config", _niente_rete)


def _architetture_servibili() -> list[str]:
    """The architecture keys of the model registry, read from its source.

    Importing ``freetoken.models.register`` pulls the model packages in and costs this
    suite minutes; the registry is a dict literal, so its keys are read off the syntax
    tree instead. The count is asserted so a registry that stops being a literal fails
    here loudly rather than silently testing nothing.
    """
    import ast
    import io
    import os

    import freetoken

    sorgente = os.path.join(os.path.dirname(freetoken.__file__), "models", "register.py")
    albero = ast.parse(io.open(sorgente, encoding="utf-8").read())
    for nodo in ast.walk(albero):
        bersaglio = getattr(nodo, "target", None)
        nome = getattr(bersaglio, "id", None)
        if nome == "_MODEL_REGISTRY" and isinstance(nodo.value, ast.Dict):
            chiavi = [k.value for k in nodo.value.keys if isinstance(k, ast.Constant)]
            assert len(chiavi) >= 30, f"registro letto male: {len(chiavi)} architetture"
            return chiavi
    raise AssertionError("_MODEL_REGISTRY non trovato in register.py")




@pytest.mark.parametrize("modulo", _MODULI_ARGS)
def test_every_servable_architecture_resolves_to_a_real_dialect(modulo: str, senza_rete):
    """Whatever this engine can serve, `--tool-call-parser auto` must name a dialect for it.

    The chain no longer ends with a catch-all, and that is right -- guessing hands the
    caller a tool call as prose with no error anywhere. But a refusal is only right for a
    model this engine cannot serve: refusing one that is in the registry turns a silent
    wrong dialect into a node that will not start at all, which is worse and was measured:
    with the catch-all removed and no `llama` branch, LlamaForCausalLM -- a family with its
    own loader, its own detector and its own `--tool-call-parser llama3` -- raised.

    So the invariant is the registry, not a list kept by hand: every architecture the model
    register can load must come out of the chain as a dialect the parser actually has.
    """
    import importlib

    dedurre = _infer_from(importlib.import_module(modulo))
    archi = _architetture_servibili()
    rifiutate: dict[str, str] = {}
    sconosciute: dict[str, str] = {}
    for arch in archi:
        try:
            dialetto = dedurre(arch)
        except Exception as exc:  # noqa: BLE001 -- the point is to report every one
            rifiutate[arch] = f"{type(exc).__name__}"
            continue
        if dialetto not in SUPPORTED_TOOL_CALL_PARSERS:
            sconosciute[arch] = dialetto
    assert not rifiutate, f"{modulo}: architetture servibili senza dialetto: {rifiutate}"
    assert not sconosciute, f"{modulo}: dialetti inesistenti: {sconosciute}"


@pytest.mark.parametrize("modulo", _MODULI_ARGS)
def test_the_family_is_read_from_the_architecture_not_from_the_file_name(modulo: str, senza_rete):
    """The GGUF header says `qwen35`, and that is what must decide.

    `general.architecture` reaches the marker through model_type and architectures, so the
    engine can know the family of a renamed file or a symlink. The chain checked qwen3_5 and
    qwen3.5 and not qwen35, which left the production Qwen3.8 GGUF depending on the string
    "Qwen3.8" being in the FILE NAME -- and the wrong dialect is silent.

    This line first went into `engine/args.py`, which nothing at runtime imported, so the
    node kept the old answer while the commit said otherwise. That copy is gone now --
    see `test_la_regola_vive_in_un_modulo_solo`, which keeps it gone.
    """
    import importlib

    dedurre = _infer_from(importlib.import_module(modulo))

    assert dedurre("Qwen35GGUFForCausalLM") == "qwen3_coder"
    assert dedurre("Qwen35MoeGGUFForCausalLM") == "qwen3_coder"
    # the family named by the architecture, with no help from the path
    assert dedurre("/srv/pesi/modello-anonimo.gguf qwen35 Qwen35GGUFForCausalLM") == "qwen3_coder"
    # and the family that lost its only mapping when the catch-all went away
    assert dedurre("LlamaForCausalLM") == "llama3"
    assert dedurre("/models/Meta-Llama-3.1-8B-Instruct") == "llama3"


@pytest.mark.parametrize("modulo", _MODULI_ARGS)
def test_an_unservable_model_is_refused_by_name(modulo: str, senza_rete):
    """No catch-all: a model the chain does not know stops startup and names itself."""
    import importlib

    dedurre = _infer_from(importlib.import_module(modulo))

    with pytest.raises(ValueError, match="tool-call-parser"):
        dedurre("/models/Phi-4-mini-instruct")


def test_la_regola_vive_in_un_modulo_solo():
    """One rule, one module -- and it must be the module the engine imports.

    Why this test exists. `_infer_tool_call_parser` lived in two files:
    `server/args.py`, which `server/launch.py` imports, and `engine/args.py`, which
    nothing at runtime imported. Three separate fixes were written into the dead copy.
    Each time the commit message was true and the engine's behaviour did not change --
    the third time it was the qwen35 branch, so the production GGUF kept picking its
    dialect from the FILE NAME while the fix sat in a module no process loaded.

    Deleting the copy fixes today. This test fixes it for good: it fails the moment a
    second definition appears anywhere under `python/freetoken`, whatever the file is
    called -- so the next person to copy the rule finds out from a red test instead of
    from a node that answers with the wrong dialect and no error.

    The third assertion is the one that makes the check mean something: the surviving
    definition must sit next to the `launch.py` that loads it. One copy in the WRONG
    module would satisfy the count and still be dead code.
    """
    radice = pathlib.Path(__file__).resolve().parents[2] / "python" / "freetoken"
    definizioni = sorted(
        percorso.relative_to(radice).as_posix()
        for percorso in radice.rglob("*.py")
        if "def _infer_tool_call_parser" in percorso.read_text(encoding="utf-8", errors="replace")
    )
    assert definizioni, "la regola del dialetto e' sparita del tutto"
    assert len(definizioni) == 1, (
        f"la regola e' duplicata in {definizioni}: una copia prendera' le correzioni e "
        f"l'altra restera' quella viva, come e' gia' successo tre volte"
    )
    assert definizioni == ["server/args.py"], (
        f"la regola sta in {definizioni[0]}, ma il motore importa `server/args.py` "
        f"(`server/launch.py`, `from .args import parse_args`): li' dentro e' codice morto"
    )
    launch = (radice / "server" / "launch.py").read_text(encoding="utf-8")
    assert "from .args import parse_args" in launch, (
        "`launch.py` non importa piu' `server/args.py`: questo test stava verificando "
        "un legame che non esiste piu', e va rifatto sul modulo che importa adesso"
    )
