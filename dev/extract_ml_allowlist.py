#!/usr/bin/env python3
"""Derive Latu ML's attribute registry from the server's own allowlist.

A `Fetch` does not reach a model by reflection alone: `MLUtils.validate` checks the method
name against `ALLOWED_ATTRIBUTES`, a `Seq[(Class, Set[String])]` in one Scala file, and refuses
anything else with `CONNECT_ML.ATTRIBUTE_NOT_ALLOWED`. Nothing in the protocol describes that
list, so a client either transcribes it or reads it. This reads it.

    python dev/extract_ml_allowlist.py                # summarise
    python dev/extract_ml_allowlist.py --write        # rewrite priv/ml_attributes.exs
    python dev/extract_ml_allowlist.py --check        # exit 1 if that file is out of date

Three things come out, and they are three different kinds of claim:

  * **`allowed`** is the Scala list, transcribed by a parser rather than by hand: one entry per
    `(classOf[X], Set(...))` pair, with `X` resolved to a JVM class name. It is what the file
    says, and a re-vendor diffs it.
  * **`classes`** is what a *concrete* model or summary can be asked, which the file does not
    say: `validate` walks `cls.isInstance(obj)`, so an object's allowlist is the union of every
    entry whose class it inherits from. The hierarchy is resolved through PySpark's mirror of
    Spark's own — the class hierarchies match name for name, and `dev/probe_ml.exs` is what
    confirms it against a server rather than a mirror.
  * **`returns`** says which of four shapes an answer takes. `:frame` is a DataFrame, from
    PySpark's `try_remote_attribute_relation` decorator — the difference between a `Fetch`
    wrapped as a command and one wrapped as a relation. `:summary` is another cached object,
    which answers on `MlCommandResult`'s `summary` arm, and `:operators` is a *list* of them
    on the `operator_info` arm. `:value` is a literal. All four are read from PySpark's
    decorators and return annotations, not decided here.

The Scala file is fetched by tag from GitHub, so this needs a network; `--scala` takes a local
copy instead. Everything else comes from the same PySpark `dev/extract_ml_operators.py` reads.
"""

import argparse
import ast
import importlib
import inspect
import json
import os
import pathlib
import re
import sys
import textwrap
import typing
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from extract_ml_operators import (  # noqa: E402  - after the sys.path line above
    CONVERTERS,
    IDENTITY_TYPES,
    clean_doc,
)

os.environ["SPARK_CONNECT_MODE_ENABLED"] = "1"
os.environ.pop("SPARK_REMOTE", None)

SCALA_PATH = (
    "sql/connect/server/src/main/scala/org/apache/spark/sql/connect/ml/MLUtils.scala"
)
SCALA_URL = "https://raw.githubusercontent.com/apache/spark/v%s/" + SCALA_PATH

# The `pyspark.ml` modules a class can be found in, and the Scala package each mirrors.
# `base`, `wrapper` and `param` hold the classes Scala keeps in the root `org.apache.spark.ml`.
PACKAGES = {
    "base": "",
    "classification": "classification",
    "clustering": "clustering",
    "evaluation": "evaluation",
    "feature": "feature",
    "fpm": "fpm",
    "param": "",
    "recommendation": "recommendation",
    "regression": "regression",
    "tree": "tree",
    "tuning": "tuning",
    "util": "util",
    "wrapper": "",
}

JAVA_ROOT = "org.apache.spark.ml"

# Where the *concrete* models and summaries live. Narrower than PACKAGES, which also has to
# cover the traits the allowlist keys on: `base`, `wrapper`, `param`, `tree` and `util` hold
# PySpark's own plumbing (`JavaModel`, `JavaPredictionModel`) alongside those traits, and a
# module-per-class generated for either would be a module for something the server never
# returns.
OPERATOR_MODULES = [
    "classification",
    "clustering",
    "feature",
    "fpm",
    "recommendation",
    "regression",
]

# Concrete in Python, client-side in fact: `OneVsRestModel` is assembled by PySpark out of
# per-class models and has no `ObjectRef` of its own. `dev/extract_ml_operators.py` leaves it
# out for the same reason.
EXCLUDE_CLASSES = {"OneVsRestModel"}

# In the allowlist but not an attribute of any model: `ConnectHelper` is the server's own
# helper object, the receiver for `invoke_helper_attr` and `invoke_helper_relation`. It has no
# Python class and no `ObjectRef` — a `Fetch` names it by the literal id below — so it is not a
# row in `classes`. It gets a section of its own instead; see `helper_calls`.
EXCLUDE = {"ConnectHelper"}

# The one method on it that PySpark does not reach through `invoke_helper_*`, and so has no call
# site for this to read: `RemoteMLWriter.handleOverwrite` and `util._remove_dfs_dir` both build
# the wrapper by hand and call `_call_java(\"handleOverwrite\", path, shouldOverwrite)`.
#
# It is the meta-algorithms', because only they write a directory themselves — `MlCommand.Write`
# carries its own `should_overwrite` for everything else. So its two arguments are **pinned**
# here rather than read, with the call site they were read off, and the row it produces says
# `read_by_hand: true` so that nothing downstream mistakes it for something the extractor
# derived. A helper method appearing in the Scala with neither a call site nor an entry here
# stops the run rather than vanishing.
HELPERS_WITHOUT_CALL_SITE = {
    "handleOverwrite": {
        "why": "PySpark calls it through `_call_java`, not `invoke_helper_*`",
        "python": "RemoteMLWriter.handleOverwrite",
        "function": "handle_overwrite",
        "returns": "value",
        "args": [
            {"name": "path", "type": "string"},
            {"name": "should_overwrite", "type": "boolean"},
        ],
        "doc": "Delete what is at a path, so that a meta-algorithm can write a directory over "
        "it. `MlCommand.Write` carries its own overwrite flag; a pipeline writes its own "
        "directory and needs this instead.",
    }
}

# A `DataType()` constructor at a call site, as a `Latu.ML.Plan.param_type()`. Three of the ten
# calls pass `(value, DataType)` rather than a bare value — `pyspark.ml.connect.serialize`'s
# explicit arm — because an empty list and a nested one cannot be inferred from the value.
DATA_TYPES = {
    "StringType": "string",
    "DoubleType": "double",
    "FloatType": "double",
    "IntegerType": "int",
    "LongType": "long",
    "BooleanType": "boolean",
}

# What an argument's Python annotation means on the wire. `DataFrame` is the one that is not a
# literal at all: it rides `Fetch.Method.Args`'s `input` arm as a relation, where every other
# type rides `param`. The rest are the same vocabulary `dev/extract_ml_operators.py` writes for
# params, so `Latu.ML.Plan.literal/2` needs nothing new to build them.
ARG_TYPES = {
    "DataFrame": "frame",
    "Vector": "vector",
    "Matrix": "matrix",
    "int": "int",
    "float": "double",
    "str": "string",
    "bool": "boolean",
}

# `predict(value: T)` is declared on the generic `PredictionModel[T]`, so its real type is
# whichever class the model parameterises that base with — `_JavaProbabilisticClassificationModel
# [Vector]` and its kind. Three regression models forget to write the parameter at all while
# every sibling writes it, so `T` cannot be resolved for them:
#
#     class RandomForestRegressionModel(_JavaRegressionModel[Vector], ...)   # written
#     class LinearRegressionModel(_JavaRegressionModel, ...)                 # not
#
# It is an upstream slip rather than a difference — a regression model predicts from a features
# vector like everything else does — and it is pinned here rather than guessed at, so that a
# release which fixes it shows up as three lines that can go. Filed as
# `latu-upstream-findings.md`.
TYPEVAR_PINS = {
    "DecisionTreeRegressionModel": "Vector",
    "FMRegressionModel": "Vector",
    "LinearRegressionModel": "Vector",
}


def snake(name):
    """The same rule as `dev/extract_ml_operators.py`, for attribute and class names."""
    out = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    out = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", out)
    return out.lower()


# =============================================
# The Scala side
# =============================================


def fetch_scala(version, local):
    if local is not None:
        return pathlib.Path(local).read_text()
    with urllib.request.urlopen(SCALA_URL % version) as response:
        return response.read().decode("utf-8")


def parse_allowed(source):
    """The `ALLOWED_ATTRIBUTES` entries, in the file's own order.

    The list is Scala, not data, so this is a parser and not a loader — but the shape it has to
    read is narrow: `(classOf[Name], Set("a", "b"))`, optionally with type parameters
    (`classOf[PredictionModel[_, _]]`) and written across as many lines as `scalafmt` felt
    like. Anything in the block this cannot read is reported rather than skipped, which is the
    only reason a regex is safe here.
    """
    start = source.index("ALLOWED_ATTRIBUTES")
    end = source.index("private def validate", start)
    block = source[start:end]

    entries = []
    for match in re.finditer(
        r"classOf\[\s*([A-Za-z0-9_]+)\s*(?:\[[^\]]*\])?\s*\]\s*,\s*Set\(([^)]*)\)", block
    ):
        name = match.group(1)
        methods = re.findall(r'"([^"]+)"', match.group(2))
        entries.append((name, sorted(set(methods))))

    found = len(re.findall(r"classOf\[", block))
    if found != len(entries):
        raise SystemExit(
            "read %d of %d classOf entries in ALLOWED_ATTRIBUTES — the shape has changed"
            % (len(entries), found)
        )
    return entries


def scala_imports(source):
    """Explicit single-class imports, as `{short: fully.qualified}`.

    `MLUtils.scala` imports most of `org.apache.spark.ml` with wildcards, but names a few
    classes one by one — `tree.{DecisionTreeModel, TreeEnsembleModel}` and five out of `util`.
    Those are exact, so they take precedence, and every name the module rule also resolves is
    checked against them.
    """
    out = {}
    for package, names in re.findall(r"^import ([\w.]+)\.\{([^}]+)\}", source, re.M):
        for name in names.split(","):
            name = name.strip()
            if name and "=>" not in name:
                out[name] = "%s.%s" % (package, name)
    for package, name in re.findall(r"^import ([\w.]+)\.([A-Z]\w*)$", source, re.M):
        out[name] = "%s.%s" % (package, name)
    return out


# =============================================
# The PySpark side
# =============================================


def python_classes():
    """Every class in `pyspark.ml`, by name, including the private mixins.

    Spark's summary *traits* — `ClassificationSummary`, `TrainingSummary` — are private classes
    in PySpark under a leading underscore, so both spellings are indexed and a name resolving
    to two different classes is an error rather than a coin toss.
    """
    index = {}
    for module in PACKAGES:
        mod = importlib.import_module("pyspark.ml." + module)
        for attr in dir(mod):
            obj = getattr(mod, attr)
            if inspect.isclass(obj) and obj.__module__ == mod.__name__:
                index.setdefault(attr.lstrip("_"), []).append((module, attr, obj))
    return index


def relation_valued():
    """`(class, method)` for every attribute PySpark decorates `try_remote_attribute_relation`.

    Walked with a class stack rather than by `ast.walk`, because one of them is a *nested*
    function: `VectorIndexerModel.categoryMaps` defines and decorates `categoryMapsDF` inside
    its own body, and that inner name is the one on the server's allowlist.
    """
    root = pathlib.Path(importlib.import_module("pyspark.ml").__file__).parent
    marked = set()

    def visit(node, owner):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, child.name)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names = [
                    d.id if isinstance(d, ast.Name) else getattr(d, "attr", "")
                    for d in child.decorator_list
                ]
                if owner and "try_remote_attribute_relation" in names:
                    marked.add((owner, child.name))
                visit(child, owner)
            else:
                visit(child, owner)

    for path in sorted(root.glob("*.py")):
        visit(ast.parse(path.read_text()), None)
    return marked


# =============================================
# The helper object
# =============================================


def helper_calls():
    """Every `invoke_helper_*` call site in `pyspark.ml`, as a row per helper method.

    The names are the server's, from `ALLOWED_ATTRIBUTES`; what each one *takes* is only here,
    at the call site, because `ConnectHelper` has no Python class and so no signature to read.
    So this walks for the calls and resolves each argument expression — a bare name against the
    enclosing function's annotation, a `self.getX()` against that class's own param table, an
    explicit `(value, DataType)` tuple by reading the `DataType`.

    Which shape a helper answers with is read the same way: `invoke_helper_relation` is a
    DataFrame, and an `invoke_helper_attr` whose result PySpark wraps in `RemoteModelRef` is a
    model reference rather than a value.
    """
    root = pathlib.Path(importlib.import_module("pyspark.ml").__file__).parent
    found = {}

    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text())
        # A call whose *result* PySpark wraps in a `RemoteModelRef` answers with a model. The
        # wrapping is in the parent expression, so the inner calls are marked on the way past.
        wrapped = {
            id(node.args[0])
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "RemoteModelRef"
            and node.args
        }
        visit_helpers(tree, path.stem, None, None, wrapped, found)

    return found


def visit_helpers(node, module, cls, function, wrapped, found):
    """Walk with the enclosing class and function in hand — both are needed to type an
    argument, and `ast.walk` throws them away."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.ClassDef):
            visit_helpers(child, module, child, function, wrapped, found)
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            visit_helpers(child, module, cls, child, wrapped, found)
        else:
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id in ("invoke_helper_attr", "invoke_helper_relation")
            ):
                record_helper(child, module, cls, function, wrapped, found)
            visit_helpers(child, module, cls, function, wrapped, found)


def record_helper(call, module, cls, function, wrapped, found):
    where = "%s.%s" % (cls.name if cls else module, function.name if function else "?")

    if not call.args or not isinstance(call.args[0], ast.Constant):
        raise SystemExit("%s: a helper call whose method name is not a literal" % where)

    method = call.args[0].value
    relation = call.func.id == "invoke_helper_relation"
    shape = "frame" if relation else ("model" if id(call) in wrapped else "value")

    row = {
        "name": snake(method),
        "method": method,
        "returns": shape,
        "model_class": java_name(module, cls.name) if shape == "model" else None,
        "python": where,
        # The Elixir name a generated function takes: PySpark's own, snaked. `python` stays
        # verbatim because it is provenance — which call site every argument was read from.
        "function": snake(function.name) if function else None,
        "read_by_hand": False,
        "doc": helper_doc(function),
        "args": [helper_arg(a, cls, function, where) for a in call.args[1:]],
    }

    if method in found and found[method] != row:
        raise SystemExit(
            "%s: two call sites for `%s` that do not agree — read both" % (where, method)
        )
    found[method] = row


def helper_doc(function):
    """The enclosing function's first docstring paragraph, where there is one worth having.

    `__init__`'s docstring is its own signature line, which says nothing about the helper it
    calls, so it is left out rather than rendered.
    """
    if function is None or function.name == "__init__":
        return None
    return clean_doc(ast.get_docstring(function))


def helper_arg(expr, cls, function, where):
    """One call-site argument as `{name, type}`, or a refusal naming what could not be read."""
    # `(value, DataType())` — the explicit arm. The type is written down; the name comes from
    # the value beside it.
    if isinstance(expr, ast.Tuple) and len(expr.elts) == 2:
        return {
            "name": expr_name(expr.elts[0], where),
            "type": data_type_of(expr.elts[1], where),
        }

    # `self.getWeightCol() if self.isDefined(...) else ""` — one call site, and the fallback is
    # a fact rather than a convention: a helper's arguments are **positional**, so every one is
    # sent every time, and a param Spark gives no default has to become something. What that
    # something is, is written down here rather than guessed at in Elixir.
    if isinstance(expr, ast.IfExp):
        argument = helper_arg(expr.body, cls, function, where)
        if not isinstance(expr.orelse, ast.Constant):
            raise SystemExit(
                "%s: `%s` falls back to something that is not a literal"
                % (where, ast.unparse(expr))
            )
        argument["unset"] = expr.orelse.value
        return argument

    # `self.getK()` — a param of the enclosing class, whose type its own converter gives.
    if (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Attribute)
        and expr.func.attr.startswith("get")
        and isinstance(expr.func.value, ast.Name)
        and expr.func.value.id == "self"
    ):
        wire = expr.func.attr[3].lower() + expr.func.attr[4:]
        return {"name": snake(wire), "type": param_type(cls, wire, where)}

    # `model.uid`, the client-generated uid a helper-built model is registered under.
    if isinstance(expr, ast.Attribute) and expr.attr == "uid":
        return {"name": "uid", "type": "string"}

    # `list(vocabulary)` is `vocabulary`; a bare name is itself. Either way the type is the
    # enclosing function's annotation for it.
    inner = expr.args[0] if is_call_to(expr, "list") else expr
    if isinstance(inner, ast.Name):
        return {"name": snake(inner.id), "type": annotation_type(function, inner.id, where)}

    raise SystemExit(
        "%s: cannot read the argument `%s` — add a case to `helper_arg`"
        % (where, ast.unparse(expr))
    )


def is_call_to(expr, name):
    return (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Name)
        and expr.func.id == name
        and len(expr.args) == 1
    )


def expr_name(expr, where):
    """A readable name for the value half of a `(value, DataType)` pair."""
    if is_call_to(expr, "list"):
        expr = expr.args[0]
    if isinstance(expr, ast.Name):
        return snake(expr.id)
    # `[float(p) for p in params]` and `[list(x) for x in arrayOfLabels]`: the iterable is what
    # the caller passed, and its name is the one worth carrying.
    if isinstance(expr, (ast.ListComp, ast.GeneratorExp)) and expr.generators:
        return expr_name(expr.generators[0].iter, where)
    raise SystemExit("%s: cannot name the argument `%s`" % (where, ast.unparse(expr)))


def data_type_of(expr, where):
    """`ArrayType(ArrayType(StringType()))` as `{:list, {:list, :string}}`."""
    if not isinstance(expr, ast.Call) or not isinstance(expr.func, ast.Name):
        raise SystemExit("%s: `%s` is not a DataType constructor" % (where, ast.unparse(expr)))
    if expr.func.id == "ArrayType":
        if not expr.args:
            raise SystemExit("%s: an ArrayType with no element type" % where)
        return list_of(data_type_of(expr.args[0], where))
    if expr.func.id in DATA_TYPES:
        return DATA_TYPES[expr.func.id]
    raise SystemExit(
        "%s: `%s` is a DataType nothing here maps — add it to DATA_TYPES and to "
        "`Latu.ML.Plan.literal/2` if it is a new shape" % (where, expr.func.id)
    )


def annotation_type(function, name, where):
    """The declared type of one of the enclosing function's parameters."""
    if function is None:
        raise SystemExit("%s: `%s` is not a parameter of anything" % (where, name))
    for argument in function.args.args + function.args.kwonlyargs:
        if argument.arg == name:
            if argument.annotation is None:
                raise SystemExit("%s: `%s` has no annotation to read" % (where, name))
            kind = type_from_annotation(ast.unparse(argument.annotation))
            if kind is None:
                raise SystemExit(
                    "%s: `%s` is annotated %s, which nothing in ARG_TYPES maps"
                    % (where, name, ast.unparse(argument.annotation))
                )
            return kind
    raise SystemExit("%s: `%s` is not a parameter of it" % (where, name))


def type_from_annotation(text):
    """An annotation's source text as a `Latu.ML.Plan.param_type()`, or None."""
    text = text.strip().strip("\"'")
    nested = re.fullmatch(r"(?:typing\.)?List\[(.+)\]", text)
    if nested:
        inner = type_from_annotation(nested.group(1))
        return list_of(inner) if inner else None
    return ARG_TYPES.get(text.split(".")[-1])


def list_of(inner):
    return "{:list, %s}" % (inner if inner.startswith("{") else ":" + inner)


def param_type(cls, wire, where):
    """A param's declared type, read off the class the call site sits in.

    The same converter table `dev/extract_ml_operators.py` reads, imported rather than copied:
    a `self.getK()` at a call site and a `k:` in the operator's own param table have to mean
    the same type, or the two registries disagree about one param.
    """
    from pyspark.ml.param import Param

    if cls is None:
        raise SystemExit("%s: `self.get%s()` outside a class" % (where, wire))

    owner = find_class(cls.name)

    for attr in dir(owner):
        try:
            value = inspect.getattr_static(owner, attr)
        except AttributeError:
            continue
        if isinstance(value, Param) and value.name == wire:
            converter = getattr(value.typeConverter, "__name__", str(value.typeConverter))
            declared = CONVERTERS.get(converter)
            if declared is None:
                declared = IDENTITY_TYPES.get((owner.__name__, wire))
            if declared is None:
                raise SystemExit(
                    "%s: %s.%s has converter %s, which nothing maps"
                    % (where, owner.__name__, wire, converter)
                )
            return declared
    raise SystemExit("%s: %s has no param named %s" % (where, cls.name, wire))


def find_class(name):
    """A class by name, anywhere in the `pyspark.ml` modules this script reads."""
    for module in PACKAGES:
        found = getattr(importlib.import_module("pyspark.ml." + module), name, None)
        if inspect.isclass(found):
            return found
    raise SystemExit("no class named %s in pyspark.ml" % name)


def java_name(module, name):
    """A Python class as its JVM name, PySpark's own module rule."""
    return ".".join(filter(None, [JAVA_ROOT, PACKAGES.get(module, module), name]))


def attribute_of(cls, method):
    """PySpark's own function behind an attribute name, or None if it has none."""
    try:
        value = inspect.getattr_static(cls, method)
    except AttributeError:
        return None
    if isinstance(value, property):
        value = value.fget
    if isinstance(value, (staticmethod, classmethod)):
        value = value.__func__
    return inspect.unwrap(getattr(value, "func", value))


def doc_of(cls, method):
    """PySpark's description of an attribute, or None.

    `clean_doc` is imported from `dev/extract_ml_operators.py` rather than copied: the rule for
    turning a PySpark docstring into a generated `@doc` is one rule, and two copies of it drift.
    A docstring inherited from `object` describes nothing and is dropped before it gets there.
    """
    function = attribute_of(cls, method)
    text = inspect.getdoc(function) if function is not None else None
    if not text or text == inspect.getdoc(object):
        return None
    return clean_doc(text)


def summary_dataset(cls):
    """Which frame `CreateSummary` has to be given to rebuild this model's summary.

    `HasTrainingSummary._summary_dataset` defaults to `self.transform(train_dataset)`, and
    seven of the ten models that have a summary override it to return the raw training frame
    instead. Sending the wrong one would only ever fail under eviction, which is the hardest
    kind of bug to notice, so it is read off the class rather than assumed.

    An override whose body is anything but `return train_dataset` stops the extractor: the rule
    is only safe because every override in 4.2.0 is that one line.
    """
    if not any(inspect.isclass(c) and "_summary_dataset" in c.__dict__ for c in [cls]):
        return "transformed" if summary_classes(cls) is not None else None

    source = textwrap.dedent(inspect.getsource(cls.__dict__["_summary_dataset"]))
    body = ast.parse(source).body[0].body
    returns_raw = (
        len(body) == 1
        and isinstance(body[0], ast.Return)
        and isinstance(body[0].value, ast.Name)
        and body[0].value.id == "train_dataset"
    )

    if not returns_raw:
        raise SystemExit(
            "%s._summary_dataset is not `return train_dataset` — read it and widen this rule"
            % cls.__name__
        )

    return "training_frame"


def summary_classes(cls):
    """Every summary class this model's `summary` can answer with, in the source's own order.

    PySpark keeps it on a `_summaryCls` property, which is how it wraps the ref the server
    sends back. Eight of the ten return one class outright. Two — `LogisticRegressionModel` and
    `RandomForestClassificationModel` — branch on the *fitted* model's `numClasses` and return
    the binary summary or the multiclass one, so no client can know which before it asks:

        def _summaryCls(self) -> type:
            if self.numClasses <= 2:
                return BinaryLogisticRegressionTrainingSummary
            return LogisticRegressionTrainingSummary

    Calling the property is therefore the wrong question. Reading its returns is the right one,
    and a list of candidates is the honest answer: a one-element list where PySpark commits, a
    two-element one where it does not. A branch returning anything but a bare name is refused —
    that is a shape this has never seen and should be read rather than guessed at.
    """
    value = inspect.getattr_static(cls, "_summaryCls", None)
    if not isinstance(value, property):
        return None
    source = textwrap.dedent(inspect.getsource(value.fget))
    module = sys.modules[cls.__module__]
    returns = [n for n in ast.walk(ast.parse(source).body[0]) if isinstance(n, ast.Return)]
    names = []
    # By line, not by `ast.walk`'s order: the walk is breadth-first, so a bare `return` at the
    # end of the body would come out ahead of the one inside the `if` above it.
    for node in sorted(returns, key=lambda n: n.lineno):
        if not isinstance(node.value, ast.Name):
            raise SystemExit(
                "%s._summaryCls returns %s, which is not a bare class name — read it"
                % (cls.__name__, ast.unparse(node.value))
            )
        names.append(node.value.id)
    return [getattr(module, name) for name in names]


def returns(cls, method, relations):
    """Which of the four shapes an attribute's answer takes.

      * `:frame` — a DataFrame, so the `Fetch` is wrapped as a relation. PySpark marks these
        with `try_remote_attribute_relation`.
      * `:summary` — another cached object, whose ref the client turns into `"<model>.summary"`
        and fetches through. `MlCommandResult` answers on its `summary` arm, not `param`.
      * `:operators` — a *list* of cached objects. `TreeEnsembleModel.trees` is the only one:
        the server answers on the `operator_info` arm with refs to the sub-models, which
        PySpark splits on a comma and wraps. Measured, not read — `dev/probe_ml.exs` fetched
        one and Latu ML's own decoder said "the server answered with operator_info where a
        param belongs".
      * `:value` — a literal.

    A summary is recognised from PySpark's return annotation, which names the summary class for
    `evaluate`; `HasTrainingSummary.summary` is annotated with a type variable instead, and is
    recognised by the model carrying a `_summaryCls` at all. A list of operators is recognised
    from the same place: `List[SomethingModel]`.
    """
    if (cls.__name__, method) in relations or any(
        (owner.__name__, method) in relations for owner in cls.__mro__
    ):
        return "frame"
    function = attribute_of(cls, method)
    annotation = getattr(function, "__annotations__", {}).get("return")
    if names_a(annotation, "Summary"):
        return "summary"
    if lists_a(annotation, "Model"):
        return "operators"
    if method == "summary" and summary_classes(cls) is not None:
        return "summary"
    return "value"


def element_class(cls, method):
    """For an `:operators` attribute, the JVM class of the sub-models it answers with.

    The same annotation that says the shape says the element: `TreeEnsembleModel.trees` is
    `List[DecisionTreeRegressionModel]` on a regressor and `List[DecisionTreeClassificationModel]`
    on a classifier. Without it a caller would get four references and no idea what they were,
    which is most of why `:operators` had no home.
    """
    function = attribute_of(cls, method)
    annotation = getattr(function, "__annotations__", {}).get("return")
    if isinstance(annotation, str):
        match = re.fullmatch(r"(?:typing\.)?List\[[\w.]*?(\w+)\]", annotation)
        name = match.group(1) if match else None
    else:
        args = typing.get_args(annotation) if annotation is not None else ()
        concrete = [a for a in args if inspect.isclass(a)]
        name = concrete[0].__name__ if len(concrete) == 1 else None
    if name is None:
        raise SystemExit(
            "%s.%s answers with a list of operators and nothing names the element class — "
            "read its annotation" % (cls.__name__, method)
        )
    module = sys.modules[cls.__module__]
    element = getattr(module, name, None)
    if element is None:
        raise SystemExit(
            "%s.%s names %s, which is not in %s" % (cls.__name__, method, name, module.__name__)
        )
    package = PACKAGES[element.__module__.rsplit(".", 1)[1]]
    return ".".join(filter(None, [JAVA_ROOT, package, element.__name__]))


def names_a(annotation, suffix):
    """Whether an annotation is a class whose name ends in `suffix`.

    PySpark writes these both ways — a forward-reference string on a plain method, a real class
    on a `cached_property` — so both are read.
    """
    if isinstance(annotation, str):
        return annotation.endswith(suffix)
    return inspect.isclass(annotation) and annotation.__name__.endswith(suffix)


def lists_a(annotation, suffix):
    """Whether an annotation is `List[X]` for an `X` whose name ends in `suffix`."""
    if isinstance(annotation, str):
        return bool(re.fullmatch(r"(?:typing\.)?List\[[\w.]*%s\]" % suffix, annotation))
    if typing.get_origin(annotation) not in (list, typing.List):
        return False
    args = typing.get_args(annotation)
    return bool(args) and all(names_a(arg, suffix) for arg in args)


def call_java_arity(cls, method):
    """Arity read off PySpark's own call site, for a name it does not expose as an attribute.

    Three allowlisted methods have no Python attribute of that name because PySpark wraps them
    under a different one — `KMeansModel.clusterCenters` is `_call_java("clusterCenterMatrix")`
    and reshapes the result. The call still says how many arguments the server's method takes,
    which is the only thing the registry wants, and it says it rather than guessing it.
    """
    try:
        source = textwrap.dedent(inspect.getsource(cls))
    except (OSError, TypeError):
        return None
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_call_java"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == method
        ):
            return len(node.args) - 1
    return None


def arity(cls, method):
    """How many arguments PySpark's version of an attribute takes, or None if nothing says.

    A property answers with no arguments; a method's own parameters are what a `Fetch` chain
    would have to carry, and ML2 generates an accessor only for the zero-argument ones — the
    rest stay with `Latu.ML.attribute/3` and its explicit argument list. Where PySpark has no
    attribute of that name, its own `_call_java` call site is asked instead.
    """
    try:
        value = inspect.getattr_static(cls, method)
    except AttributeError:
        return call_java_arity(cls, method)
    if isinstance(value, property):
        return 0
    if isinstance(value, (staticmethod, classmethod)):
        value = value.__func__
    unwrapped = inspect.unwrap(getattr(value, "func", value))
    if not inspect.isfunction(unwrapped):
        # `functools.cached_property` and anything else that is neither: it answers with no
        # arguments, like a property.
        return 0
    spec = inspect.getfullargspec(unwrapped)
    return max(len(spec.args) - 1, 0)


def signature_of(cls, method):
    """PySpark's parameters for an attribute, minus `self`, or None if there are none to read."""
    try:
        value = inspect.getattr_static(cls, method)
    except AttributeError:
        # A name PySpark reaches by `_call_java` under a different one. `call_java_arity` still
        # gives the arity; nothing gives the types.
        return None
    if isinstance(value, property):
        return None
    if isinstance(value, (staticmethod, classmethod)):
        value = value.__func__
    unwrapped = inspect.unwrap(getattr(value, "func", value))
    if not inspect.isfunction(unwrapped):
        return None
    return list(inspect.signature(unwrapped).parameters.values())[1:]


def typevar_binding(cls):
    """The concrete class a `Generic[T]` base is parameterised with, up the MRO.

    One parameter is all PySpark's ML hierarchy ever uses, so a base with exactly one concrete
    argument is unambiguous; anything else is left to `TYPEVAR_PINS` or refused.
    """
    if cls.__name__ in TYPEVAR_PINS:
        return TYPEVAR_PINS[cls.__name__]
    for base in cls.__mro__:
        for original in getattr(base, "__orig_bases__", ()) or ():
            concrete = [a for a in typing.get_args(original) if inspect.isclass(a)]
            if len(concrete) == 1:
                return concrete[0].__name__
    return None


def arg_type(cls, annotation):
    """One annotation as a `Latu.ML.Plan.param_type()`, or None if nothing here maps it."""
    if isinstance(annotation, typing.TypeVar):
        bound = typevar_binding(cls)
        return ARG_TYPES.get(bound) if bound else None
    if inspect.isclass(annotation):
        return ARG_TYPES.get(annotation.__name__)
    if typing.get_origin(annotation) is typing.Union:
        members = [arg_type(cls, a) for a in typing.get_args(annotation)]
        if any(m is None for m in members):
            return None
        return "{:either, [%s]}" % ", ".join(":" + m for m in members)
    if isinstance(annotation, str):
        return ARG_TYPES.get(annotation.replace("'", "").split(".")[-1])
    return None


def arguments(cls, method, method_arity):
    """The arguments a `Fetch` on this attribute has to carry, named and typed.

    ML2 generated an accessor only where a `Fetch` needs nothing but a name. These are the
    hundred that need more, and what they need is not guessable from the wire: `predict` takes
    a `Vector`, `recommendForAllUsers` an `int`, `evaluate` a whole DataFrame — three different
    arms of `Fetch.Method.Args`, distinguishable only from PySpark's annotations.

    Refuses rather than guesses, in both directions. A signature that disagrees with the arity
    read off the call site means the two sources have drifted, and an annotation nothing here
    maps means a type the plan layer has never had to build. Either is a reading to do by hand,
    not a default to fall back on.
    """
    if not method_arity:
        return None
    parameters = signature_of(cls, method)
    if parameters is None:
        return None
    if len(parameters) != method_arity:
        raise SystemExit(
            "%s.%s takes %d arguments by its signature and %d by its `_call_java` call site — "
            "read it and fix one of them" % (cls.__name__, method, len(parameters), method_arity)
        )
    out = []
    for parameter in parameters:
        kind = arg_type(cls, parameter.annotation)
        if kind is None:
            raise SystemExit(
                "%s.%s's `%s` is annotated %r, which nothing in ARG_TYPES maps — add it there "
                "and to `Latu.ML.Plan.literal/2` if it is a new shape"
                % (cls.__name__, method, parameter.name, parameter.annotation)
            )
        out.append({"name": snake(parameter.name), "type": kind})
    return out


# =============================================
# Putting the two together
# =============================================


def collect(source, version):
    entries = parse_allowed(source)
    imports = scala_imports(source)
    index = python_classes()
    relations = relation_valued()

    problems = []
    allowed = []
    trait_of = {}

    helper_methods = None

    for name, methods in entries:
        if name in EXCLUDE:
            if name == "ConnectHelper":
                helper_methods = methods
            continue

        candidates = index.get(name, [])
        if len(candidates) != 1:
            problems.append(
                "%s: %d classes in pyspark.ml answer to that name" % (name, len(candidates))
            )
            continue
        module, attr, cls = candidates[0]

        package = PACKAGES[module]
        derived = ".".join(filter(None, [JAVA_ROOT, package, name]))
        java = imports.get(name, derived)
        if name in imports and imports[name] != derived:
            problems.append(
                "%s: imported as %s, the module rule says %s" % (name, imports[name], derived)
            )
            continue

        trait_of[cls] = java
        allowed.append({"class": java, "python": attr, "methods": methods})

    concrete = []
    for module in OPERATOR_MODULES:
        mod = importlib.import_module("pyspark.ml." + module)
        for attr in sorted(dir(mod)):
            if attr.startswith("_") or attr in EXCLUDE_CLASSES:
                continue
            obj = getattr(mod, attr)
            if not inspect.isclass(obj) or obj.__module__ != mod.__name__:
                continue
            if inspect.isabstract(obj):
                continue
            kind = kind_of(obj, attr)
            if kind is None:
                continue

            inherited = [(cls, java) for cls, java in trait_of.items() if issubclass(obj, cls)]
            if not inherited:
                continue

            methods = {}
            for cls, java in inherited:
                for entry in allowed:
                    if entry["class"] != java:
                        continue
                    for method in entry["methods"]:
                        methods.setdefault(method, []).append(java)

            summary = summary_classes(obj)
            concrete.append(
                {
                    "name": snake(attr),
                    "python": attr,
                    "class": ".".join(filter(None, [JAVA_ROOT, PACKAGES[module], attr])),
                    "kind": kind,
                    "summary_classes": summary_names(summary),
                    "summary_dataset": summary_dataset(obj),
                    "from": sorted({java for _cls, java in inherited}),
                    "attributes": [
                        {
                            "name": snake(method),
                            "wire": method,
                            "returns": returns(obj, method, relations),
                            "arity": arity(obj, method),
                            "args": arguments(obj, method, arity(obj, method)),
                            "element_class": (
                                element_class(obj, method)
                                if returns(obj, method, relations) == "operators"
                                else None
                            ),
                            "doc": doc_of(obj, method),
                        }
                        for method in sorted(methods)
                    ],
                }
            )

    helpers = helper_calls()
    problems += helper_problems(helper_methods, helpers)

    for method, pinned in HELPERS_WITHOUT_CALL_SITE.items():
        if method in (helper_methods or []):
            helpers[method] = dict(
                pinned, name=snake(method), method=method, model_class=None, read_by_hand=True
            )

    return (
        allowed,
        sorted(concrete, key=lambda c: (c["kind"], c["name"])),
        [helpers[method] for method in sorted(helpers)],
        problems,
    )


def helper_problems(scala_methods, helpers):
    """The two sources on `ConnectHelper` have to agree, and silence is not an outcome.

    The **names** are the server's, from `ALLOWED_ATTRIBUTES`; the **arguments** are PySpark's,
    from the call sites. A method in one and not the other is a real finding either way: a call
    PySpark makes that the server would refuse, or a method a release added that nothing here
    can type.
    """
    if scala_methods is None:
        return ["ConnectHelper is no longer in ALLOWED_ATTRIBUTES — the helper route is gone"]

    problems = []
    for method in sorted(set(helpers) - set(scala_methods)):
        problems.append(
            "%s: PySpark calls it and the server's allowlist does not have it" % method
        )
    for method in sorted(set(scala_methods) - set(helpers) - set(HELPERS_WITHOUT_CALL_SITE)):
        problems.append(
            "%s: the server allows it and no `invoke_helper_*` call site says what it takes — "
            "read it by hand and excuse it in HELPERS_WITHOUT_CALL_SITE" % method
        )
    for method in sorted(set(HELPERS_WITHOUT_CALL_SITE) - set(scala_methods)):
        problems.append(
            "%s: excused as having no call site, and the server no longer allows it" % method
        )
    return problems


def summary_names(summaries):
    """`summary_classes`' answer as JVM class names, or None where there is no summary."""
    if summaries is None:
        return None
    return [
        ".".join(
            filter(
                None,
                [JAVA_ROOT, PACKAGES[summary.__module__.rsplit(".", 1)[1]], summary.__name__],
            )
        )
        for summary in summaries
    ]


def kind_of(obj, name):
    """`:model`, `:summary`, or None for anything that is neither."""
    from pyspark.ml.base import Model

    if issubclass(obj, Model):
        return "model"
    if name.endswith("Summary"):
        return "summary"
    return None


# =============================================
# Rendering
# =============================================

LINE_LENGTH = 98


def render_helper(helper, indent):
    """One `ConnectHelper` method as a map, in `render_attribute`'s shape."""
    pad = " " * indent
    fields = [
        ("name", ":" + helper["name"]),
        ("method", '"%s"' % helper["method"]),
        ("returns", ":" + helper["returns"]),
        (
            "model_class",
            "nil" if helper["model_class"] is None else '"%s"' % helper["model_class"],
        ),
        ("python", '"%s"' % helper["python"]),
        ("function", ":" + helper["function"]),
        ("read_by_hand", "true" if helper["read_by_hand"] else "false"),
        ("args", render_args(helper["args"], indent + 2)),
        ("doc", elixir_string(helper["doc"])),
    ]
    single = "%s%%{%s}" % (pad, ", ".join("%s: %s" % f for f in fields))
    if len(single) + 1 <= LINE_LENGTH:
        return single

    last = len(fields) - 1
    lines = []
    for i, (key, value) in enumerate(fields):
        separator = "" if i == last else ","
        if "\n" in value:
            lines.append("%s%s: %s%s" % (" " * (indent + 2), key, value, separator))
        else:
            lines.append(pair(indent + 2, key, value, separator).rstrip("\n"))
    return "%s%%{\n%s\n%s}" % (pad, "\n".join(lines), pad)


def render(allowed, classes, helpers, version):
    out = [
        "# Generated by dev/extract_ml_allowlist.py from Spark %s. Do not hand-edit.\n" % version,
        "#\n",
        "# `allowed` is `ALLOWED_ATTRIBUTES` in the server's own MLUtils.scala, by a parser.\n",
        "# `classes` is what each concrete model and summary can actually be asked: the server\n",
        "# walks `cls.isInstance(obj)`, so an object's allowlist is the union up its chain,\n",
        "# resolved here through PySpark's mirror of it and confirmed by dev/probe_ml.exs.\n",
        "#\n",
        "# `:summary_classes` is every class a model's `summary` can answer with, in the\n",
        "# order PySpark's `_summaryCls` returns them: one entry where it commits, two where\n",
        "# it branches on the fitted model, which no client can know before it asks.\n",
        "#\n",
        "# `:summary_dataset` is which frame `CreateSummary` needs to rebuild a lost summary:\n",
        "# the raw training frame, or the model's own transform of it.\n",
        "#\n",
        "# `:returns` says how an attribute answers: `:value` a literal, `:frame` a DataFrame\n",
        "# (a `Fetch` as a relation), `:summary` another cached object, `:operators` a list of\n",
        "# them. `:arity` is how many arguments it takes and `:doc` is PySpark's own first\n",
        "# paragraph — both nil where PySpark has no matching attribute to read them from.\n",
        "#\n",
        "# `:element_class` is the sub-model class an `:operators` attribute answers with, and\n",
        "# is nil for every other shape.\n",
        "#\n",
        "# `:args` names and types those arguments, from PySpark's own annotations. `:frame` is\n",
        "# a DataFrame, which rides `Fetch.Method.Args`'s `input` arm; every other type is a\n",
        "# literal on the `param` arm, in the same vocabulary a param uses.\n",
        "#\n",
        "# `helpers` is the server's own `ConnectHelper` object, which is in the allowlist but\n",
        "# is an attribute of nothing: a `Fetch` names it by the literal id\n",
        "# `______ML_CONNECT_HELPER______` and no model is involved. Its names are the Scala's\n",
        "# and its arguments are PySpark's call sites', because it has no class and so no\n",
        "# signature; `:python` says which call site each was read from. `:returns` is `:value`,\n",
        "# `:frame`, or `:model` for the three that build a model and answer with its reference.\n",
        "# `:function` is the Elixir name a generated one takes, which is PySpark's own snaked.\n",
        "# `:read_by_hand` marks the one method with no `invoke_helper_*` call site to read:\n",
        "# `handleOverwrite`, whose arguments are pinned in the extractor with their reason.\n",
        "#\n",
        "#     python dev/extract_ml_allowlist.py --write\n",
        "%{\n",
        '  spark: "%s",\n' % version,
        "  allowed: [\n",
    ]
    for entry in allowed:
        out.append("    %{\n")
        out.append(pair(6, "class", '"%s"' % entry["class"]))
        out.append(pair(6, "python", '"%s"' % entry["python"]))
        out.append(wrapped_list("methods", entry["methods"], 6) + "\n")
        out.append("    },\n")
    if allowed:
        out[-1] = "    }\n"
    out.append("  ],\n")
    out.append("  classes: [\n")
    for row in classes:
        out.append("    %{\n")
        out.append(pair(6, "name", ":" + row["name"]))
        out.append(pair(6, "python", '"%s"' % row["python"]))
        out.append(pair(6, "class", '"%s"' % row["class"]))
        out.append(pair(6, "kind", ":" + row["kind"]))
        summary = row["summary_classes"]
        if summary is None:
            out.append(pair(6, "summary_classes", "nil"))
        else:
            out.append(wrapped_list("summary_classes", summary, 6) + ",\n")
        dataset = row["summary_dataset"]
        out.append(pair(6, "summary_dataset", "nil" if dataset is None else ":" + dataset))
        out.append(wrapped_list("from", row["from"], 6) + ",\n")
        out.append("      attributes: [\n")
        last = len(row["attributes"]) - 1
        out.append(
            ",\n".join(
                render_attribute(a, 8, "" if i == last else ",")
                for i, a in enumerate(row["attributes"])
            )
            + "\n"
        )
        out.append("      ]\n")
        out.append("    },\n")
    if classes:
        out[-1] = "    }\n"
    out.append("  ],\n")
    out.append("  helpers: [\n")
    last = len(helpers) - 1
    out.append(
        ",\n".join(render_helper(h, 4) for h in helpers) + ("\n" if helpers else "")
    )
    out.append("  ]\n")
    out.append("}\n")
    return "".join(out)


def elixir_string(text):
    """A Python string as an Elixir string literal, or `nil`."""
    if text is None:
        return "nil"
    return json.dumps(text, ensure_ascii=False).replace("#{", "\\#{")


def pair(indent, key, value, trailing=","):
    """`key: value,` wrapped the way `mix format` wraps it when the line is too long."""
    pad = " " * indent
    line = "%s%s: %s%s" % (pad, key, value, trailing)
    if len(line) <= LINE_LENGTH:
        return line + "\n"
    return "%s%s:\n%s  %s%s\n" % (pad, key, pad, value, trailing)


def render_args(entries, indent):
    """`nil`, or the argument list, wrapped where `mix format` would wrap it.

    Four attributes take four arguments and only those four need more than a line. The
    formatter breaks the *list*, one map per line, rather than breaking after `args:` — so this
    returns a value that already carries its own newlines and `render_attribute` prints it
    without measuring again.
    """
    if not entries:
        return "nil"
    maps = [
        "%%{name: :%s, type: %s%s}"
        % (
            entry["name"],
            entry["type"] if entry["type"].startswith("{") else ":" + entry["type"],
            # Only a helper argument ever has one, and only one of them does.
            "" if "unset" not in entry else ", unset: " + elixir_string(entry["unset"]),
        )
        for entry in entries
    ]
    single = "[%s]" % ", ".join(maps)
    # `args: ` and the trailing comma, which the formatter measures too.
    if indent + len("args: ") + len(single) + 1 <= LINE_LENGTH:
        return single
    pad = " " * indent
    return "[\n%s\n%s]" % (",\n".join("%s  %s" % (pad, m) for m in maps), pad)


def render_attribute(attribute, indent, trailing):
    """One attribute as a map, on one line where `mix format` would leave it on one.

    `trailing` is the separator that will follow — the formatter measures the comma too, and
    four of these maps are exactly one character short of fitting without it.
    """
    pad = " " * indent
    fields = [
        ("name", ":" + attribute["name"]),
        ("wire", '"%s"' % attribute["wire"]),
        ("returns", ":" + attribute["returns"]),
        ("arity", "nil" if attribute["arity"] is None else str(attribute["arity"])),
        ("args", render_args(attribute["args"], indent + 2)),
        (
            "element_class",
            "nil"
            if attribute["element_class"] is None
            else '"%s"' % attribute["element_class"],
        ),
        ("doc", elixir_string(attribute["doc"])),
    ]
    single = "%s%%{%s}" % (pad, ", ".join("%s: %s" % f for f in fields))
    if len(single) + len(trailing) <= LINE_LENGTH:
        return single

    last = len(fields) - 1
    lines = []
    for i, (key, value) in enumerate(fields):
        separator = "" if i == last else ","
        if "\n" in value:
            # Already wrapped by `render_args`, which knew the indent; `pair` would measure the
            # whole block and wrap it a second time.
            lines.append("%s%s: %s%s" % (" " * (indent + 2), key, value, separator))
        else:
            lines.append(pair(indent + 2, key, value, separator).rstrip("\n"))
    return "%s%%{\n%s\n%s}" % (pad, "\n".join(lines), pad)


def wrapped_list(key, values, indent):
    """`key: ["a", "b"]`, one item per line when the single line would be too long.

    No trailing newline: the caller adds the comma, because whether there is one depends on
    where the entry sits.
    """
    pad = " " * indent
    items = ['"%s"' % v for v in values]
    single = "%s%s: [%s]" % (pad, key, ", ".join(items))
    if len(single) <= LINE_LENGTH:
        return single
    body = ",\n".join("%s  %s" % (pad, item) for item in items)
    return "%s%s: [\n%s\n%s]" % (pad, key, body, pad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pyspark", type=pathlib.Path, help="path to the pyspark package's parent")
    ap.add_argument("--scala", help="a local MLUtils.scala, instead of fetching it by tag")
    ap.add_argument("--json", action="store_true", help="dump rows as JSON, not a summary")
    ap.add_argument("--write", action="store_true", help="rewrite priv/ml_attributes.exs")
    ap.add_argument("--check", action="store_true", help="exit 1 if that file is out of date")
    args = ap.parse_args()

    here = pathlib.Path(__file__).resolve().parent
    root = here.parent

    if args.pyspark is not None:
        sys.path.insert(0, str(args.pyspark))
    else:
        latu_dev = os.environ.get("LATU_DEV")
        candidate = pathlib.Path(latu_dev) if latu_dev else root.parent / "latu" / "dev"
        found = list(candidate.glob(".venv/lib/python*/site-packages/pyspark"))
        if not found:
            sys.exit(
                "no pyspark found — check out latu beside this repo, set LATU_DEV, "
                "or pass --pyspark"
            )
        sys.path.insert(0, str(found[0].parent))

    import pyspark

    version = pyspark.__version__
    allowed, classes, helpers, problems = collect(fetch_scala(version, args.scala), version)

    if problems:
        sys.exit("\n".join(["the allowlist is not clean:"] + ["  " + p for p in problems]))

    if args.json:
        json.dump(
            {"allowed": allowed, "classes": classes, "helpers": helpers}, sys.stdout, indent=1
        )
        return

    target = root / "priv" / "ml_attributes.exs"
    source = render(allowed, classes, helpers, version)

    if args.check:
        current = target.read_text() if target.exists() else ""
        if current != source:
            sys.exit(
                "%s is out of date — run: python dev/extract_ml_allowlist.py --write" % target
            )
        print("%s is current for Spark %s" % (target, version))
        return

    if args.write:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source)
        print(
            "%d allowlist entries over %d classes, %d helpers -> %s (run `mix format` next)"
            % (len(allowed), len(classes), len(helpers), target)
        )
        return

    frames = sum(1 for c in classes for a in c["attributes"] if a["returns"] == "frame")
    total = sum(len(c["attributes"]) for c in classes)
    kinds = {}
    for row in classes:
        kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1
    print("Spark %s: %d allowlist entries" % (version, len(allowed)))
    print("  %d classes reach them: %s" % (len(classes), kinds))
    print("  %d attributes in all, %d of them DataFrame-valued" % (total, frames))
    typed = [a for c in classes for a in c["attributes"] if a["args"]]
    shapes = {}
    for attribute in typed:
        for argument in attribute["args"]:
            shapes[argument["type"]] = shapes.get(argument["type"], 0) + 1
    print("  %d take arguments, and every one is typed: %s" % (len(typed), shapes))
    without = [
        "%s.%s" % (c["python"], a["wire"])
        for c in classes
        for a in c["attributes"]
        if a["arity"] is None
    ]
    print("  %d have no PySpark attribute to read an arity from" % len(without))
    for name in sorted(set(without)):
        print("      %s" % name)
    shapes = {}
    for helper in helpers:
        shapes[helper["returns"]] = shapes.get(helper["returns"], 0) + 1
    print("  %d ConnectHelper methods, by shape: %s" % (len(helpers), shapes))
    for name, pinned in sorted(HELPERS_WITHOUT_CALL_SITE.items()):
        print("      %s is pinned by hand: %s" % (name, pinned["why"]))


if __name__ == "__main__":
    main()
