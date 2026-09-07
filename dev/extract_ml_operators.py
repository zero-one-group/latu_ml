#!/usr/bin/env python3
"""Derive Latu ML's operator registry from PySpark's own param tables.

The twin of Latu's `dev/extract_functions.py`, and the answer to the objection that closed
MLlib out of Latu: a hundred operators' parameters transcribed by hand would drift and nothing
would notice. Nothing here is transcribed. Every row is read off a live `pyspark.ml` class —
its params, their type converters, their defaults and their docs — so a Spark bump is a re-run
and a diff.

    python dev/extract_ml_operators.py                # summarise
    python dev/extract_ml_operators.py --write        # rewrite priv/ml_operators.exs
    python dev/extract_ml_operators.py --check        # exit 1 if that file is out of date

**Every public operator class must be a row or be excluded by name**, the same rule
`extract_functions.py` holds: a Spark upgrade adding an estimator is exactly the moment a
silent omission would cost the most.

Two things about the environment, both load-bearing:

  * `SPARK_CONNECT_MODE_ENABLED` is set and `SPARK_REMOTE` is cleared before PySpark is
    imported. Under Connect, `JavaWrapper._new_java_obj` is decorated
    `try_remote_return_java_class` and *returns the class name* rather than reaching a JVM,
    which is where every `class:` in the registry comes from. Classic mode would need a
    running SparkContext to instantiate anything.
  * Nothing may reach a server. `StopWordsRemover.__init__` asks one for two of its defaults
    (`stopWordsRemoverGetDefaultOrUS`, and the English stop-word list); `invoke_helper_attr`
    is stubbed so the class still builds, and those two params are recorded as having no
    committable default. See SERVER_SUPPLIED_DEFAULTS.
"""

import argparse
import ast
import importlib
import inspect
import json
import math
import os
import pathlib
import re
import sys
import textwrap

os.environ["SPARK_CONNECT_MODE_ENABLED"] = "1"
os.environ.pop("SPARK_REMOTE", None)

# The modules holding operator classes. `stat` holds no `Estimator`/`Transformer` at all
# (`Summarizer`, `ChiSquareTest` and friends are plain helpers reached through `ConnectHelper`,
# not operators with params), and `image`, `functions`, `torch` and `deepspeed` hold none
# either.
MODULES = [
    "classification",
    "clustering",
    "evaluation",
    "feature",
    "fpm",
    "pipeline",
    "recommendation",
    "regression",
    "tuning",
]

# Not rows, and each for its own reason.
EXCLUDE = {
    # Client-side meta-algorithms. They are not server operators at all: PySpark loops over
    # stages and folds in Python, and so will `latu_ml` (roadmap ML5). A registry row would
    # promise a `Fit` that has no wire form.
    "Pipeline",
    "PipelineModel",
    "CrossValidator",
    "CrossValidatorModel",
    "TrainValidationSplit",
    "TrainValidationSplitModel",
    "OneVsRest",
    "OneVsRestModel",
    # An abstract base that `inspect.isabstract` does not catch: it defines no `_java_obj` and
    # exists only for the six real evaluators to inherit from.
    "JavaEvaluator",
}

# PySpark class name -> Elixir function name, where the mechanical rule reads badly. Both are
# names Spark itself writes solid, and `word2_vec` / `n_gram` would be this package inventing
# a spelling. Everything else goes through `snake/1`.
NAME_OVERRIDES = {
    "Word2Vec": "word2vec",
    "Word2VecModel": "word2vec_model",
    "NGram": "ngram",
}

# `TypeConverters` -> the declared type `Latu.ML.Plan.literal/2` coerces to. The converter is
# PySpark's own client-side coercion and the reason `reg_param: 1` reaches the server as a
# double; `Latu.ML.Plan.param_type/0` is the Elixir side of this table, and a converter absent
# here stops the extractor rather than guessing.
CONVERTERS = {
    "toInt": "int",
    "toFloat": "double",
    "toBoolean": "boolean",
    "toString": "string",
    "toListInt": "{:list, :int}",
    "toListFloat": "{:list, :double}",
    "toListString": "{:list, :string}",
    "toVector": "vector",
    "toMatrix": "matrix",
    # A list of lists of doubles: `Bucketizer.splitsArray`, the only one in 4.2.0. The roadmap
    # said Spark had no nested-list param; it has exactly this one.
    "toListListFloat": "{:list, {:list, :double}}",
    # `identity` means PySpark declares no coercion. One param has it — `RegexTokenizer.gaps`,
    # whose Scala type is `BooleanParam` — so the type is pinned by hand rather than dropped,
    # and an `identity` param that is not in this table stops the extractor.
    "identity": None,
}

IDENTITY_TYPES = {("RegexTokenizer", "gaps"): "boolean"}

# Params whose default a *server* supplies, so no value can be committed. Both are
# `StopWordsRemover`'s, and both are why `invoke_helper_attr` has to be stubbed at all.
SERVER_SUPPLIED_DEFAULTS = {
    ("StopWordsRemover", "locale"),
    ("StopWordsRemover", "stopWords"),
}

# Where a Python module's classes live in Scala. `pyspark.ml.base` mirrors the root
# `org.apache.spark.ml` package; everything else mirrors name for name.
JAVA_ROOT = "org.apache.spark.ml"


# What the stubbed helper answers, per method. Type-correct rather than meaningful: each value
# is thrown away by SERVER_SUPPLIED_DEFAULTS, but PySpark runs it through a `TypeConverters`
# call on the way in, so the wrong shape raises inside `__init__`. A helper call that is not
# here stops the extractor, which is how a future `__init__` that reaches for a server gets
# noticed rather than silently stubbed.
HELPER_STUBS = {
    "stopWordsRemoverGetDefaultOrUS": "",
    "stopWordsRemoverLoadDefaultStopWords": [],
}


def snake(name):
    """PySpark's CamelCase as an Elixir atom.

    A boundary is a lowercase or digit followed by an uppercase, or an uppercase followed by an
    uppercase-then-lowercase — so `LogisticRegression` is `logistic_regression`, `GBTClassifier`
    is `gbt_classifier`, and `LDA` stays `lda`.
    """
    if name in NAME_OVERRIDES:
        return NAME_OVERRIDES[name]
    out = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    out = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", out)
    return out.lower()


def clean_doc(text):
    """PySpark's prose as Markdown, or None.

    The first paragraph only, whitespace collapsed: the rest of a PySpark docstring is
    reStructuredText directives, a Returns section and doctests, none of which belongs in a
    generated `@doc`. Three RST spellings survive into that first paragraph and are rewritten,
    because ExDoc renders Markdown and would otherwise show the markup verbatim:

        `Label <url>`_       ->  [Label](url)
        :py:class:`~a.b.C`   ->  `a.b.C`
        :math:`x`            ->  `x`

    `dev/extract_ml_allowlist.py` imports this rather than keeping a second copy of the rule.
    """
    if not text:
        return None
    paragraph = text.split("\n\n")[0].strip()
    if paragraph.startswith(".."):
        return None
    out = " ".join(paragraph.split())
    out = re.sub(r"`([^`<]+?)\s*<([^>]+)>`_+", r"[\1](\2)", out)
    out = re.sub(r":(?:py:)?\w+:`~?([^`]+)`", r"`\1`", out)
    return out


def _stub_helper():
    """Stop `StopWordsRemover.__init__` reaching a server, without stopping it building."""
    import pyspark.ml.feature as feature
    import pyspark.ml.util as util

    def stub(method, *_args):
        if method not in HELPER_STUBS:
            raise RuntimeError("no stub for helper method %r — see HELPER_STUBS" % method)
        return HELPER_STUBS[method]

    util.invoke_helper_attr = stub
    feature.invoke_helper_attr = stub


def operator_classes():
    """Every concrete operator class PySpark exposes, with its module and kind."""
    from pyspark.ml.base import Estimator, Model, Transformer
    from pyspark.ml.evaluation import Evaluator
    from pyspark.ml.wrapper import JavaParams

    found, skipped = [], []
    for module in MODULES:
        mod = importlib.import_module("pyspark.ml." + module)
        for name in sorted(dir(mod)):
            if name.startswith("_"):
                continue
            obj = getattr(mod, name)
            if not inspect.isclass(obj) or obj.__module__ != mod.__name__:
                continue
            if name in EXCLUDE:
                skipped.append(name)
                continue
            # An abstract class is a Python-side base (`Classifier`, `RegressionModel`); a
            # non-`JavaParams` one is client-side machinery. Neither is a server operator.
            if inspect.isabstract(obj) or not issubclass(obj, JavaParams):
                skipped.append(name)
                continue
            if issubclass(obj, Evaluator):
                kind = "evaluator"
            elif issubclass(obj, Model):
                kind = "model"
            elif issubclass(obj, Estimator):
                kind = "estimator"
            elif issubclass(obj, Transformer):
                kind = "transformer"
            else:
                # A concrete operator that is none of the four: it carries params and a JVM
                # class, and its one method reaches the server through the `ConnectHelper`
                # object rather than through `Fit` or `Transform`. Two on 4.2.0 —
                # `PowerIterationClustering` and `PrefixSpan` — and the shape is read rather
                # than named, so a third arrives as a row in the diff instead of vanishing.
                kind = "helper"
            found.append((module, name, obj, kind))
    return found, skipped


def larger_better(cls):
    """How an evaluator answers `isLargerBetter`, read off its own override.

    The method is **not** on the server's allowlist, so a `Fetch` cannot ask for it. The six
    evaluators know the answer anyway, because each overrides `isLargerBetter` client-side to
    make it work over Connect — and an override is a Python expression this can read rather
    than a value only a running server could give. Three shapes in 4.2.0:

        return True                                    -> True
        return self.getMetricName() in [...]           -> ("only", [...])
        return self.getMetricName() not in [...]       -> ("except", [...])

    Anything else stops the run. A tuning fold picks its best model by maximising or minimising
    this, so a wrong answer here is a silently wrong `bestModel` — the last thing to guess at.
    """
    source = inspect.getsource(cls.isLargerBetter)
    tree = ast.parse(textwrap.dedent(source))
    body = [node for node in tree.body[0].body if not isinstance(node, ast.Expr)]

    if len(body) != 1 or not isinstance(body[0], ast.Return):
        raise SystemExit(
            "%s.isLargerBetter is not a single return; read it and extend larger_better()"
            % cls.__name__
        )

    value = body[0].value

    if isinstance(value, ast.Constant) and isinstance(value.value, bool):
        return value.value

    problem = (
        "%s.isLargerBetter is not one of the three shapes larger_better() reads. It is:\n%s"
        % (cls.__name__, textwrap.indent(source, "    "))
    )

    if not isinstance(value, ast.Compare) or len(value.ops) != 1:
        raise SystemExit(problem)

    # The left side must be `self.getMetricName()` and nothing else: a comparison against some
    # other accessor would mean the same shape saying a different thing.
    left = value.left
    if not (
        isinstance(left, ast.Call)
        and isinstance(left.func, ast.Attribute)
        and left.func.attr == "getMetricName"
        and isinstance(left.func.value, ast.Name)
        and left.func.value.id == "self"
    ):
        raise SystemExit(problem)

    names = value.comparators[0]
    if isinstance(names, (ast.List, ast.Tuple)):
        metrics = [element.value for element in names.elts]
    elif isinstance(names, ast.Constant):
        metrics = [names.value]
    else:
        raise SystemExit(problem)

    if not all(isinstance(metric, str) for metric in metrics):
        raise SystemExit(problem)

    op = value.ops[0]
    if isinstance(op, (ast.In, ast.Eq)):
        return ("only", metrics)
    if isinstance(op, (ast.NotIn, ast.NotEq)):
        return ("except", metrics)
    raise SystemExit(problem)


def java_class(module, name):
    """The JVM class name a Python operator class stands for.

    PySpark's own rule, from `JavaMLReader._java_loader_class`: replace `pyspark` with
    `org.apache.spark` in the module path. Checked against `_java_obj` for every class that
    carries one, which is every estimator, transformer and evaluator — so the rule is only
    *trusted* for models, and only after it has agreed everywhere it could be tested.
    """
    return "%s.%s.%s" % (JAVA_ROOT, module, name)


def model_classes(cls):
    """Every model class an estimator's `_create_model` can return, in the source's own order.

    Read from the source rather than guessed from the name: `QuantileDiscretizer` fits a
    `Bucketizer`, and nothing about the names says so.

    Forty-one of the forty-two return one class. `LDA` returns two, branching on a param the
    caller set:

        def _create_model(self, java_model):
            if self.getOptimizer() == "em":
                return DistributedLDAModel(java_model)
            else:
                return LocalLDAModel(java_model)

    A list, therefore, rather than a name and a `None` — the same shape and the same reason as
    `_summaryCls` in `dev/extract_ml_allowlist.py`. `Latu.ML.Registry` collapses a one-element
    list back to a single `model_class`, so nothing downstream of it changes for the forty-one.
    """
    create = getattr(cls, "_create_model", None)
    if create is None:
        return []
    try:
        source = inspect.getsource(create)
    except (OSError, TypeError):
        return []
    found = re.findall(r"return\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", source)
    return list(dict.fromkeys(found))


def param_rows(cls, instance, twin):
    """One row per param the class declares.

    Params are read off the *class* with `getattr_static`, never off the instance:
    `Params.params` calls `getattr` over `dir(self)`, and PySpark has `cached_property`
    attributes (`RandomForestClassificationModel.trees`) that its property filter misses and
    that reach for a server when touched.
    """
    from pyspark.ml.param import Param

    defaults = {p.name: v for p, v in instance._defaultParamMap.items()}
    others = {p.name: v for p, v in twin._defaultParamMap.items()}
    uid = instance.uid

    rows, problems = [], []
    seen = {}
    for attr in sorted(dir(cls)):
        try:
            value = inspect.getattr_static(cls, attr)
        except AttributeError:
            continue
        if not isinstance(value, Param):
            continue

        wire = value.name
        converter = getattr(value.typeConverter, "__name__", str(value.typeConverter))
        if converter not in CONVERTERS:
            problems.append("%s.%s: unknown converter %s" % (cls.__name__, wire, converter))
            continue
        declared = CONVERTERS[converter]
        if declared is None:
            declared = IDENTITY_TYPES.get((cls.__name__, wire))
            if declared is None:
                problems.append(
                    "%s.%s: converter is `identity` and no type is pinned for it"
                    % (cls.__name__, wire)
                )
                continue

        name = snake(wire)
        if name in seen and seen[name] != wire:
            problems.append(
                "%s: %r and %r both spell %r" % (cls.__name__, seen[name], wire, name)
            )
            continue
        seen[name] = wire

        rows.append(
            {
                "name": name,
                "wire": wire,
                "type": declared,
                "default": default_of(cls, wire, defaults, others, uid),
                "doc": clean_doc(value.doc),
            }
        )
    return rows, problems


def default_of(cls, wire, defaults, others, uid):
    """A param's default, in a form that is the same on every machine.

    Defaults are documentation and never reach the wire (`Latu.ML.Plan.params/1`), so the only
    requirement is that they read true and do not move. Three kinds move, and each gets a
    marker rather than a value that would be a lie in the next run's diff:

      * an output column defaults to the operator's own random uid — `{:uid_suffix, s}`;
      * `StopWordsRemover`'s locale and stop-word list come from a server — `:server_supplied`;
      * anything unstable — `:varies`. Two ways in: a default that differs between two
        instances of the same class, and a default that *equals* `hash(class name)`, which is
        how `HasSeed` builds one. The second needs its own test because Python salts string
        hashing per process, not per instance: two instances agree, two runs do not. There is
        nothing true to write for `seed` anyway — the server's own default is the JVM hash of
        the Scala class name, a different number again.

    `others` is a second instance of the same class; between it and the hash test, neither
    check is a list of param names to keep up to date.
    """
    if (cls.__name__, wire) in SERVER_SUPPLIED_DEFAULTS:
        return {"marker": "server_supplied"}
    if wire not in defaults:
        return None
    value = defaults[wire]
    if isinstance(value, str) and value.startswith(uid):
        return {"uid_suffix": value[len(uid) :]}
    if not same(others.get(wire, value), value) or value == hash(cls.__name__):
        return {"marker": "varies"}
    return value


def same(left, right):
    """Equality that calls two NaNs the same value.

    `Imputer.missingValue` defaults to NaN, and `nan != nan`, so plain `==` would report the
    one genuinely stable NaN in Spark's param tables as unstable.
    """
    if isinstance(left, float) and isinstance(right, float):
        return left == right or (math.isnan(left) and math.isnan(right))
    return left == right


def collect():
    """Every operator row, plus whatever went wrong building them."""
    _stub_helper()

    classes, skipped = operator_classes()
    rows, problems = [], []

    for module, name, cls, kind in classes:
        try:
            instance, twin = cls(), cls()
        except Exception as exc:  # noqa: BLE001 - the reason is reported, not swallowed
            problems.append("%s: cannot instantiate (%s: %s)" % (name, type(exc).__name__, exc))
            continue

        expected = java_class(module, name)
        actual = getattr(instance, "_java_obj", None)
        if isinstance(actual, str) and actual != expected:
            problems.append(
                "%s: _java_obj is %s, the module rule says %s" % (name, actual, expected)
            )
            continue

        params, trouble = param_rows(cls, instance, twin)
        problems.extend(trouble)

        models = model_classes(cls)
        row = {
            "name": snake(name),
            "python": name,
            "group": module,
            "kind": kind,
            "class": expected,
            "doc": clean_doc(inspect.getdoc(cls)),
            "params": params,
        }
        # Only an evaluator has one, and every evaluator must: `larger_better` raises rather
        # than falling back on the base class's `return True`, because a default here would be
        # a guess that silently picks the wrong `bestModel`.
        if kind == "evaluator":
            row["larger_better"] = larger_better(cls)
        if models:
            row["model_classes"] = [
                java_class(module_of(model, module), model) for model in models
            ]
        rows.append(row)

    names = {}
    for row in rows:
        names.setdefault(row["name"], []).append(row["python"])
    for name, owners in sorted(names.items()):
        if len(owners) > 1:
            problems.append("%r is the name of %s" % (name, " and ".join(owners)))

    return sorted(rows, key=lambda r: (r["group"], r["name"])), skipped, problems


_MODULE_OF = {}


def module_of(class_name, fallback):
    """Which `pyspark.ml` module defines a class, for a model named by `_create_model`."""
    if not _MODULE_OF:
        for module in MODULES:
            mod = importlib.import_module("pyspark.ml." + module)
            for attr in dir(mod):
                obj = getattr(mod, attr)
                if inspect.isclass(obj) and obj.__module__ == mod.__name__:
                    _MODULE_OF[attr] = module
    return _MODULE_OF.get(class_name, fallback)


# =============================================
# Rendering
# =============================================


def elixir_integer(value):
    """A Python int as `mix format` writes it: digits grouped in threes past five digits."""
    digits = str(abs(value))
    sign = "-" if value < 0 else ""
    if len(digits) <= 5:
        return sign + digits
    head = len(digits) % 3 or 3
    groups = [digits[:head]] + [digits[i : i + 3] for i in range(head, len(digits), 3)]
    return sign + "_".join(groups)


def elixir_float(value):
    """A Python float as an Elixir float literal.

    Python and Elixir disagree twice. `repr(1e-06)` is `1e-06`, which Elixir will not parse —
    a float literal needs digits either side of the point and a bare exponent. And NaN is a
    value `Imputer.missingValue` really defaults to, with no literal form at all, so it is an
    atom the docs render as `NaN`; it is documentation, and never reaches the wire.
    """
    if math.isnan(value):
        return ":nan"
    if math.isinf(value):
        return ":infinity" if value > 0 else ":negative_infinity"

    mantissa, _, exponent = repr(value).partition("e")
    if "." not in mantissa:
        mantissa += ".0"
    rendered = mantissa if not exponent else "%se%d" % (mantissa, int(exponent))
    if float(rendered) != value:
        raise ValueError("%r does not round-trip as %s" % (value, rendered))
    return rendered


def elixir(value):
    """A Python value as Elixir source. Only what a Spark param default can be."""
    if value is None:
        return "nil"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        return elixir_integer(value)
    if isinstance(value, float):
        return elixir_float(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False).replace("#{", "\\#{")
    if isinstance(value, dict) and "marker" in value:
        return ":" + value["marker"]
    if isinstance(value, dict) and "uid_suffix" in value:
        return "{:uid_suffix, %s}" % elixir(value["uid_suffix"])
    if isinstance(value, (list, tuple)):
        return "[%s]" % ", ".join(elixir(v) for v in value)
    raise ValueError("cannot render %r as an Elixir literal" % (value,))


# `mix format`'s line length, from .formatter.exs. Emitting what the formatter would emit is
# not tidiness: `mix check` runs `format --check-formatted`, and a generated file that needs
# reformatting turns every regeneration into a two-step dance with a diff in between.
LINE_LENGTH = 98


def class_list(key, values, indent):
    """`key: ["a", "b"],` on one line, or one per line where `mix format` would break it."""
    pad = " " * indent
    single = "%s%s: [%s]," % (pad, key, ", ".join('"%s"' % v for v in values))
    if len(single) <= LINE_LENGTH:
        return single + "\n"
    inner = ",\n".join('%s  "%s"' % (pad, v) for v in values)
    return "%s%s: [\n%s\n%s],\n" % (pad, key, inner, pad)


def pair(pad, key, value, trailing=","):
    """`key: value,` wrapped the way `mix format` wraps it when the line is too long.

    The trailing comma counts towards the line length, so it is part of the measurement rather
    than something the caller sticks on afterwards.
    """
    line = "%s%s: %s%s" % (pad, key, value, trailing)
    if len(line) <= LINE_LENGTH:
        return line
    return "%s%s:\n%s  %s%s" % (pad, key, pad, value, trailing)


def elixir_larger_better(row):
    """`true`, or `{:only, [...]}` / `{:except, [...]}` keyed on the metric name."""
    value = row["larger_better"]
    if isinstance(value, bool):
        return "true" if value else "false"
    shape, metrics = value
    return "{:%s, [%s]}" % (shape, ", ".join('"%s"' % metric for metric in metrics))


def render_param(param, indent):
    pad = " " * indent
    inner = pad + "  "
    lines = [
        "%s%%{" % pad,
        pair(inner, "name", ":" + param["name"]),
        pair(inner, "wire", '"%s"' % param["wire"]),
        pair(inner, "type", atom_type(param["type"])),
        pair(inner, "default", elixir(param["default"])),
        pair(inner, "doc", elixir(param["doc"]), trailing=""),
        "%s}" % pad,
    ]
    return "\n".join(lines)


def atom_type(declared):
    return declared if declared.startswith("{") else ":" + declared


def render(rows, version):
    out = [
        "# Generated by dev/extract_ml_operators.py from PySpark %s. Do not hand-edit.\n"
        % version,
        "#\n",
        "# Every row is read off a `pyspark.ml` class: its params, their converters, their\n",
        "# defaults and their docs. Regenerate with:\n",
        "#\n",
        "#     python dev/extract_ml_operators.py --write\n",
        "#\n",
        "# An evaluator row carries `:larger_better`, read off its own client-side\n",
        "# `isLargerBetter` override: `true`, or `{:only, metrics}` / `{:except, metrics}` keyed\n",
        "# on `metric_name`. The server's allowlist has no such method, so this is the only\n",
        "# place the answer exists — and a tuning fold picks its best model by it.\n",
        "#\n",
        "# `:default` is documentation and never sent — see `Latu.ML.Plan.params/1`. Four have\n",
        "# no plain literal: `{:uid_suffix, s}` is derived from the operator's own uid,\n",
        "# `:server_supplied` is one only a running server knows, `:varies` is one that is not\n",
        "# the same twice (`seed`), and `:nan` is NaN.\n",
        "%{\n",
        '  spark: "%s",\n' % version,
        "  operators: [\n",
    ]
    for row in rows:
        out.append("    %{\n")
        out.append("      name: :%s,\n" % row["name"])
        out.append('      python: "%s",\n' % row["python"])
        out.append("      group: :%s,\n" % row["group"])
        out.append("      kind: :%s,\n" % row["kind"])
        out.append(pair("      ", "class", '"%s"' % row["class"]) + "\n")
        if "model_classes" in row:
            out.append(class_list("model_classes", row["model_classes"], 6))
        if "larger_better" in row:
            out.append(pair("      ", "larger_better", elixir_larger_better(row)) + "\n")
        out.append(pair("      ", "doc", elixir(row["doc"])) + "\n")
        if row["params"]:
            out.append("      params: [\n")
            out.append(",\n".join(render_param(p, 8) for p in row["params"]))
            out.append("\n      ]\n")
        else:
            out.append("      params: []\n")
        out.append("    },\n")
    if rows:
        out[-1] = "    }\n"
    out.append("  ]\n")
    out.append("}\n")
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pyspark", type=pathlib.Path, help="path to the pyspark package's parent")
    ap.add_argument("--json", action="store_true", help="dump rows as JSON, not a summary")
    ap.add_argument("--write", action="store_true", help="rewrite priv/ml_operators.exs")
    ap.add_argument("--check", action="store_true", help="exit 1 if that file is out of date")
    args = ap.parse_args()

    here = pathlib.Path(__file__).resolve().parent
    root = here.parent

    if args.pyspark is not None:
        sys.path.insert(0, str(args.pyspark))
    else:
        # Latu's venv, beside this repo or named by LATU_DEV, is the pinned PySpark.
        latu_dev = os.environ.get("LATU_DEV")
        candidates = [pathlib.Path(latu_dev)] if latu_dev else [root.parent / "latu" / "dev"]
        for candidate in candidates:
            found = list(candidate.glob(".venv/lib/python*/site-packages/pyspark"))
            if found:
                sys.path.insert(0, str(found[0].parent))
                break
        else:
            sys.exit(
                "no pyspark found — check out latu beside this repo, set LATU_DEV, "
                "or pass --pyspark"
            )

    import pyspark

    version = pyspark.__version__
    rows, skipped, problems = collect()

    if problems:
        sys.exit("\n".join(["the registry is not clean:"] + ["  " + p for p in problems]))

    if args.json:
        json.dump(rows, sys.stdout, indent=1)
        return

    target = root / "priv" / "ml_operators.exs"
    source = render(rows, version)

    if args.check:
        current = target.read_text() if target.exists() else ""
        if current != source:
            sys.exit(
                "%s is out of date — run: python dev/extract_ml_operators.py --write" % target
            )
        print("%s is current for PySpark %s" % (target, version))
        return

    if args.write:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source)
        print("%d operators -> %s (run `mix format` next)" % (len(rows), target))
        return

    kinds = {}
    for row in rows:
        kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1
    params = sum(len(row["params"]) for row in rows)
    print("PySpark %s: %d operators, %d params" % (version, len(rows), params))
    for kind in sorted(kinds):
        print("  %4d  %s" % (kinds[kind], kind))
    print("\nnot rows: %s" % ", ".join(sorted(skipped)))


if __name__ == "__main__":
    main()
