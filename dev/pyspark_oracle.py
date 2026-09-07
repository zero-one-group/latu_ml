#!/usr/bin/env python3
"""PySpark as a proto oracle for Latu ML.

Latu's oracle evaluates an expression and prints its plan. ML has no expressions: a `Fit` and a
`Fetch` are *commands*, and PySpark only builds one on its way to sending it. So this runs
PySpark's own `fit`/`transform`/attribute call with `client.execute_command` monkeypatched to
capture the command and raise — the proto PySpark *would* have sent is the fixture, and nothing
about how PySpark builds one is re-implemented here.

    export SPARK_REMOTE='sc://localhost:15003'
    python dev/pyspark_oracle.py --generate
    python dev/pyspark_oracle.py --show ml_fit_logistic_regression

The normalisers are imported from Latu's oracle rather than copied: `normalize_plan_ids`
already has a twin in `Latu.Plan.normalize_ids/1`, and a third implementation of one rule is
how they drift. So **Latu must be checked out beside this repo**, or `LATU_DEV` set to its
`dev/` directory.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import pathlib
import signal
import sys

from google.protobuf import text_format

OUT_DIR = pathlib.Path("test/wire")

# Pinned so both sides build the same bytes. PySpark generates a random uid per operator and
# Latu ML does the same; a fixture cannot have either.
ASSEMBLER_UID = "latu_ml_fixture_assembler"
ESTIMATOR_UID = "latu_ml_fixture_estimator"
EVALUATOR_UID = "latu_ml_fixture_evaluator"
MODEL_REF = "latu_ml_fixture_model"

# Where a Write goes and a Read comes from. Nothing is written when a fixture is generated —
# the command is captured before it is sent — so this path need not exist anywhere.
SAVE_PATH = "/tmp/latu_ml_fixture/saved"

# Where the two clients hand each other a saved model. On the **server's** filesystem: a write
# happens where the session is. See `--interop` below and `dev/README.md`.
INTEROP_ROOT = "/tmp/latu_ml_interop"

# A model a helper builds gets its uid from `Identifiable._randomUID`, inside the classmethod
# and out of reach of `_resetUid`. Pinning the generator is the same move `_resetUid` is for
# every other fixture: a random value held still, not a call site re-implemented.
HELPER_UID = "latu_ml_fixture_helper"


def latu_oracle():
    """Latu's `dev/pyspark_oracle.py`, loaded by path so our own name cannot shadow it."""
    default = pathlib.Path(__file__).resolve().parents[2] / "latu" / "dev"
    dev = pathlib.Path(os.environ.get("LATU_DEV", default))
    path = dev / "pyspark_oracle.py"

    if not path.exists():
        sys.exit(
            f"Latu's oracle is not at {path}.\n"
            "  Check Latu out beside this repo, or set LATU_DEV to its dev/ directory.\n"
            "  The normalisers live there so there is one implementation, not three."
        )

    spec = importlib.util.spec_from_file_location("latu_oracle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def get_session():
    """Connect via SPARK_REMOTE, or exit with instructions."""
    if not os.environ.get("SPARK_REMOTE"):
        sys.exit(
            "SPARK_REMOTE is not set.\n"
            "  export SPARK_REMOTE='sc://localhost:15003'\n"
            "and check the server is up: docker compose -f ../latu/docker-compose.yml up -d"
        )
    from pyspark.sql import SparkSession

    return SparkSession.builder.getOrCreate()


def silence_deletes():
    """Stop a garbage collection from sending a `Delete` to the server.

    `JavaWrapper.__del__` is decorated `try_remote_del`, so when a `LogisticRegressionModel`
    built here is collected, `RemoteModelRef.release_ref` fires `del_remote_cache` — a real RPC,
    for a ref that never existed, at a moment Python chooses. It can land in the middle of a
    `capture`, and it is the one thing this script promises never to do. So it is a no-op for
    the length of a run.
    """
    import pyspark.ml.util as util

    util.del_remote_cache = lambda _ref_id: None


def pin_uids():
    """Hold `Identifiable._randomUID` still, for a model a classmethod builds out of reach."""
    from pyspark.ml.util import Identifiable

    Identifiable._randomUID = classmethod(lambda cls: HELPER_UID)


class Captured(Exception):
    """The command PySpark was about to send. Raised instead of sending it."""

    def __init__(self, command):
        super().__init__("captured")
        self.command = command


def capture(client, thunk):
    """Run `thunk`, intercept the one command it sends, and hand it back unsent.

    The monkeypatch goes on the *instance*, so it is undone by deleting the attribute rather
    than by rebinding the method. Anything PySpark wraps the call in — the summary-recovery
    retry in `try_remote_call`, for one — catches `SparkException` and not this.
    """

    def intercepted(command, *_args, **_kwargs):
        raise Captured(command)

    client.execute_command = intercepted
    try:
        thunk()
    except Captured as captured:
        return captured.command
    finally:
        del client.execute_command

    raise RuntimeError("nothing was sent; PySpark did not reach execute_command")


def plan_of(command):
    import pyspark.sql.connect.proto as proto

    plan = proto.Plan()
    plan.command.CopyFrom(command)

    return plan


def root_of(relation):
    import pyspark.sql.connect.proto as proto

    plan = proto.Plan()
    plan.root.CopyFrom(relation)

    return plan


def fixtures(spark):
    """Every fixture, as name -> a function returning the message to write.

    One entry here is one golden assertion in Elixir; keep the names matching.
    """
    import pyspark.sql.connect.proto as proto  # noqa: F401
    from pyspark.ml.classification import (
        LinearSVCModel,
        LogisticRegression,
        LogisticRegressionModel,
    )
    from pyspark.ml.clustering import KMeans, PowerIterationClustering
    from pyspark.ml.connect.serialize import serialize_param
    from pyspark.ml.evaluation import RegressionEvaluator
    from pyspark.ml.feature import (
        Binarizer,
        Bucketizer,
        StandardScaler,
        StopWordsRemover,
        StringIndexerModel,
        VectorAssembler,
    )
    from pyspark.ml.fpm import FPGrowth
    from pyspark.ml.functions import array_to_vector, vector_to_array
    from pyspark.ml.linalg import DenseMatrix, DenseVector, SparseVector
    from pyspark.ml.recommendation import ALS, ALSModel
    from pyspark.ml.regression import LinearRegression
    from pyspark.ml.stat import ChiSquareTest
    from pyspark.ml.util import RemoteModelRef
    from pyspark.sql.functions import col

    client = spark.client

    # The one base frame, and the only thing both sides must build identically.
    def base():
        return spark.range(0, 4, 1)

    def assembler():
        made = VectorAssembler(inputCols=["id"], outputCol="features")
        made._resetUid(ASSEMBLER_UID)
        return made

    def estimator(**params):
        made = LogisticRegression(**params)
        made._resetUid(ESTIMATOR_UID)
        return made

    def model():
        made = LogisticRegressionModel(RemoteModelRef(MODEL_REF))
        made._resetUid(ESTIMATOR_UID)
        return made

    # A training summary has no cache key of its own: PySpark composes the string
    # `"<model_ref>.summary"` and `_extract_id_methods` splits it back into an obj_ref and a
    # leading `summary` method. `LinearSVCModel` rather than `LogisticRegressionModel` because
    # its `_summaryCls` is one class rather than two chosen by the fitted model's `numClasses`,
    # which is a server round trip this script must not make.
    def summary():
        model = LinearSVCModel(RemoteModelRef(MODEL_REF))
        model._resetUid(ESTIMATOR_UID)

        made = model._summaryCls(f"{MODEL_REF}.summary")
        made._summary_dataset = base()
        made._remote_model_obj = model._java_obj
        # `try_remote_fit` does this, and without it the model and the summary both release the
        # one `RemoteModelRef` on the way out and the second one asserts in `__del__`.
        made._remote_model_obj.add_ref()

        return made

    # A model with a param set, for the Write fixtures: a saved model carries the values its
    # estimator was given, and `serialize_ml_params` reads them off `_paramMap`.
    def saved_model():
        made = model()
        made._set(maxIter=7)

        return made

    # For the relation arm of an argument: `recommendForUserSubset` takes a DataFrame *and* an
    # int, so its `Fetch.Method.Args` carries one `input` and one `param`. Nothing else in this
    # file pins the `input` arm.
    def als_model():
        made = ALSModel(RemoteModelRef(MODEL_REF))
        made._resetUid(ESTIMATOR_UID)
        return made

    def fit(**params):
        made = estimator(**params)
        assembled = assembler().transform(base())

        return plan_of(capture(client, lambda: made.fit(assembled)))

    def fit_any(cls, **params):
        """A `Fit` for any estimator, on the bare base frame.

        The command is captured before it is sent, so nothing checks the frame against the
        estimator's expectations — which is why an `ALS` fixture does not need a ratings frame.
        The fixture is about the operator, its params and its uid, and about nothing else.
        """
        made = cls(**params)
        made._resetUid(ESTIMATOR_UID)

        return plan_of(capture(client, lambda: made.fit(base())))

    def transform_any(cls, **params):
        made = cls(**params)
        made._resetUid(ASSEMBLER_UID)

        return root_of(made.transform(base())._plan.plan(client))

    def evaluate(cls, **params):
        made = cls(**params)
        made._resetUid(EVALUATOR_UID)

        return plan_of(capture(client, lambda: made.evaluate(base())))

    def select_expr(column):
        """A projection of one built column. The only fixtures that are not ML messages."""
        return root_of(base().select(column.alias("v"))._plan.plan(client))

    return {
        # Commands. `maxIter` is an int, `regParam` and `threshold` doubles, `fitIntercept` a
        # boolean and `featuresCol` a string — one fixture covering every literal arm the
        # registry can produce, which is the claim `Latu.ML.Plan.literal/2` most needs checked.
        "ml_fit_logistic_regression": lambda: fit(
            maxIter=7,
            regParam=0.25,
            threshold=0.75,
            fitIntercept=False,
            featuresCol="features",
        ),
        # Nothing set: the params map must be empty, not full of defaults.
        "ml_fit_defaults": lambda: fit(),
        # One estimator per family, because a generated constructor is only as good as the
        # class name and param spellings the registry gave it, and those are per operator. Each
        # sets a param whose declared type differs from its neighbours'.
        "ml_fit_linear_regression": lambda: fit_any(
            LinearRegression, maxIter=5, elasticNetParam=0.5
        ),
        "ml_fit_kmeans": lambda: fit_any(KMeans, k=3, distanceMeasure="cosine"),
        "ml_fit_standard_scaler": lambda: fit_any(StandardScaler, withMean=True),
        "ml_fit_fp_growth": lambda: fit_any(FPGrowth, minSupport=0.4, itemsCol="items"),
        "ml_fit_als": lambda: fit_any(ALS, rank=5, implicitPrefs=True),
        # The two ML functions. Not operators and not commands: Spark registers them in its
        # *internal* function registry, so what goes over the wire is a plain
        # `UnresolvedFunction` with `is_internal` left unset, and these pin that shape. The
        # dtype is an ordinary string literal argument, not a second function name.
        "ml_functions_vector_to_array": lambda: select_expr(vector_to_array(col("features"))),
        "ml_functions_vector_to_array_float32": lambda: select_expr(
            vector_to_array(col("features"), "float32")
        ),
        "ml_functions_array_to_vector": lambda: select_expr(array_to_vector(col("a"))),
        # An evaluator's command, which `Latu.ML.evaluate/2` will send once ML4 gives it a verb.
        # The wire form is settled now, so the verb has only the result left to get wrong.
        "ml_evaluate_regression": lambda: evaluate(RegressionEvaluator, metricName="mae"),
        # Persistence, one fixture per arm of each command. A model is written by **reference**
        # and an unfitted operator by **class and uid**, which is the whole difference between
        # the two arms of `MlCommand.Write` — and `should_overwrite` is set even when false,
        # because PySpark sets it unconditionally and the field carries explicit presence.
        "ml_write_model": lambda: plan_of(
            capture(client, lambda: saved_model().write().save(SAVE_PATH))
        ),
        # The other arm, and the two optional halves: overwrite, and a writer option. PySpark
        # lowercases an option key on the way in; the key here is already lower so that the
        # fixture is about the map and not about the casing.
        "ml_write_estimator": lambda: plan_of(
            capture(
                client,
                lambda: estimator(maxIter=7)
                .write()
                .overwrite()
                .option("compression", "gzip")
                .save(SAVE_PATH),
            )
        ),
        # A `Read` names a class rather than an instance, so there is no uid to send — the one
        # in the saved metadata comes back in the answer.
        "ml_read_model": lambda: plan_of(
            capture(client, lambda: LogisticRegressionModel.load(SAVE_PATH))
        ),
        "ml_read_estimator": lambda: plan_of(
            capture(client, lambda: LogisticRegression.load(SAVE_PATH))
        ),
        # The `ConnectHelper` route: one fixture per shape it answers in. The object is named
        # by a literal id rather than a cache entry, and its arguments are positional — every
        # one is sent, where an `MlParams` map carries only what the caller set.
        #
        # A **value**: one string in, a list of strings back, and a command because it is not
        # a relation.
        "ml_helper_stop_words": lambda: plan_of(
            capture(client, lambda: StopWordsRemover.loadDefaultStopWords("german"))
        ),
        # A **model**: the server builds a `StringIndexerModel` out of the labels and answers
        # with its reference. The uid is the client's and is the call's first argument, which
        # is why `pin_uids` exists at all.
        "ml_helper_from_labels": lambda: plan_of(
            capture(
                client,
                lambda: StringIndexerModel.from_labels(["a", "b"], inputCol="category"),
            )
        ),
        # The same, with the one **nested list** in the whole surface — and it is typed
        # explicitly at the call site, `ArrayType(ArrayType(StringType()))`, because a list of
        # lists cannot be inferred from its value.
        "ml_helper_from_arrays_of_labels": lambda: plan_of(
            capture(
                client,
                lambda: StringIndexerModel.from_arrays_of_labels(
                    [["a", "b"], ["c"]], inputCols=["one", "two"]
                ),
            )
        ),
        # A **relation**: a DataFrame argument beside three literals, and nothing sent. Unlike
        # a summary's frame this carries no `model_summary_dataset` — there is no model.
        "ml_helper_chi_square_test": lambda: root_of(
            ChiSquareTest.test(base(), "features", "label", True)._plan.plan(client)
        ),
        # The operator arm: `PowerIterationClustering` is neither fitted nor applied, and its
        # params ride the call as **positional arguments**. Two are set here and the rest are
        # Spark's defaults, which is the whole of what `Latu.ML.Internal.operator_values!/3`
        # has to reproduce — including the empty string PySpark sends for an unset `weightCol`.
        "ml_helper_assign_clusters": lambda: root_of(
            PowerIterationClustering(k=3, maxIter=7)
            .assignClusters(base())
            ._plan.plan(client)
        ),
        # Not a helper: the *inferred* nested list, which is the other half of the same
        # question. `splitsArray` is the one nested param in 4.2.0 and it goes through
        # `_from_value` rather than through an explicit `ArrayType`, so this is what says the
        # two paths write the same bytes.
        "ml_transform_bucketizer_splits_array": lambda: transform_any(
            Bucketizer,
            splitsArray=[[-1.0e9, 0.0, 1.0e9], [-1.0e9, 1.0, 1.0e9]],
            inputCols=["x1", "x2"],
            outputCols=["b1", "b2"],
        ),
        # Arguments on a `Fetch`, one per arm. A `Vector` is the widest literal the plan layer
        # builds, and `predict` is where it lands most often.
        "ml_fetch_predict": lambda: plan_of(
            capture(client, lambda: model().predict(DenseVector([1.5, -2.5])))
        ),
        # A double argument, and one that arrives *after* a chain: the object is the model,
        # `summary` leads, and the argument hangs off the last method rather than the first.
        "ml_fetch_f_measure_by_label": lambda: plan_of(
            capture(client, lambda: summary().fMeasureByLabel(0.5))
        ),
        "ml_fetch_coefficients": lambda: plan_of(
            capture(client, lambda: model().coefficients)
        ),
        "ml_fetch_intercept": lambda: plan_of(capture(client, lambda: model().intercept)),
        # Through a summary: the object is the model, and `summary` leads the method chain.
        "ml_fetch_summary": lambda: plan_of(
            capture(client, lambda: summary().totalIterations)
        ),
        # Relations. A transform is lazy, so there is nothing to intercept.
        "ml_transform_assembler": lambda: root_of(
            assembler().transform(base())._plan.plan(client)
        ),
        "ml_transform_model": lambda: root_of(
            model().transform(assembler().transform(base()))._plan.plan(client)
        ),
        # A second transformer, for a param that is neither a string nor a list of them:
        # `threshold` is a double, and `VectorAssembler` has nothing of that shape to check.
        "ml_transform_binarizer": lambda: transform_any(
            Binarizer, threshold=0.5, inputCol="id", outputCol="above"
        ),
        # A summary attribute that answers with a frame. Unlike a model's own, this relation
        # carries `model_summary_dataset` — the server can rebuild a dropped summary from it
        # while it plans, which is why the command path needs an explicit retry and this
        # does not.
        "ml_summary_predictions": lambda: root_of(
            summary().predictions._plan.plan(client)
        ),
        # An argument that is a whole relation. `recommendForUserSubset(dataset, numItems)` is
        # the only shape that puts both arms in one method's argument list, which is what makes
        # it worth a fixture the other two are not.
        "ml_relation_recommend_for_user_subset": lambda: root_of(
            als_model().recommendForUserSubset(base(), 5)._plan.plan(client)
        ),
        # Bare literals, not plans: `Latu.ML.Linalg`'s output, element for element.
        "ml_literal_dense_vector": lambda: serialize_param(
            DenseVector([1.5, -2.5]), client
        ),
        "ml_literal_sparse_vector": lambda: serialize_param(
            SparseVector(4, [1, 3], [2.0, 4.0]), client
        ),
        # Row-major, which is what `isTransposed=True` means and what an Nx tensor hands over.
        "ml_literal_dense_matrix": lambda: serialize_param(
            DenseMatrix(2, 2, [1.0, 2.0, 3.0, 4.0], True), client
        ),
        # Non-square, so numRows and numCols are pinned in the right order — the square fixture
        # above cannot tell them apart.
        "ml_literal_dense_matrix_wide": lambda: serialize_param(
            DenseMatrix(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], True), client
        ),
    }


def interop(spark):
    """Save a model where Latu can read it, and read the one Latu saved. Returns an exit code.

    The other half is `mix run dev/probe_ml.exs -- --interop`, and `dev/README.md` has the
    order to run them in. Both sides fit the **same** estimator on the **same** rows against
    the **same** server, so their coefficients agree to the last bit and nothing has to be
    carried between them but the saved model itself. A difference here is therefore one thing
    only: a format two clients read differently.
    """
    from pyspark.ml import Pipeline, PipelineModel
    from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
    from pyspark.ml.evaluation import RegressionEvaluator
    from pyspark.ml.regression import LinearRegression
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.tuning import (
        CrossValidator,
        CrossValidatorModel,
        ParamGridBuilder,
        TrainValidationSplit,
    )

    rows = spark.sql(
        """
        SELECT
          CAST(label AS DOUBLE) AS label,
          CAST(x1 AS DOUBLE) AS x1,
          CAST(x2 AS DOUBLE) AS x2
        FROM VALUES (0.0, 0.0, 1.1), (0.0, 0.1, 1.0), (1.0, 2.0, 0.1), (1.0, 2.1, 0.2)
        AS t(label, x1, x2)
        """
    )
    features = VectorAssembler(inputCols=["x1", "x2"], outputCol="features").transform(rows)
    model = LogisticRegression(maxIter=5, regParam=0.01).fit(features)

    mine = list(model.coefficients)
    print(f"  fitted here                 {mine}")

    model.write().overwrite().save(f"{INTEROP_ROOT}/pyspark")
    print(f"  saved for Latu              {INTEROP_ROOT}/pyspark")

    failures = _read_back(
        "Latu's model, read here",
        lambda: list(LogisticRegressionModel.load(f"{INTEROP_ROOT}/latu").coefficients),
        mine,
    )

    # ML5a. A pipeline directory is laid out by the *client*, so this is the one part of the
    # format neither side gets from the server: PySpark writes `metadata` and `stages/NN_uid/`
    # in Python, Latu writes them in Elixir, and only a cross-read says they agree. The stage
    # directories underneath are the JVM's own work in both cases, which is exactly why a
    # difference here can only be the layout around them.
    stages = [
        VectorAssembler(inputCols=["x1", "x2"], outputCol="features"),
        LogisticRegression(maxIter=5, regParam=0.01),
    ]
    fitted = Pipeline(stages=stages).fit(rows)
    theirs_expected = list(fitted.stages[-1].coefficients)

    if theirs_expected != mine:
        print(f"  pipeline fit here           DIFFER from the plain fit — {theirs_expected}")
        failures += 1
    else:
        print("  pipeline fit here           identical to the plain fit")

    fitted.write().overwrite().save(f"{INTEROP_ROOT}/pyspark_pipeline")
    print(f"  saved for Latu              {INTEROP_ROOT}/pyspark_pipeline")

    failures += _read_back(
        "Latu's pipeline, read here",
        lambda: list(PipelineModel.load(f"{INTEROP_ROOT}/latu_pipeline").stages[-1].coefficients),
        mine,
    )

    # ML5b. A search has no wire form at all: the fold cut, the grid order, the mean across
    # folds and the argmax are every client's own arithmetic. So there is nothing to save and
    # read back here — the comparison is the *numbers*, printed side by side with the ones
    # `mix run dev/probe_ml.exs` prints for the same grid, seed and rows against this server.
    # The **same twelve rows** `dev/probe_ml.exs` builds, written out the same way. A grid
    # search only means something if the metrics differ across it: on four rows every
    # `regParam` scored identically, and three ties make index 0 the answer either way.
    tuning_rows = spark.sql(
        """
        SELECT CAST(y AS DOUBLE) AS label, CAST(x1 AS DOUBLE) AS x1, CAST(x2 AS DOUBLE) AS x2
        FROM VALUES
          (1.0, 0.0, 0.1), (2.1, 1.0, 0.2), (2.9, 2.0, 0.1), (4.2, 3.0, 0.3),
          (5.1, 4.0, 0.2), (5.8, 5.0, 0.4), (7.2, 6.0, 0.1), (8.1, 7.0, 0.3),
          (8.8, 8.0, 0.2), (10.3, 9.0, 0.5), (11.1, 10.0, 0.1), (11.9, 11.0, 0.4)
        AS t(y, x1, x2)
        """
    )
    tuning_frame = VectorAssembler(
        inputCols=["x1", "x2"], outputCol="features"
    ).transform(tuning_rows)

    tuning_lr = LinearRegression(maxIter=5)
    tuning_grid = ParamGridBuilder().addGrid(tuning_lr.regParam, [0.5, 0.0, 5.0]).build()
    tuning_eval = RegressionEvaluator()

    cv = CrossValidator(
        estimator=tuning_lr,
        estimatorParamMaps=tuning_grid,
        evaluator=tuning_eval,
        numFolds=2,
        seed=42,
    )
    cv_model = cv.fit(tuning_frame)

    print(f"  cv avg metrics              {list(cv_model.avgMetrics)}")
    print(f"  cv std metrics              {list(cv_model.stdMetrics)}")
    print(f"  cv best index               {_best_index(cv_model.avgMetrics, tuning_eval)}")

    tvs = TrainValidationSplit(
        estimator=tuning_lr,
        estimatorParamMaps=tuning_grid,
        evaluator=tuning_eval,
        trainRatio=0.75,
        seed=42,
    )
    tvs_model = tvs.fit(tuning_frame)

    print(f"  tvs split metrics           {list(tvs_model.validationMetrics)}")
    print(
        "  tvs best index              "
        f"{_best_index(tvs_model.validationMetrics, tuning_eval)}"
    )

    # ML5c. A search's directory is laid out entirely by the client — metadata, the encoded
    # grid, `estimator/`, `evaluator/`, `bestModel/` — so this is the only thing that says
    # whether Latu's is Spark's own shape. Note what a wrong `parent` uid in the grid would do:
    # the search still loads, and simply has nothing to vary.
    cv_model.write().overwrite().save(f"{INTEROP_ROOT}/pyspark_cv")
    print(f"  saved for Latu              {INTEROP_ROOT}/pyspark_cv")

    cv.write().overwrite().save(f"{INTEROP_ROOT}/pyspark_cv_unfitted")

    failures += _read_back(
        "Latu's search, read here",
        lambda: list(CrossValidatorModel.load(f"{INTEROP_ROOT}/latu_cv").avgMetrics),
        list(cv_model.avgMetrics),
    )

    failures += _read_back(
        "and its grid",
        lambda: _grid_of(CrossValidatorModel.load(f"{INTEROP_ROOT}/latu_cv")),
        _grid_of(cv_model),
    )

    failures += _read_back(
        "Latu's unfitted search",
        lambda: _grid_of(CrossValidator.load(f"{INTEROP_ROOT}/latu_cv_unfitted")),
        _grid_of(cv),
    )

    # The unfitted half. Nothing was fitted, so there is nothing to compare but the shape:
    # a pipeline Latu wrote has to come back here as the two stages it went in as, in order.
    failures += _read_back(
        "Latu's unfitted pipeline",
        lambda: [
            type(stage).__name__
            for stage in Pipeline.load(f"{INTEROP_ROOT}/latu_pipeline_unfitted").getStages()
        ],
        ["VectorAssembler", "LogisticRegression"],
    )

    return failures


def _grid_of(validator):
    """A validator's grid as plain pairs, so two of them can be compared across clients.

    Param objects are not equal across instances — they carry their parent's uid — so the
    comparison is by name and value, which is what the saved format actually preserves.
    """
    return [
        sorted((param.name, value) for param, value in param_map.items())
        for param_map in validator.getEstimatorParamMaps()
    ]


def _best_index(metrics, evaluator):
    """Which param map won — PySpark's own rule, so the Elixir side has something to match.

    `argmax`/`argmin` both return the *first* extreme, which is what makes a tie reproducible.
    """
    metrics = list(metrics)
    best = max(metrics) if evaluator.isLargerBetter() else min(metrics)

    return metrics.index(best)


def _read_back(label, load, expected):
    """Read something the other client wrote and compare it. Returns 0 or 1.

    A refusal is not a failure: it is what running this half before the Elixir half looks
    like, and saying so is more use than an exit code.
    """
    try:
        found = load()
    except Exception as exc:  # noqa: BLE001 - the refusal is the answer
        line = str(exc).splitlines()[0][:110]
        print(f"  {label:<28}{type(exc).__name__}: {line}")
        print("      (run `mix run dev/probe_ml.exs -- --interop` first)")
        return 0

    if found == expected:
        print(f"  {label:<28}identical")
        return 0

    print(f"  {label:<28}DIFFER — Latu's {found}, ours {expected}")
    return 1


def build(spark, name):
    made = fixtures(spark)

    if name not in made:
        sys.exit(f"no fixture named {name}. One of: {', '.join(sorted(made))}")

    return made[name]()


def generate(spark):
    """Write test/wire/<name>.{bin,txtpb} for every fixture. Returns the failures."""
    latu = latu_oracle()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for stale in list(OUT_DIR.glob("*.bin")) + list(OUT_DIR.glob("*.txtpb")):
        stale.unlink()  # renamed fixtures must not linger

    failed = []

    for name, make in sorted(fixtures(spark).items()):
        try:
            message = make()
            latu.strip_origins(message)
            latu.normalize_plan_ids(message)

            header = "# generated by dev/pyspark_oracle.py\n\n"
            # deterministic= sorts map entries: MlParams is a map field, and Python serialises
            # those in an order that varies per process. Without it two --generate runs write
            # the same message as different bytes, which only "regenerate and diff" notices.
            encoded = message.SerializeToString(deterministic=True)

            (OUT_DIR / f"{name}.bin").write_bytes(encoded)
            (OUT_DIR / f"{name}.txtpb").write_text(
                header + text_format.MessageToString(message)
            )
            print(f"  ok   {name}")
        except Exception as exc:  # noqa: BLE001
            failed.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"  FAIL {name}: {type(exc).__name__}: {exc}")

    return failed


def main():
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)  # allow `| head` without a traceback

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--generate", action="store_true", help=f"write fixtures to {OUT_DIR}/"
    )
    parser.add_argument("--show", metavar="NAME", help="print one fixture as text")
    parser.add_argument("--raw-ids", action="store_true", help="keep PySpark's plan_ids as-is")
    parser.add_argument(
        "--interop",
        action="store_true",
        help=f"save a model under {INTEROP_ROOT} for Latu, and read the one Latu saved",
    )
    args = parser.parse_args()

    if not (args.generate or args.show or args.interop):
        parser.print_help()
        return

    spark = get_session()
    silence_deletes()
    pin_uids()

    if args.interop:
        print(f"interop, both directions ({INTEROP_ROOT})")
        sys.exit(interop(spark))
    elif args.generate:
        failed = generate(spark)
        total = len(fixtures(spark))
        print(f"\n{total - len(failed)} of {total} fixtures written to {OUT_DIR}")

        if failed:
            sys.exit(1)
    else:
        latu = latu_oracle()
        message = latu.strip_origins(build(spark, args.show))

        if not args.raw_ids:
            latu.normalize_plan_ids(message)

        print(text_format.MessageToString(message))


if __name__ == "__main__":
    main()
