defmodule Latu.MLTest do
  use ExUnit.Case, async: true

  # The facade, offline. `transform/2` is a lazy builder, so all of it is testable with no
  # session and no server — which is the point of the layering.
  #
  # The doctests are the registry's: every `iex>` in `Latu.ML` and `Latu.ML.Operator` reads it,
  # and none of them touches a session. The generated constructors' docs carry no `iex>` block
  # — a fit needs a server — so their examples are the goldens in `wire_test.exs` instead.
  doctest Latu.ML
  doctest Latu.ML.Operator

  alias Latu.DataFrame
  alias Latu.ML
  alias Latu.ML.Classification
  alias Latu.ML.Estimator
  alias Latu.ML.Evaluation
  alias Latu.ML.Feature
  alias Latu.ML.Internal
  alias Latu.ML.Model
  alias Latu.ML.Layout
  alias Latu.ML.Operator
  alias Latu.ML.Pipeline
  alias Latu.ML.PipelineModel
  alias Latu.ML.Transformer
  alias Latu.Protocol.Spark.Connect, as: Proto

  defp frame, do: %DataFrame{session: nil, plan: Latu.Plan.range(0, 4, 1)}

  # The wire spelling of a param, off the estimator being searched — what `save/3` hands
  # `Layout.search_params/3`.
  defp wire_of(validator) do
    fn _uid, name -> Enum.find(validator.estimator.known, &(&1.name == name)).wire end
  end

  defp project_of(%DataFrame{plan: %Proto.Relation{rel_type: {:filter, filter}}}) do
    case filter.input do
      %Proto.Relation{rel_type: {:project, project}} -> project
      _other -> nil
    end
  end

  defp ml_relation(%DataFrame{plan: %Proto.Relation{rel_type: {:ml_relation, relation}}}) do
    relation
  end

  defp literal(arm), do: %Proto.Expression.Literal{literal_type: arm}

  defp strings(values) do
    literal(
      {:array,
       %Proto.Expression.Literal.Array{
         element_type: %Proto.DataType{
           kind: {:string, %Proto.DataType.String{collation: "UTF8_BINARY"}}
         },
         elements: Enum.map(values, &literal({:string, &1}))
       }}
    )
  end

  defp read(params) do
    result(%Proto.MlCommandResult.MlOperatorInfo{
      type: {:obj_ref, %Proto.ObjectRef{id: "m-8"}},
      params: %Proto.MlParams{params: params}
    })
  end

  defp helper_call(%DataFrame{plan: %Proto.Relation{rel_type: {:ml_relation, relation}}}) do
    {:fetch, fetch} = relation.ml_type

    hd(fetch.methods)
  end

  defp helper_method(frame), do: helper_call(frame).method
  defp helper_args(frame), do: helper_call(frame).args

  defp result(info) do
    %{ml_command_result: %Proto.MlCommandResult{result_type: {:operator_info, info}}}
  end

  describe "constructors" do
    test "translate snake_case names to Spark's own spelling" do
      assembler = Feature.vector_assembler(input_cols: [:x1, :x2], output_col: :features)

      assert %Transformer{} = assembler
      assert assembler.class == "org.apache.spark.ml.feature.VectorAssembler"

      assert assembler.params == [
               {"inputCols", {:list, :string}, [:x1, :x2]},
               {"outputCol", :string, :features}
             ]
    end

    test "send nothing the caller did not set" do
      assert Classification.logistic_regression().params == []
    end

    test "refuse an unknown param by listing the ones there are" do
      assert_raise ArgumentError, ~r/unknown param :max_iters, expected one of/, fn ->
        Classification.logistic_regression(max_iters: 10)
      end
    end

    test "give each operator a uid in PySpark's shape" do
      assert Classification.logistic_regression().uid =~ ~r/^LogisticRegression_[0-9a-f]{12}$/
    end

    test "carry the model class, since a Fit does not answer with one" do
      assert Classification.logistic_regression().model_class ==
               "org.apache.spark.ml.classification.LogisticRegressionModel"
    end

    test "Estimator.new is the escape hatch, and checks nothing" do
      estimator = Estimator.new("com.example.Thing", params: [{"numRound", :int, 100}])

      assert %Estimator{known: []} = estimator
      assert estimator.params == [{"numRound", :int, 100}]
    end

    test "build the struct their kind calls for" do
      assert %Estimator{} = Classification.logistic_regression()
      assert %Transformer{} = Feature.vector_assembler()
      assert %Latu.ML.Evaluator{} = Latu.ML.Evaluation.regression_evaluator()
    end

    # `Code.ensure_loaded!/1` is load-bearing, as it is in `twins_test.exs`:
    # `function_exported?/3` answers false for a module that is merely compiled, and five of
    # the seven generated group modules are never named anywhere else in this file.
    test "exist for every operator the registry advertises one for" do
      missing =
        for %Operator{constructor: {module, name, arity}} <- ML.operators(),
            Code.ensure_loaded!(module),
            not function_exported?(module, name, arity),
            do: "#{inspect(module)}.#{name}/#{arity}"

      assert missing == []
    end

    test "and no model has one, because a model exists only by fitting" do
      assert Enum.all?(ML.operators(kind: :model), &is_nil(&1.constructor))
    end
  end

  describe "the registry" do
    test "answers for every operator PySpark exposes" do
      assert length(ML.operators()) == 111
      assert ML.operators(kind: :estimator) |> length() == 42
      assert ML.operators(kind: :model) |> length() == 43
      assert ML.operators(kind: :transformer) |> length() == 18
      assert ML.operators(kind: :evaluator) |> length() == 6

      # The fifth kind: neither fitted nor applied, and its one method is on the helper object.
      assert ML.operators(kind: :helper) |> Enum.map(& &1.name) ==
               [:power_iteration_clustering, :prefix_span]
    end

    test "filters are and-ed, and an unknown one raises rather than matching nothing" do
      [only] = ML.operators(group: :recommendation, kind: :estimator)
      assert only.name == :als

      assert_raise ArgumentError, fn -> ML.operators(kinds: :estimator) end
    end

    test "carries the Spark version each row was extracted from" do
      assert Enum.all?(ML.operators(), &(&1.spark == "4.2.0"))
    end

    test "every status is one of the three, and only a :built or :missing one keeps a reason" do
      assert Enum.all?(ML.operators(), &(&1.status in [:probed, :built, :missing]))
      assert Enum.all?(ML.operators(status: :probed), &is_nil(&1.reason))
    end

    # The inverse of what ML2 asserted here, and for the reason ML2 named: the probe skipped
    # evaluators while no verb sent an `Evaluate`, and `Latu.ML.evaluate/2` is that verb. An
    # evaluator is now run like anything else, and none of the six may be left behind.
    #
    # Branching on the probe file rather than assuming it: `priv/ml_probe.exs` is gitignored and
    # regenerated on demand, so a fresh checkout has everything `:built` and that is not a
    # failure — it is the claim a package with no probe result is entitled to make.
    test "an evaluator is probed like anything else" do
      case ML.operators(status: :probed) do
        [] ->
          assert ML.operators(status: :built) == ML.operators()

        _probed ->
          assert length(ML.operators(kind: :evaluator, status: :probed)) == 6
      end
    end

    test "operator/1 misses with nil, operator!/1 with the near spellings" do
      assert ML.operator(:no_such_thing) == nil
      assert ML.operator(:standard_scaler).kind == :estimator

      assert_raise ArgumentError, ~r/Did you mean one of .*:logistic_regression/, fn ->
        ML.operator!(:logistic_regresion)
      end
    end

    test "params/1 is the table the constructor's own doc is written from" do
      names = ML.params(:vector_assembler) |> Enum.map(& &1.name)
      assert names == [:handle_invalid, :input_cols, :output_col]

      assert %{wire: "handleInvalid", type: :string, default: "error"} =
               ML.params(:vector_assembler) |> Enum.find(&(&1.name == :handle_invalid))
    end

    test "attributes/1 takes a module, a class or a model, and answers the same either way" do
      by_module = ML.attributes(Latu.ML.Clustering.KMeansModel)

      assert by_module == ML.attributes("org.apache.spark.ml.clustering.KMeansModel")

      assert by_module ==
               ML.attributes(%Model{
                 ref: "m-1",
                 session: nil,
                 class: "org.apache.spark.ml.clustering.KMeansModel"
               })

      assert %{name: :cluster_center_matrix, returns: :value, arity: 0} =
               Enum.find(by_module, &(&1.name == :cluster_center_matrix))
    end

    test "and answers nil for a class it does not know" do
      assert ML.attributes("com.example.Thing") == nil
    end
  end

  defp kmeans do
    %Model{ref: "m-1", session: nil, class: "org.apache.spark.ml.clustering.KMeansModel"}
  end

  describe "generated accessors" do
    alias Latu.ML.Clustering.KMeansModel
    alias Latu.ML.Recommendation.ALSModel

    test "exist for exactly the allowlisted attributes this package can send" do
      assert KMeansModel.__info__(:functions) |> Keyword.keys() |> Enum.sort() ==
               [:class, :cluster_center_matrix, :has_summary, :num_features, :predict, :summary]
    end

    test "and an argument-taker is generated at one plus its arguments" do
      assert KMeansModel.__info__(:functions)[:predict] == 2
      assert KMeansModel.__info__(:functions)[:num_features] == 1
    end

    test "an argument's count comes from the registry, not from the caller" do
      assert_raise ArgumentError, ~r/predict takes 1 arguments — value \(:vector\)/, fn ->
        ML.attribute(kmeans(), "predict", [])
      end
    end

    test "carry the class they belong to" do
      assert KMeansModel.class() == "org.apache.spark.ml.clustering.KMeansModel"
    end

    test "refuse a model of another class, naming the module that would take it" do
      assert_raise ArgumentError, ~r/Try Latu.ML.Clustering.KMeansModel/, fn ->
        Latu.ML.Classification.LogisticRegressionModel.coefficients(kmeans())
      end
    end

    test "a frame-valued attribute is a lazy builder, not an action" do
      frame = ALSModel.item_factors(%Model{kmeans() | class: ALSModel.class()})

      assert %DataFrame{session: nil} = frame
      assert {:fetch, fetch} = ml_relation(frame).ml_type
      assert fetch.obj_ref.id == "m-1"
      assert Enum.map(fetch.methods, & &1.method) == ["itemFactors"]
      assert ml_relation(frame).model_summary_dataset == nil
    end
  end

  describe "transform/2" do
    test "is lazy: a frame in, a frame out, no session touched" do
      assembler = Feature.vector_assembler(input_cols: [:x1], output_col: :features)
      out = ML.transform(assembler, frame())

      assert %DataFrame{session: nil} = out
      assert {:transform, transform} = ml_relation(out).ml_type
      assert {:transformer, operator} = transform.operator
      assert operator.type == :OPERATOR_TYPE_TRANSFORMER
      assert transform.params.params["inputCols"]
    end

    test "a fitted model transforms by reference, resending its params as PySpark does" do
      model = %Model{ref: "m-1", session: nil, params: [{"threshold", :double, 0.7}]}

      assert {:transform, transform} = ml_relation(ML.transform(model, frame())).ml_type
      assert {:obj_ref, ref} = transform.operator
      assert ref.id == "m-1"
      assert transform.params.params["threshold"].literal_type == {:double, 0.7}
    end

    test "composes, because the result is an ordinary frame" do
      assembler = Feature.vector_assembler(input_cols: [:x1], output_col: :features)
      model = %Model{ref: "m-1", session: nil}

      nested = frame() |> then(&ML.transform(assembler, &1)) |> then(&ML.transform(model, &1))

      assert {:transform, outer} = ml_relation(nested).ml_type
      assert {:ml_relation, _inner} = outer.input.rel_type
    end
  end

  describe "error_kind/1 and hint/1" do
    # The whole point of the table: `error_class` carries the dotted name, and Spark's own docs
    # write these bare. A matcher built from the docs compares against a string that never
    # arrives, which is a bug that looks like "the error never happens".
    test "match the dotted class the server actually sends" do
      assert ML.error_kind(%Latu.Error{kind: :rpc, error_class: "CONNECT_ML.CACHE_INVALID"}) ==
               :cache_invalid

      assert ML.error_kind(%Latu.Error{kind: :rpc, error_class: "CACHE_INVALID"}) == nil
    end

    test "cover the five classes 4.2.0 raises" do
      classes = [
        {"CONNECT_ML.CACHE_INVALID", :cache_invalid},
        {"CONNECT_ML.ATTRIBUTE_NOT_ALLOWED", :attribute_not_allowed},
        {"CONNECT_ML.MODEL_SUMMARY_LOST", :summary_lost},
        {"CONNECT_ML.MODEL_SIZE_OVERFLOW_EXCEPTION", :model_too_large},
        {"CONNECT_ML.ML_CACHE_SIZE_OVERFLOW_EXCEPTION", :cache_full}
      ]

      for {class, kind} <- classes do
        error = %Latu.Error{kind: :rpc, error_class: class}

        assert ML.error_kind(error) == kind
        assert is_binary(ML.hint(error))
      end
    end

    test "and say nothing about an error that is not one of them" do
      other = %Latu.Error{kind: :rpc, error_class: "UNRESOLVED_COLUMN"}

      assert ML.error_kind(other) == nil
      assert ML.hint(other) == nil
      assert ML.hint(%Latu.Error{kind: :protocol, message: "nope"}) == nil
    end
  end

  describe "pipelines, offline" do
    # A pipeline has no server-side anything, so the fold, the composition and the format are
    # all testable with no session at all — a pipeline of transformers reaches no server, since
    # a fit only acts where there is an estimator.
    defp assembler do
      Feature.vector_assembler(input_cols: [:x1, :x2], output_col: :features)
    end

    defp binarizer do
      Feature.binarizer(input_col: :x1, output_col: :above, threshold: 0.5)
    end

    test "a pipeline is a builder: no session, no server, no cache entry" do
      pipeline = ML.pipeline([assembler(), Classification.logistic_regression()])

      assert %Pipeline{} = pipeline
      assert pipeline.uid =~ ~r/^Pipeline_[0-9a-f]{12}$/
      assert length(pipeline.stages) == 2
    end

    test "and refuses anything that is not a stage, before a fit has cached anything" do
      assert_raise ArgumentError, ~r/is not a pipeline stage/, fn ->
        ML.pipeline([assembler(), Evaluation.regression_evaluator()])
      end
    end

    test "fitting one with no estimator in it sends nothing at all" do
      stages = [assembler(), binarizer()]

      assert {:ok, %PipelineModel{} = model} = ML.fit(ML.pipeline(stages), frame())
      assert model.stages == stages
      assert model.uid =~ ~r/^PipelineModel_[0-9a-f]{12}$/
    end

    test "and transforming composes every stage, lazily" do
      {:ok, model} = ML.fit(ML.pipeline([assembler(), binarizer()]), frame())
      composed = ML.transform(model, frame())

      # Two ml_relations, one inside the other, and the innermost input is the frame we passed.
      assert %{rel_type: {:ml_relation, outer}} = composed.plan
      assert {:transform, %{input: %{rel_type: {:ml_relation, _inner}}}} = outer.ml_type
    end

    test "a pipeline model with nothing cached in it has nothing to delete" do
      {:ok, model} = ML.fit(ML.pipeline([assembler()]), frame())

      assert ML.delete(model) == :ok
    end

    # `delete([fitted, best])` is what a script that fitted a pipeline and a search writes, and
    # the first version of `delete/1` had a clause for `[%Model{}| _]` and none for a list of
    # composites — so it answered with a `FunctionClauseError` and a screenful of inspected
    # struct, where `CLAUDE.md` promises a refusal that names the fix. `dev/example.exs` found
    # it; these two are what keep it found.
    test "and a mixed list of composites is one call, not a clause error" do
      {:ok, one} = ML.fit(ML.pipeline([assembler()]), frame())
      {:ok, two} = ML.fit(ML.pipeline([binarizer()]), frame())

      assert ML.delete([one, two]) == :ok
    end

    test "while something the cache never holds is refused by name, at the top level" do
      refusal = ~r/is not something the ML cache holds/

      assert_raise ArgumentError, refusal, fn -> ML.delete(:nope) end
      assert_raise ArgumentError, refusal, fn -> ML.delete([assembler()]) end
      assert_raise ArgumentError, refusal, fn -> ML.delete(%{ref: "not-a-model"}) end
    end

    test "an unfitted pipeline has no session, so save/3 asks for one" do
      unfitted = ~r/a pipeline has no session of its own until it is fitted/

      assert_raise ArgumentError, unfitted, fn ->
        ML.save(ML.pipeline([assembler()]), "/tmp/p")
      end
    end

    test "and load/3 names the two spellings that name no operator" do
      assert_raise ArgumentError, ~r/:pipeline or :pipeline_model/, fn ->
        ML.load(nil, "pyspark.ml.pipeline.Pipeline", "/tmp/p")
      end
    end
  end

  describe "the metadata a pipeline is saved with" do
    # This file is the whole compatibility claim, and it comes down to `class`. Read against
    # Scala's writer the two formats differ by that string alone — and the readers are not
    # symmetric: PySpark 4.2's pipeline reader never looks at it, Scala's `loadMetadata` does
    # `require(className == expectedClassName)`. So the PySpark spelling here is what makes a
    # saved pipeline readable by PySpark and refused by Scala. Deliberate, documented in
    # `docs/decisions.md`, and the one key that says who really wrote it keeps it honest.
    setup do
      json =
        Layout.metadata(
          :pipeline_model,
          "PipelineModel_0123456789ab",
          %{"stageUids" => ["a", "b"], "language" => "Python"},
          "4.2.0"
        )

      %{metadata: JSON.decode!(json), json: json}
    end

    test "carries every field PySpark's writer does", %{metadata: metadata} do
      assert metadata["class"] == "pyspark.ml.pipeline.PipelineModel"
      assert metadata["uid"] == "PipelineModel_0123456789ab"
      assert metadata["sparkVersion"] == "4.2.0"
      assert metadata["paramMap"] == %{"stageUids" => ["a", "b"], "language" => "Python"}
      assert metadata["defaultParamMap"] == %{}
      assert is_integer(metadata["timestamp"])
    end

    test "and one more, saying who actually wrote it", %{metadata: metadata} do
      assert %{"version" => version, "note" => note} = metadata["latuMl"]
      assert is_binary(version)
      assert note =~ "latu_ml"
      assert note =~ "written by PySpark"
      assert note =~ "not by Scala Spark"
    end

    # The reason the extra key is safe: PySpark's `DefaultParamsReader` hands back a plain dict
    # and pulls named keys, and Scala's `parseMetadata` extracts six fields with json4s paths
    # into a `Metadata` case class. Neither enumerates the key set.
    test "which is a key both readers carry and neither reads", %{json: json} do
      assert json =~ ~s("class":"pyspark.ml.pipeline.PipelineModel")

      assert JSON.decode!(json) |> Map.keys() |> Enum.sort() == [
               "class",
               "defaultParamMap",
               "latuMl",
               "paramMap",
               "sparkVersion",
               "timestamp",
               "uid"
             ]
    end
  end

  describe "larger_better?/1" do
    # The registry's one value that no probe can check: `isLargerBetter` is not on the server's
    # allowlist, so it exists only as a client-side override in PySpark's six evaluators, and
    # the extractor reads those. A wrong answer here is silent — a grid search that reports the
    # worst model as the best one — so the six are pinned by hand against PySpark's source.
    test "the six overrides, as the extractor read them" do
      values =
        for operator <- ML.operators(kind: :evaluator),
            into: %{},
            do: {operator.name, operator.larger_better}

      assert values == %{
               binary_classification_evaluator: true,
               clustering_evaluator: true,
               ranking_evaluator: true,
               regression_evaluator: {:only, ["r2", "var"]},
               multilabel_classification_evaluator: {:except, ["hammingLoss"]},
               multiclass_classification_evaluator:
                 {:except,
                  [
                    "weightedFalsePositiveRate",
                    "falsePositiveRateByLabel",
                    "logLoss",
                    "hammingLoss"
                  ]}
             }
    end

    test "and nothing that is not an evaluator carries one" do
      assert Enum.filter(ML.operators(), &(&1.kind != :evaluator and &1.larger_better != nil)) ==
               []
    end

    test "a metric that is set decides, and an unset one falls back to the default" do
      assert ML.larger_better?(Evaluation.regression_evaluator(metric_name: "r2"))
      refute ML.larger_better?(Evaluation.regression_evaluator(metric_name: "rmse"))

      # `rmse` is the default, so these two have to agree.
      refute ML.larger_better?(Evaluation.regression_evaluator())
    end

    test "an evaluator whose answer never depends on the metric says so for any of them" do
      for metric <- ["areaUnderROC", "areaUnderPR"] do
        assert ML.larger_better?(Evaluation.binary_classification_evaluator(metric_name: metric))
      end
    end

    test "an evaluator this registry never heard of raises, naming what it was" do
      stranger = %Latu.ML.Evaluator{
        class: "com.example.MyEvaluator",
        uid: "MyEvaluator_0123456789ab",
        params: [],
        known: []
      }

      assert_raise ArgumentError, ~r/com\.example\.MyEvaluator/, fn ->
        ML.larger_better?(stranger)
      end
    end
  end

  describe "the fold cut, offline" do
    # Every pair is a plan, so the cut can be read without a server. It is worth reading,
    # because it is the one part of a search that fails silently: a cut whose train and
    # validation overlap still fits, still scores, and still names a best model.
    setup do
      lr = Classification.logistic_regression()

      validator =
        ML.cross_validator(
          estimator: lr,
          param_maps: ML.param_grid(lr, reg_param: [0.1, 0.01]),
          evaluator: Evaluation.binary_classification_evaluator(),
          num_folds: 3,
          seed: 42
        )

      %{validator: validator, lr: lr}
    end

    test "one pair per fold", %{validator: validator} do
      assert length(Internal.folds(frame(), validator)) == 3
    end

    test "the draw is added once, above both filters", %{validator: validator} do
      [{train, validation} | _rest] = Internal.folds(frame(), validator)

      # Both sides filter the *same* projected relation. If the draw were inside the
      # predicates instead, Spark would redraw it per evaluation and the two would overlap.
      assert project_of(train) == project_of(validation)
      assert project_of(train) != nil
    end

    test "and it is named for the validator, as PySpark names it", %{validator: validator} do
      [{train, _validation} | _rest] = Internal.folds(frame(), validator)

      assert inspect(project_of(train)) =~ validator.uid <> "_rand"
    end

    test "a fold_col cut adds no draw at all", %{lr: lr} do
      validator =
        ML.cross_validator(
          estimator: lr,
          param_maps: ML.param_grid(lr, reg_param: [0.1]),
          evaluator: Evaluation.binary_classification_evaluator(),
          num_folds: 2,
          fold_col: "fold"
        )

      [{train, validation} | _rest] = Internal.folds(frame(), validator)

      assert project_of(train) == nil
      assert project_of(validation) == nil
    end

    test "a train/validation split is one pair, not num_folds of them", %{lr: lr} do
      validator =
        ML.train_validation_split(
          estimator: lr,
          param_maps: ML.param_grid(lr, reg_param: [0.1]),
          evaluator: Evaluation.binary_classification_evaluator(),
          train_ratio: 0.75,
          seed: 42
        )

      assert [{train, validation}] = Internal.folds(frame(), validator)
      assert project_of(train) == project_of(validation)
    end

    test "a seed makes the same cut twice, and no seed does not", %{validator: validator} do
      # Not the relations themselves: `plan_id` counts up per relation built, so two identical
      # plans are never `==`. What reproducibility means here is the draw, so that is what is
      # compared — and `inspect` renders it without the id.
      draw = fn v ->
        [{train, _validation} | _rest] = Internal.folds(frame(), v)
        inspect(project_of(train).expressions)
      end

      assert draw.(validator) =~ "rand(42)"
      assert draw.(validator) == draw.(validator)

      # `rand()` with no seed draws one per build, so the plan differs between them. Latu says
      # so for every seeded function, and a search is where it matters most.
      unseeded = %{validator | seed: nil}
      refute draw.(unseeded) == draw.(unseeded)
    end
  end

  describe "applying a param map" do
    # The two ways a search fails without failing: a grid whose params reach nothing, and a
    # param that lands on the wrong stage. Both produce a completed search and a best model.
    setup do
      lr = Classification.logistic_regression(max_iter: 1)
      assembler = Feature.vector_assembler(input_cols: [:x1, :x2], output_col: :features)

      %{lr: lr, assembler: assembler, pipeline: ML.pipeline([assembler, lr])}
    end

    test "reaches the stage that owns the param, and leaves the rest alone", context do
      %{lr: lr, assembler: assembler, pipeline: pipeline} = context

      [param_map] = ML.param_grid(lr, reg_param: [0.25])
      [applied_assembler, applied_lr] = Internal.apply_params(pipeline, param_map).stages

      assert applied_assembler == assembler
      assert {"regParam", :double, 0.25} in applied_lr.params
    end

    test "replaces a param the constructor already set, rather than sending it twice", context do
      %{lr: lr} = context

      [param_map] = ML.param_grid(lr, max_iter: [50])
      applied = Internal.apply_params(lr, param_map)

      assert Enum.count(applied.params, fn {wire, _type, _value} -> wire == "maxIter" end) == 1
      assert {"maxIter", :int, 50} in applied.params
    end

    test "a grid built from something the search never touches is refused", context do
      %{lr: lr, assembler: assembler} = context

      # A grid for `lr`, but the search is over the assembler alone. Every param map would
      # score identically and the first would 'win'.
      validator =
        ML.cross_validator(
          estimator: assembler,
          param_maps: ML.param_grid(lr, reg_param: [0.1, 0.01]),
          evaluator: Evaluation.binary_classification_evaluator()
        )

      assert_raise ArgumentError, ~r/not the estimator being searched/, fn ->
        Internal.covered!(validator)
      end
    end

    test "and a grid for a stage inside the pipeline is not", context do
      %{lr: lr, pipeline: pipeline} = context

      validator =
        ML.cross_validator(
          estimator: pipeline,
          param_maps: ML.param_grid(lr, reg_param: [0.1, 0.01]),
          evaluator: Evaluation.binary_classification_evaluator()
        )

      assert Internal.covered!(validator) == :ok
    end
  end

  describe "best_index/1" do
    setup do
      lr = Classification.logistic_regression()
      %{grid: ML.param_grid(lr, reg_param: [0.1, 0.01, 0.001]), lr: lr}
    end

    defp searched(grid, lr, evaluator, metrics) do
      %Latu.ML.CrossValidatorModel{
        uid: "CrossValidatorModel_0123456789ab",
        session: nil,
        best_model: nil,
        avg_metrics: metrics,
        validator: ML.cross_validator(estimator: lr, param_maps: grid, evaluator: evaluator)
      }
    end

    test "maximises where a bigger metric is better", %{grid: grid, lr: lr} do
      auc = Evaluation.binary_classification_evaluator()

      assert ML.best_index(searched(grid, lr, auc, [0.7, 0.9, 0.8])) == 1
    end

    test "and minimises where it is not", %{grid: grid, lr: lr} do
      rmse = Evaluation.regression_evaluator()

      assert ML.best_index(searched(grid, lr, rmse, [0.7, 0.9, 0.5])) == 2
    end

    test "a tie goes to the first, as np.argmax does it", %{grid: grid, lr: lr} do
      auc = Evaluation.binary_classification_evaluator()

      assert ML.best_index(searched(grid, lr, auc, [0.9, 0.9, 0.1])) == 0
    end
  end

  describe "the grid a saved search carries" do
    # Spark stores a grid entry as `parent`, `name`, `value`, `isJson` — and the value is JSON
    # *inside* a JSON document, so 0.5 is the string "0.5" and not the number. That is the
    # writer (`json.dumps(v)`) and the reader (`json.loads(...)`) both, and a file with the
    # number in it does not load in PySpark. Pinned because nothing about it is guessable.
    setup do
      lr = Classification.logistic_regression()

      %{
        lr: lr,
        wire_of: fn
          _uid, :reg_param -> "regParam"
          _uid, other -> to_string(other)
        end,
        name_of: fn _uid, "regParam" -> :reg_param end
      }
    end

    test "an entry names its parent and the wire spelling", context do
      %{lr: lr, wire_of: wire_of} = context

      [[entry]] = Layout.encode_grid(ML.param_grid(lr, reg_param: [0.5]), wire_of)

      assert entry["parent"] == lr.uid
      assert entry["name"] == "regParam"
      assert entry["isJson"] == true
    end

    test "and the value is encoded twice, so it is a string", context do
      %{lr: lr, wire_of: wire_of} = context

      [[entry]] = Layout.encode_grid(ML.param_grid(lr, reg_param: [0.5]), wire_of)

      assert entry["value"] == "0.5"
      assert is_binary(entry["value"])

      # The whole document, as it lands on disk: a JSON string inside JSON.
      assert JSON.encode!(entry) =~ ~s("value":"0.5")
    end

    test "a grid round-trips through both", context do
      %{lr: lr, wire_of: wire_of, name_of: name_of} = context

      grid = ML.param_grid(lr, reg_param: [0.5, 0.05])

      assert Layout.decode_grid(Layout.encode_grid(grid, wire_of), name_of) == grid
    end

    test "an operator-valued param is refused rather than half-written", context do
      %{wire_of: wire_of} = context

      operator = %Latu.ML.Helper{class: "x", params: [], known: []}

      assert_raise ArgumentError, ~r/does not write that shape/, fn ->
        Layout.encode_grid([[{"LR_1", :classifier, operator}]], wire_of)
      end
    end

    test "and so is one read back from a directory", context do
      %{name_of: name_of} = context

      entry = %{"parent" => "LR_1", "name" => "regParam", "value" => "epm_x0", "isJson" => false}

      assert_raise ArgumentError, ~r/does not read that shape/, fn ->
        Layout.decode_grid([[entry]], name_of)
      end
    end
  end

  describe "the params a saved search carries" do
    setup do
      lr = Classification.logistic_regression()

      %{
        validator:
          ML.cross_validator(
            estimator: lr,
            param_maps: ML.param_grid(lr, reg_param: [0.1]),
            evaluator: Evaluation.binary_classification_evaluator(),
            num_folds: 4,
            seed: 7
          )
      }
    end

    # `getAndSetParams` calls `getParam(name)` for every key in `paramMap` and raises on one
    # the instance does not have. `CrossValidatorModel` has no `collectSubModels` where
    # `CrossValidator` does — so a fitted search has to carry **fewer** keys than an unfitted
    # one, or PySpark refuses to load it. PySpark never trips on this because it extracts from
    # the instance in front of it; a client writing the map itself has to know.
    test "an unfitted one says whether it collected", %{validator: validator} do
      params = Layout.search_params(validator, :cross_validator, wire_of(validator))

      assert params["collectSubModels"] == false
      assert params["numFolds"] == 4
      assert params["seed"] == 7
      assert params["foldCol"] == ""
    end

    test "and a fitted one must not, because the model has no such param", %{validator: v} do
      params = Layout.search_params(v, :cross_validator_model, wire_of(v))

      refute Map.has_key?(params, "collectSubModels")
      assert params["numFolds"] == 4
    end

    test "a seed that was never set is left out rather than written as null", %{validator: v} do
      validator = %{v | seed: nil}
      params = Layout.search_params(validator, :cross_validator, wire_of(validator))

      refute Map.has_key?(params, "seed")
    end
  end

  describe "the helper object, offline" do
    # `ConnectHelper` is a singleton the server resolves by name, so everything here is about
    # the shape of the call rather than about a model. What a live server does with them is
    # `test/integration/helper_test.exs`; what PySpark builds is `wire_test.exs`.
    test "the eleven methods, by the shape they answer in" do
      assert length(ML.helpers()) == 11
      assert ML.helpers() |> Enum.frequencies_by(& &1.returns) == %{frame: 5, model: 3, value: 3}
    end

    # Ten come from PySpark's call sites. `handleOverwrite` has none — PySpark reaches it
    # through `_call_java` — so its arguments are pinned in the extractor, and the row is
    # marked rather than passing for something derived.
    test "and one is marked as read by hand" do
      assert [pinned] = Enum.filter(ML.helpers(), & &1.read_by_hand)
      assert pinned.name == :handle_overwrite
      assert Enum.map(pinned.args, & &1.name) == [:path, :should_overwrite]
    end

    test "an unknown one names how many there are" do
      assert_raise ArgumentError, ~r/no helper method :nope/, fn -> ML.helper(nil, :nope) end
      assert_raise ArgumentError, ~r/lists the 11 there are/, fn -> ML.helper(nil, :nope) end
    end

    test "each verb refuses the other's shape, rather than sending it" do
      frame_valued = ~r/chiSquareTest answers with a DataFrame; use helper_frame/

      assert_raise ArgumentError, frame_valued, fn ->
        ML.helper(nil, :chi_square_test, [frame(), :features, :label, false])
      end

      assert_raise ArgumentError, ~r/answers with a value; use helper\/3/, fn ->
        ML.helper_frame(:stop_words_remover_load_default_stop_words, ["german"])
      end
    end

    # The frame is an argument rather than a receiver, so the session comes from it — the same
    # rule `fit/2` follows, one layer along.
    test "a frame-valued helper takes its session from the frame it was given" do
      relation = Latu.ML.Stat.correlation(frame(), :features)

      assert relation.session == frame().session
      assert {:ml_relation, _} = relation.plan.rel_type
    end

    test "the three statistical tests each name their own method" do
      methods =
        [
          Latu.ML.Stat.chi_square_test(frame(), :features, :label),
          Latu.ML.Stat.correlation(frame(), :features),
          Latu.ML.Stat.kolmogorov_smirnov_test(frame(), :value, "norm", [0.0, 1.0])
        ]
        |> Enum.map(&helper_method/1)

      assert methods == ["chiSquareTest", "correlation", "kolmogorovSmirnovTest"]
    end

    # An operator that is neither fitted nor applied. Its verb refuses the other one's
    # operator by name, as a generated accessor refuses a model of the wrong class.
    test "a helper operator carries no uid, because nothing sends one" do
      pic = Latu.ML.Clustering.power_iteration_clustering(k: 2)

      assert %Latu.ML.Helper{} = pic
      refute Map.has_key?(pic, :uid)
      assert pic.params == [{"k", :int, 2}]
    end

    test "and its verb refuses the other operator" do
      pic = Latu.ML.Clustering.power_iteration_clustering(k: 2)

      assert_raise ArgumentError, ~r/that verb is org.apache.spark.ml.fpm.PrefixSpan's/, fn ->
        ML.find_frequent_sequential_patterns(pic, frame())
      end
    end

    # Positional and complete: what the caller set, then Spark's default, then the empty string
    # PySpark's own call site sends for the one param Spark gives no default at all.
    test "every argument is sent, set or not" do
      pic = Latu.ML.Clustering.power_iteration_clustering(k: 3)
      [_frame | rest] = helper_args(ML.assign_clusters(pic, frame()))

      assert Enum.map(rest, & &1.args_type) == [
               param: literal({:integer, 3}),
               param: literal({:integer, 20}),
               param: literal({:string, "random"}),
               param: literal({:string, "src"}),
               param: literal({:string, "dst"}),
               param: literal({:string, ""})
             ]
    end
  end

  describe "save and load, offline" do
    # Everything that can be refused before a session is involved. What a live server does with
    # the commands is `test/integration/persistence_test.exs`, and what PySpark builds for the
    # same call is `wire_test.exs`.
    test "a model carries its own session, so a second one is refused" do
      assert_raise ArgumentError, ~r/unknown keys \[:session\]/, fn ->
        ML.save(kmeans(), "/tmp/x", session: :other)
      end
    end

    test "and an unfitted operator has none, so it is asked for" do
      assert_raise ArgumentError, ~r/has not been fitted, so it has no session of its own/, fn ->
        ML.save(Classification.logistic_regression(), "/tmp/x")
      end
    end

    test "a frame is not something to save" do
      assert_raise ArgumentError, ~r/is not something this package can save/, fn ->
        ML.save(frame(), "/tmp/x", session: :stand_in)
      end
    end

    test "loading needs a name it can resolve to a class and a kind" do
      assert_raise ArgumentError, ~r/names nothing to load/, fn ->
        ML.load(nil, "org.apache.spark.ml.classification.LogisticRegression", "/tmp/x")
      end

      assert_raise ArgumentError, ~r/is a training summary, and a summary is not saved/, fn ->
        ML.load(nil, Latu.ML.Classification.BinaryLogisticRegressionSummary, "/tmp/x")
      end
    end
  end

  describe "the operator info a Read answers with" do
    # The riskiest new decode: a param comes back as a bare literal with no declared type
    # beside it, so the arm the server chose is the only thing that says what it is. Every one
    # below has to rebuild as the same literal, because the very next transform resends it.
    test "types each param from its own arm, and rebuilds it unchanged" do
      literals = %{
        "maxIter" => literal({:integer, 7}),
        "seed" => literal({:long, 4_000_000_000}),
        "regParam" => literal({:double, 0.25}),
        "fitIntercept" => literal({:boolean, false}),
        "featuresCol" => literal({:string, "features"}),
        "inputCols" => strings(["x1", "x2"])
      }

      assert {:ok, info} = ML.Result.operator_info(read(literals))

      assert info.params == [
               {"featuresCol", :string, "features"},
               {"fitIntercept", :boolean, false},
               {"inputCols", {:list, :string}, ["x1", "x2"]},
               {"maxIter", :int, 7},
               {"regParam", :double, 0.25},
               {"seed", :long, 4_000_000_000}
             ]

      assert Latu.ML.Plan.params(info.params).params == literals
    end

    test "sorted by name, because a map on the wire has no order" do
      assert {:ok, info} =
               ML.Result.operator_info(
                 read(%{"z" => literal({:integer, 1}), "a" => literal({:integer, 2})})
               )

      assert Enum.map(info.params, &elem(&1, 0)) == ["a", "z"]
    end

    # A Vector-valued param — `Word2Vec` has none, but a third-party operator may, and the
    # arm is the one `Latu.ML.Linalg` owns on both sides.
    test "reads a Vector param back as a tensor" do
      vector = Latu.ML.Linalg.to_literal(Nx.tensor([1.5, -2.5], type: :f64))

      assert {:ok, info} = ML.Result.operator_info(read(%{"centre" => vector}))
      assert [{"centre", :vector, tensor}] = info.params
      assert Nx.to_flat_list(tensor) == [1.5, -2.5]
      assert Latu.ML.Plan.params(info.params).params == %{"centre" => vector}
    end

    test "carries the reference, the uid and the params a model was saved with" do
      info = %Proto.MlCommandResult.MlOperatorInfo{
        type: {:obj_ref, %Proto.ObjectRef{id: "m-9"}},
        uid: "LogisticRegressionModel_0123456789ab"
      }

      assert {:ok, decoded} = ML.Result.operator_info(result(info))
      assert decoded.ref == "m-9"
      assert decoded.name == nil
      assert decoded.uid == "LogisticRegressionModel_0123456789ab"
      assert decoded.params == []
    end

    test "and a class name where nothing was cached" do
      info = %Proto.MlCommandResult.MlOperatorInfo{
        type: {:name, "org.apache.spark.ml.classification.LogisticRegression"},
        uid: "LogisticRegression_0123456789ab"
      }

      assert {:ok, decoded} = ML.Result.operator_info(result(info))
      assert decoded.ref == nil
      assert decoded.name == "org.apache.spark.ml.classification.LogisticRegression"
    end

    # A refusal names the param and what came back, because the alternative is a model that
    # loads and then sends something else on its next transform.
    test "refuses a param it cannot rebuild, naming it" do
      timestamp = literal({:timestamp, 0})

      assert {:error, error} = ML.Result.operator_info(read(%{"when" => timestamp}))
      assert error.kind == :protocol
      assert error.message =~ "the param when came back as the literal arm :timestamp"
    end

    test "and says so too when the answer is not an operator at all" do
      other = %Proto.MlCommandResult{result_type: {:summary, "m-1.summary"}}

      assert {:error, error} = ML.Result.operator_info(%{ml_command_result: other})
      assert error.message =~ "the server answered with summary where an operator belongs"
    end
  end

  describe "inspect" do
    test "a model leads with its class and a short ref" do
      model = %Model{
        ref: "4b8aa52d-0d65-48e0-864a-3c1d76a5df4a",
        session: nil,
        class: kmeans().class
      }

      assert inspect(model) == ~s{#Latu.ML.Model<KMeansModel "4b8aa52d…">}
    end

    test "a model whose class the fit decides shows every class it could be" do
      lda = Latu.ML.Clustering.lda(k: 2)
      model = %Model{ref: "m-1", session: nil, candidates: lda.model_candidates}

      assert inspect(model) ==
               ~s{#Latu.ML.Model<DistributedLDAModel | LocalLDAModel "m-1">}
    end

    test "an estimator shows the params that were set, and only those" do
      assert inspect(Classification.logistic_regression(max_iter: 10)) ==
               "#Latu.ML.Estimator<LogisticRegression [maxIter: 10]>"

      assert inspect(Classification.logistic_regression()) ==
               "#Latu.ML.Estimator<LogisticRegression>"
    end
  end
end
