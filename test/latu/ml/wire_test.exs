defmodule Latu.ML.WireTest do
  use ExUnit.Case, async: true

  # The goldens: what this package builds, against what PySpark builds for the same thing.
  #
  # Nothing here re-implements the facade. `Latu.ML.Internal` holds the struct-to-proto mapping
  # that `Latu.ML.fit/2` and friends use, so a change to how a `Fit` is assembled shows up here
  # rather than passing quietly. Fixtures come from `dev/pyspark_oracle.py --generate`, which
  # captures the command PySpark was about to send and never sends it.

  alias Latu.ML
  alias Latu.ML.Classification
  alias Latu.ML.Clustering
  alias Latu.ML.Evaluation
  alias Latu.ML.Feature
  alias Latu.ML.FPM
  alias Latu.ML.Internal
  alias Latu.ML.Linalg
  alias Latu.ML.Model
  alias Latu.ML.Recommendation
  alias Latu.ML.Registry
  alias Latu.ML.Regression
  alias Latu.ML.SparseVector
  alias Latu.ML.Wire

  # Both sides generate a random uid per operator, so the fixtures pin one.
  defp assembler do
    [input_cols: [:id], output_col: :features]
    |> Feature.vector_assembler()
    |> Map.put(:uid, Wire.assembler_uid())
  end

  defp estimator(opts \\ []) do
    opts
    |> Classification.logistic_regression()
    |> Map.put(:uid, Wire.estimator_uid())
  end

  defp model(params \\ []) do
    %Model{ref: Wire.model_ref(), session: nil, uid: Wire.estimator_uid(), params: params}
  end

  # `LinearSVCModel` for the summary fixtures, because its summary class is one class rather
  # than two the fitted model picks between — the oracle must not ask a server which.
  defp summary do
    ML.summary(%Model{
      ref: Wire.model_ref(),
      session: nil,
      uid: Wire.estimator_uid(),
      class: "org.apache.spark.ml.classification.LinearSVCModel",
      dataset: Wire.base().plan
    })
  end

  defp features, do: ML.transform(assembler(), Wire.base())

  # A generated constructor, with the fixture's uid. Every operator's class name and param
  # spellings come from `priv/ml_operators.exs`, so one golden per family is what says the
  # extractor read them right — the vertical slice only ever proved `LogisticRegression`.
  defp pinned(operator, uid) do
    Map.put(operator, :uid, uid)
  end

  defp fit_golden(operator, name) do
    operator
    |> pinned(Wire.estimator_uid())
    |> Internal.fit_plan(Wire.base().plan)
    |> Wire.assert_wire_command(name)
  end

  describe "relations" do
    test "a transform by an unfitted transformer" do
      Wire.assert_wire(features(), "ml_transform_assembler")
    end

    test "a transform by a fitted model" do
      Wire.assert_wire(ML.transform(model(), features()), "ml_transform_model")
    end

    # The one relation that carries `model_summary_dataset`: the server can rebuild a dropped
    # summary from it while it plans, which is why this arm needs no retry and the command arm
    # does.
    test "a summary attribute that answers with a frame carries the frame that rebuilds it" do
      summary()
      |> ML.attribute_frame(:predictions)
      |> Wire.assert_wire("ml_summary_predictions")
    end

    # A DataFrame argument rides `Fetch.Method.Args`'s `input` arm as a relation, where every
    # other type rides `param`. This is the only fixture with both arms in one argument list.
    test "a frame attribute whose arguments are a DataFrame and an int" do
      model = %Model{model() | class: "org.apache.spark.ml.recommendation.ALSModel"}

      model
      |> ML.attribute_frame(:recommendForUserSubset, [Wire.base(), 5])
      |> Wire.assert_wire("ml_relation_recommend_for_user_subset")
    end

    # A second transformer, for a param that is neither a string nor a list of them.
    test "a transform carrying a double param" do
      [threshold: 0.5, input_col: :id, output_col: :above]
      |> Feature.binarizer()
      |> pinned(Wire.assembler_uid())
      |> ML.transform(Wire.base())
      |> Wire.assert_wire("ml_transform_binarizer")
    end
  end

  describe "commands" do
    # One fixture over every literal arm the registry can produce: an int, two doubles, a
    # boolean and a string. This is the assertion `Latu.ML.Plan.literal/2` most needs, because
    # the coercion rules were read off PySpark's source rather than measured.
    test "a fit carrying every kind of param" do
      lr =
        estimator(
          max_iter: 7,
          reg_param: 0.25,
          threshold: 0.75,
          fit_intercept: false,
          features_col: :features
        )

      lr
      |> Internal.fit_plan(features().plan)
      |> Wire.assert_wire_command("ml_fit_logistic_regression")
    end

    # Defaults are documented, never sent. If this ever fails by gaining params, the registry
    # has started serialising things the caller did not set.
    test "a fit with nothing set sends an empty params map" do
      estimator()
      |> Internal.fit_plan(features().plan)
      |> Wire.assert_wire_command("ml_fit_defaults")
    end

    # One estimator per family. The frame is the bare base rather than an assembled one: what
    # is under test is the operator and its params, and a `Fit` carries the dataset as given.
    test "a fit for a regressor" do
      fit_golden(
        Regression.linear_regression(max_iter: 5, elastic_net_param: 0.5),
        "ml_fit_linear_regression"
      )
    end

    test "a fit for a clusterer" do
      fit_golden(Clustering.k_means(k: 3, distance_measure: "cosine"), "ml_fit_kmeans")
    end

    test "a fit for a feature estimator" do
      fit_golden(Feature.standard_scaler(with_mean: true), "ml_fit_standard_scaler")
    end

    test "a fit for a frequent-pattern miner" do
      fit_golden(FPM.fp_growth(min_support: 0.4, items_col: :items), "ml_fit_fp_growth")
    end

    test "a fit for a recommender" do
      fit_golden(Recommendation.als(rank: 5, implicit_prefs: true), "ml_fit_als")
    end

    # `Latu.ML.evaluate/2` sends this, and an evaluator has no lazy form, so the builder is
    # what a golden can hold. It is also what says the six generated evaluator constructors
    # carry the right class and param spellings.
    test "an evaluate" do
      Evaluation.regression_evaluator(metric_name: "mae")
      |> pinned(Wire.evaluator_uid())
      |> Internal.evaluate_plan(Wire.base().plan)
      |> Wire.assert_wire_command("ml_evaluate_regression")
    end

    # Persistence, one test per arm. A model is written by reference because the server holds
    # it; an unfitted operator by class and uid because the server holds nothing and has to
    # build the instance from the message before it can write it.
    test "a model is written by reference" do
      [{"maxIter", :int, 7}]
      |> model()
      |> Internal.save_plan(Wire.save_path(), false, %{})
      |> Wire.assert_wire_command("ml_write_model")
    end

    test "an unfitted estimator is written by class, with overwrite and a writer option" do
      estimator(max_iter: 7)
      |> Internal.save_plan(Wire.save_path(), true, %{"compression" => "gzip"})
      |> Wire.assert_wire_command("ml_write_estimator")
    end

    # A `Read` names a class and sends no uid. The kind is what decides whether the answer is
    # cached, so both are pinned.
    test "a read names a model class" do
      "org.apache.spark.ml.classification.LogisticRegressionModel"
      |> Internal.load_plan(:model, Wire.save_path())
      |> Wire.assert_wire_command("ml_read_model")
    end

    test "and an estimator's" do
      "org.apache.spark.ml.classification.LogisticRegression"
      |> Internal.load_plan(:estimator, Wire.save_path())
      |> Wire.assert_wire_command("ml_read_estimator")
    end

    test "a fetch for a Vector-valued attribute" do
      model()
      |> Internal.fetch_plan(:coefficients)
      |> Wire.assert_wire_command("ml_fetch_coefficients")
    end

    test "a fetch for a scalar attribute" do
      model()
      |> Internal.fetch_plan(:intercept)
      |> Wire.assert_wire_command("ml_fetch_intercept")
    end

    # A summary is reached through the model that owns it, so the `Fetch` names the model and
    # `summary` leads the chain. PySpark spells that as a dotted string and splits it apart
    # again; this package keeps the parts apart, and the bytes are the same either way.
    test "a fetch through a model's training summary" do
      summary()
      |> Internal.fetch_plan(:totalIterations)
      |> Wire.assert_wire_command("ml_fetch_summary")
    end

    # `Latu.ML.attributes/1` advertises `:total_iterations`; the server's allowlist says
    # `totalIterations`. `wire_name/2` is what stops those being two surfaces, and reusing the
    # golden above is what says the snake spelling reaches **PySpark's bytes** rather than some
    # other plan that also happens to work.
    #
    # The wire-name line would pass on the old `to_string/1`; the snake line is the one that
    # would not, and the pass-through line is what keeps an unknown class reachable.
    test "and a snake_case name reaches the same bytes as the wire name" do
      summary = summary()

      assert Internal.wire_name(summary, :total_iterations) == "totalIterations"
      assert Internal.wire_name(summary, "totalIterations") == "totalIterations"
      assert Internal.wire_name(summary, :notAnAttribute) == "notAnAttribute"

      summary
      |> Internal.fetch_plan(Internal.wire_name(summary, :total_iterations))
      |> Wire.assert_wire_command("ml_fetch_summary")
    end

    # The arguments. The probe proves the server *accepts* what this package sends; only these
    # prove it is what PySpark sends. Each argument's type comes from `priv/ml_attributes.exs`,
    # so a fixture that matches is also the registry's types being right on the wire.
    test "a fetch carrying a Vector argument" do
      model = %Model{
        model()
        | class: "org.apache.spark.ml.classification.LogisticRegressionModel"
      }

      argument = Internal.arguments!(model, "predict", [Nx.tensor([1.5, -2.5], type: :f64)])

      model
      |> Internal.fetch_plan(:predict, argument)
      |> Wire.assert_wire_command("ml_fetch_predict")
    end

    # The argument hangs off the *last* method, not the first: the chain is `summary` then
    # `fMeasureByLabel(0.5)`, and only the second carries anything.
    test "a fetch carrying a double argument, after a chain" do
      summary = summary()
      argument = Internal.arguments!(summary, "fMeasureByLabel", [0.5])

      summary
      |> Internal.fetch_plan(:fMeasureByLabel, argument)
      |> Wire.assert_wire_command("ml_fetch_f_measure_by_label")
    end
  end

  # The `ConnectHelper` route, one test per shape it answers in. What these pin that the probe
  # cannot: the object is named by a literal id rather than a cache entry, the arguments are
  # positional and complete, and the one nested list in the surface is built the way PySpark
  # builds it.
  describe "the helper object" do
    test "a method that answers with a value" do
      :stop_words_remover_load_default_stop_words
      |> Registry.helper!()
      |> Internal.helper_plan(["german"])
      |> Wire.assert_wire_command("ml_helper_stop_words")
    end

    # The uid is the **client's** and is the call's first argument: it is how the model the
    # server registers and the one the caller holds end up agreeing on one.
    test "a method that builds a model, with the uid it will carry" do
      :string_indexer_model_from_labels
      |> Registry.helper!()
      |> Internal.helper_plan([Wire.helper_uid(), ["a", "b"]])
      |> Wire.assert_wire_command("ml_helper_from_labels")
    end

    # The one nested list in the whole surface, and the reason `Latu.ML.Plan.literal/2` grew a
    # recursive arm. PySpark types this one explicitly at the call site because a list of lists
    # cannot be inferred from its value.
    test "a nested list, typed explicitly at the call site" do
      :string_indexer_model_from_labels_array
      |> Registry.helper!()
      |> Internal.helper_plan([Wire.helper_uid(), [["a", "b"], ["c"]]])
      |> Wire.assert_wire_command("ml_helper_from_arrays_of_labels")
    end

    # A relation, and the only one in this file that carries a frame as an *argument* with no
    # model anywhere: `model_summary_dataset` stays nil because there is no summary to rebuild
    # and nothing to rebuild it from.
    test "a method that answers with a DataFrame" do
      :chi_square_test
      |> Registry.helper!()
      |> Internal.helper_relation_plan([Wire.base(), :features, :label, true])
      |> Wire.assert_wire("ml_helper_chi_square_test")
    end

    # The operator arm, through the facade rather than the builder, because what is under test
    # is the filling-in: two params set, the rest Spark's defaults, and `weightCol` the empty
    # string PySpark's own call site sends for a param Spark gives no default at all.
    test "an operator whose params ride the call as positional arguments" do
      [k: 3, max_iter: 7]
      |> Clustering.power_iteration_clustering()
      |> ML.assign_clusters(Wire.base())
      |> Wire.assert_wire("ml_helper_assign_clusters")
    end

    # The other half of the nested-list question: `Bucketizer.splitsArray` is the one nested
    # *param* in 4.2.0 and it goes through PySpark's inference rather than an explicit type, so
    # this is what says the two paths write the same bytes.
    test "and the nested param, whose type PySpark infers" do
      [
        splits_array: [[-1.0e9, 0.0, 1.0e9], [-1.0e9, 1.0, 1.0e9]],
        input_cols: [:x1, :x2],
        output_cols: [:b1, :b2]
      ]
      |> Feature.bucketizer()
      |> pinned(Wire.assembler_uid())
      |> ML.transform(Wire.base())
      |> Wire.assert_wire("ml_transform_bucketizer_splits_array")
    end
  end

  # `Latu.ML.Linalg`'s own round trip proves this package can read what it writes. These prove
  # PySpark can — the element order, the null placeholders, and the deprecated `struct_type`
  # field that both still write.
  describe "linalg literals" do
    test "a dense vector" do
      Nx.tensor([1.5, -2.5], type: :f64)
      |> Linalg.to_literal()
      |> Wire.assert_wire_message("ml_literal_dense_vector")
    end

    test "a sparse vector" do
      %SparseVector{size: 4, indices: [1, 3], values: [2.0, 4.0]}
      |> Linalg.to_literal()
      |> Wire.assert_wire_message("ml_literal_sparse_vector")
    end

    test "a dense matrix, row-major" do
      Nx.tensor([[1.0, 2.0], [3.0, 4.0]], type: :f64)
      |> Linalg.to_literal()
      |> Wire.assert_wire_message("ml_literal_dense_matrix")
    end

    # Where the square fixture cannot: numRows before numCols, on the wire, against PySpark.
    test "a non-square dense matrix" do
      Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], type: :f64)
      |> Linalg.to_literal()
      |> Wire.assert_wire_message("ml_literal_dense_matrix_wide")
    end
  end
end
