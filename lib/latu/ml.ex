defmodule Latu.ML do
  @moduledoc """
  Spark MLlib from Elixir, over Spark Connect.

  The algorithms run on the cluster, where the data is; the Elixir side holds plans and
  handles. That is the whole difference from `Scholar`, which runs on one node in memory and
  hands you a model you can inspect and ship. Reach for Scholar when the training set fits a
  node; reach for this when it does not, or when the features already live in a lakehouse.

  ## The shapes

  Two, and Spark chose them, not this package:

    * **Actions** reach the server and return `{:ok, _} | {:error, %Latu.Error{}}`, with `!`
      twins that raise — `fit/2`, `evaluate/2`, `attribute/2`, `save/3`, `load/3`, `delete/1`.
    * **Lazy builders** take a struct and hand one back, touching nothing — `transform/2`
      returns a `Latu.DataFrame` and no bytes move until you collect it.

  ## A model is not a value

  `fit/2` hands back a `Latu.ML.Model`: a reference into a per-session cache on the server,
  which offloads to disk under memory pressure and is gone when the session ends. Nothing on
  the BEAM can notice a model going away, and there is no finalizer to lean on, so the caller
  says when it ends — `with_model/3` where the model's life fits an expression, `fit/2` plus
  `delete/1` where it does not.

  ## Example

      alias Latu.ML
      alias Latu.ML.{Classification, Feature}

      df = Latu.sql!(session, "SELECT * FROM training")

      assembler = Feature.vector_assembler(input_cols: [:x1, :x2], output_col: :features)
      features = ML.transform(assembler, df)

      lr = Classification.logistic_regression(max_iter: 10, reg_param: 0.01)

      {:ok, coefficients} =
        ML.with_model(lr, features, fn model ->
          ML.attribute(model, :coefficients)
        end)
  """

  require Logger

  # A dropped summary. `error_class` carries the **dotted** name, not the bare one —
  # `dev/probe_ml.exs` printed it against a 4.2.0 server rather than this being read off a
  # source file, which matters because the roadmap and Spark's own docs write these bare.
  @summary_lost "CONNECT_ML.MODEL_SUMMARY_LOST"

  alias Latu.DataFrame
  alias Latu.Error
  alias Latu.ML.CrossValidator
  alias Latu.ML.CrossValidatorModel
  alias Latu.ML.Estimator
  alias Latu.ML.Evaluator
  alias Latu.ML.Helper
  alias Latu.ML.Internal
  alias Latu.ML.Layout
  alias Latu.ML.Model
  alias Latu.ML.Pipeline
  alias Latu.ML.PipelineModel
  alias Latu.ML.Operator
  alias Latu.ML.Param
  alias Latu.ML.Plan
  alias Latu.ML.Registry
  alias Latu.ML.Result
  alias Latu.ML.Summary
  alias Latu.ML.TrainValidationSplit
  alias Latu.ML.TrainValidationSplitModel
  alias Latu.ML.Transformer

  @typedoc """
  One object the session's ML cache is holding, as `cache_info/1` reports it.

  `size` is the bytes the cache charged the object: `0` for a summary always, and `0` for a
  model too where the session has memory control off.
  """
  @type cache_entry :: %{id: String.t(), class: String.t(), size: non_neg_integer()}

  @typedoc """
  Anything `save/3` writes and `load/3` answers with: a fitted model, or an operator that has
  not been fitted.
  """
  @type persistable ::
          Model.t()
          | Estimator.t()
          | Transformer.t()
          | Evaluator.t()
          | Pipeline.t()
          | PipelineModel.t()

  @typedoc """
  How `load/3` names what is at a path — an operator name from the registry, a generated model
  module, or `{class, kind}` for a class this package does not know.
  """
  @type loadable :: atom() | {String.t(), Plan.kind()}

  @doc """
  Fit an estimator, or a whole pipeline, on a frame.

  An action: the fit runs on the server and the result is a handle. The frame is where the
  session comes from, so there is no session argument.

  The model carries the estimator's uid and params, because a `Fit` answers with an object
  reference and nothing else — no uid, no params, no warning.

  ## A pipeline

  A `Latu.ML.Pipeline` is not a server operator: there is no `Fit` for one on the wire, so this
  folds over its stages and does one action per estimator, exactly as PySpark does. What comes
  back is a `Latu.ML.PipelineModel` that **owns** every model it fitted — `delete/1` on it
  releases them all, and `with_model/3` brackets the whole thing.

  The fold has one surprise worth stating. The frame is transformed by each stage on the way
  through, but only **up to the last estimator**: stages after it are carried into the pipeline
  model untouched, because there is nothing left to learn from them. And if a later stage fails,
  the models fitted before it are released before the error is returned, so a half-built
  pipeline leaks nothing.

      {:ok, model} = ML.fit(ML.pipeline([assembler, scaler, lr]), training)

  ## A grid search

  A `Latu.ML.CrossValidator` or `Latu.ML.TrainValidationSplit` has no `Fit` on the wire either.
  This runs the search: `num_folds * length(param_maps)` fits, each scored by the evaluator on
  its held-out slice, and then **one more** — the winning param map refit on the whole frame,
  because the fold models each saw only a fraction of the rows.

  Serial, one fit at a time. PySpark runs the grid on a thread pool; whether several processes
  can drive one session's channel at once is not something this package has measured, so it
  does not claim it. `docs/decisions.md`.

  A sub-model is released as soon as its metric is read, so a 5-by-6 search holds one cache
  entry rather than thirty. `collect_sub_models: true` keeps them and hands them back, and they
  are then yours — `delete/1` on the returned model releases them along with the winner.

      cv = ML.cross_validator(estimator: lr, param_maps: grid, evaluator: evaluator)
      {:ok, searched} = ML.fit(cv, training)
      searched.avg_metrics
  """
  @spec fit(Estimator.t(), DataFrame.t()) :: {:ok, Model.t()} | {:error, Error.t()}
  def fit(%Estimator{} = estimator, %DataFrame{} = data) do
    plan = Internal.fit_plan(estimator, data.plan)

    with {:ok, execution} <- Latu.Client.execute_command(data.session, plan),
         {:ok, ref} <- Result.object_ref(execution) do
      {:ok,
       %Model{
         ref: ref,
         session: data.session,
         class: estimator.model_class,
         candidates: estimator.model_candidates,
         uid: estimator.uid,
         dataset: data.plan,
         params: estimator.params
       }}
    end
  end

  @spec fit(Pipeline.t(), DataFrame.t()) :: {:ok, PipelineModel.t()} | {:error, Error.t()}
  def fit(%Pipeline{} = pipeline, %DataFrame{} = data) do
    last = last_estimator(pipeline.stages)

    pipeline.stages
    |> Enum.with_index()
    |> Enum.reduce_while({[], data}, fn {stage, index}, {done, frame} ->
      case fit_stage(stage, frame, index, last) do
        {:ok, fitted, next} -> {:cont, {[fitted | done], next}}
        {:error, error} -> {:halt, {:error, error, done}}
      end
    end)
    |> case do
      {done, _frame} ->
        {:ok,
         %PipelineModel{
           uid: Internal.uid(Layout.class(:pipeline_model)),
           session: data.session,
           stages: Enum.reverse(done)
         }}

      {:error, error, done} ->
        # Nothing asked for these, and nothing else will give them back. Only some of the
        # stages hold anything on the server — a transformer carried through holds nothing —
        # so `cached_models/1` picks out the ones that do, and they go back in one `Delete`.
        done |> Enum.flat_map(&cached_models/1) |> release()

        {:error, error}
    end
  end

  @spec fit(CrossValidator.t(), DataFrame.t()) ::
          {:ok, CrossValidatorModel.t()} | {:error, Error.t()}
  def fit(%CrossValidator{} = validator, %DataFrame{} = data) do
    search(validator, data, fn best_model, scored, sub_models ->
      # Mean and **population** standard deviation across folds, one pair per param map.
      # `np.std`'s default is `ddof=0` and PySpark takes the default, so a sample standard
      # deviation here would disagree with PySpark by a factor no test would obviously catch.
      across = transpose(scored)

      %CrossValidatorModel{
        uid: Internal.uid("org.apache.spark.ml.tuning.CrossValidatorModel"),
        session: data.session,
        best_model: best_model,
        avg_metrics: Enum.map(across, &mean/1),
        std_metrics: Enum.map(across, &population_std/1),
        sub_models: sub_models,
        validator: validator
      }
    end)
  end

  @spec fit(TrainValidationSplit.t(), DataFrame.t()) ::
          {:ok, TrainValidationSplitModel.t()} | {:error, Error.t()}
  def fit(%TrainValidationSplit{} = validator, %DataFrame{} = data) do
    search(validator, data, fn best_model, scored, sub_models ->
      %TrainValidationSplitModel{
        uid: Internal.uid("org.apache.spark.ml.tuning.TrainValidationSplitModel"),
        session: data.session,
        best_model: best_model,
        validation_metrics: hd(scored),
        sub_models: sub_models && hd(sub_models),
        validator: validator
      }
    end)
  end

  # The index of the last estimator, or -1 where there is none: everything at or before it is
  # applied to the frame on the way through, and everything after it is carried as it is.
  defp last_estimator(stages) do
    stages
    |> Enum.with_index()
    |> Enum.reduce(-1, fn {stage, index}, last ->
      if estimator?(stage), do: index, else: last
    end)
  end

  defp estimator?(%Estimator{}), do: true
  defp estimator?(%Pipeline{}), do: true
  defp estimator?(_stage), do: false

  defp fit_stage(stage, frame, index, last) when index > last, do: {:ok, stage, frame}

  defp fit_stage(stage, frame, index, last) do
    if estimator?(stage) do
      with {:ok, fitted} <- fit(stage, frame) do
        # The last estimator's own output is never needed: nothing after it is fitted.
        {:ok, fitted, if(index < last, do: transform(fitted, frame), else: frame)}
      end
    else
      {:ok, stage, transform(stage, frame)}
    end
  end

  @doc "See `fit/2`. Raises instead of returning an error."
  @spec fit!(
          Estimator.t() | Pipeline.t() | CrossValidator.t() | TrainValidationSplit.t(),
          DataFrame.t()
        ) :: Model.t() | PipelineModel.t() | searched()
  def fit!(estimator, %DataFrame{} = data) do
    bang!(fit(estimator, data))
  end

  @doc """
  A pipeline: stages fitted as one, in order.

  A builder, not an action — there is no `Fit` for a pipeline on the wire, so this reaches no
  server and needs no session. `Latu.ML.fit/2` is what folds over the stages, and what it hands
  back is a `Latu.ML.PipelineModel`.

  A stage is an estimator, a transformer, a model already fitted, or another pipeline. Anything
  else is refused here rather than partway through a fit that has already cached two models.

      ML.pipeline([
        Feature.vector_assembler(input_cols: [:x1, :x2], output_col: :features),
        Feature.standard_scaler(input_col: :features, output_col: :scaled),
        Classification.logistic_regression(features_col: :scaled, max_iter: 10)
      ])
  """
  @spec pipeline([Pipeline.stage()]) :: Pipeline.t()
  def pipeline(stages) when is_list(stages) do
    Enum.each(stages, &stage!/1)

    %Pipeline{uid: Internal.uid(Layout.class(:pipeline)), stages: stages}
  end

  @stages [Estimator, Transformer, Model, Pipeline, PipelineModel]

  defp stage!(%module{}) when module in @stages, do: :ok

  defp stage!(other) do
    raise ArgumentError,
          "#{inspect(other)} is not a pipeline stage. A stage is an estimator, a transformer, " <>
            "a model already fitted, or another pipeline."
  end

  @doc """
  Apply a transformer or a fitted model to a frame.

  A lazy builder, not an action: the frame it hands back carries the ML relation and nothing
  has run. `Latu.schema/1` will answer from it without executing, and `Latu.collect/2` is
  what finally reaches the server.
  """
  @spec transform(Transformer.t() | Model.t() | PipelineModel.t() | searched(), DataFrame.t()) ::
          DataFrame.t()
  def transform(operator, data)

  def transform(%Transformer{} = transformer, %DataFrame{} = data) do
    %DataFrame{
      session: data.session,
      plan: Internal.transform_relation(transformer, data.plan)
    }
  end

  def transform(%Model{} = model, %DataFrame{} = data) do
    %DataFrame{session: data.session, plan: Internal.transform_relation(model, data.plan)}
  end

  # A pipeline model is every stage in turn, and still a lazy builder: what comes back carries
  # each stage's relation and nothing has run.
  # A search result scores with its winner and nothing else. The fold models are not an
  # ensemble — they are a measurement that has already been taken.
  def transform(%CrossValidatorModel{} = model, %DataFrame{} = data) do
    transform(model.best_model, data)
  end

  def transform(%TrainValidationSplitModel{} = model, %DataFrame{} = data) do
    transform(model.best_model, data)
  end

  def transform(%PipelineModel{} = model, %DataFrame{} = data) do
    Enum.reduce(model.stages, data, &transform/2)
  end

  @doc """
  Score a frame with an evaluator: one number comes back.

  An action, and the third of Spark's three verbs. The frame is where the session comes from,
  as it is for `fit/2`, and it is a frame a model has already scored — an evaluator reads
  `prediction_col` and `label_col`, not features. Which number it is is the evaluator's
  `metric_name`, whose default is its own: `areaUnderROC` for a binary classifier, `rmse` for
  a regressor.

      scored = ML.transform(model, test)
      {:ok, auc} = ML.evaluate(Evaluation.binary_classification_evaluator(), scored)

  Nothing is cached: unlike a fit, an evaluate answers with the metric itself.
  """
  @spec evaluate(Evaluator.t(), DataFrame.t()) :: {:ok, float()} | {:error, Error.t()}
  def evaluate(%Evaluator{} = evaluator, %DataFrame{} = data) do
    plan = Internal.evaluate_plan(evaluator, data.plan)

    with {:ok, execution} <- Latu.Client.execute_command(data.session, plan) do
      Result.param(execution)
    end
  end

  @doc "See `evaluate/2`. Raises instead of returning an error."
  @spec evaluate!(Evaluator.t(), DataFrame.t()) :: float()
  def evaluate!(%Evaluator{} = evaluator, %DataFrame{} = data) do
    bang!(evaluate(evaluator, data))
  end

  @doc """
  Whether a bigger number from `evaluate/2` is a better one, for this evaluator.

  Pure — it reaches no server, because there is nothing there to ask. `isLargerBetter` is not
  on the server's allowlist, so PySpark's six evaluators each override it client-side to make
  it work over Connect, and those overrides are what the registry extracts. This resolves one
  against the evaluator's own `metric_name`, set or default.

      iex> Latu.ML.larger_better?(Latu.ML.Evaluation.regression_evaluator())
      false
      iex> Latu.ML.larger_better?(Latu.ML.Evaluation.regression_evaluator(metric_name: "r2"))
      true

  Cross-validation and train/validation split pick their best model by this, which is why it
  is public: a caller comparing metrics by hand should not have to remember that `rmse` is
  minimised and `r2` maximised.
  """
  @spec larger_better?(Evaluator.t()) :: boolean()
  defdelegate larger_better?(evaluator), to: Internal

  # Every kind that is a directory this package wrote rather than something the server reads.
  @meta_kinds Internal.meta_algorithms()

  @typedoc """
  What a grid search hands back: the winner, and the score of everything it tried.

  `transform/2` scores with the winner, and `delete/1` releases it along with any fold models
  the search was asked to keep.
  """
  @type searched :: CrossValidatorModel.t() | TrainValidationSplitModel.t()

  @typedoc "Anything `delete/1` can release: a model, a composite that owns models, or a list."
  @type deletable :: Model.t() | PipelineModel.t() | searched()

  @typedoc """
  One combination from a grid: which operator's param, by uid, and what to set it to.

  A uid rather than a struct because the operator being tuned is often a stage of a pipeline,
  and because it is how Spark's saved format keys them (`parent` and `name`). Build these with
  `param_grid/2` rather than by hand.
  """
  @type param_map :: [{String.t(), atom(), term()}]

  @doc """
  A grid of params to try, as data.

  Every combination of the values given, which is what cross-validation and train/validation
  split search over. A builder in the sense that nothing runs: no session, no server, and the
  params are checked against the operator's own table now rather than after ten fits.

      grid = ML.param_grid(lr, reg_param: [0.1, 0.01], max_iter: [10, 100])
      length(grid)
      #=> 4

  Each entry says **which** operator's param it sets, by uid. That matters because the thing
  being tuned is often a pipeline and the param belongs to one stage of it:

      lr = Classification.logistic_regression()
      pipeline = ML.pipeline([assembler, lr])
      grid = ML.param_grid(lr, reg_param: [0.1, 0.01])

      ML.cross_validator(estimator: pipeline, param_maps: grid, evaluator: evaluator)

  To vary params on two operators at once, pass the pairs together — the product is taken
  across all of them, not per operator:

      ML.param_grid([{assembler, [handle_invalid: ["skip", "keep"]]}, {lr, [max_iter: [10]]}])

  ## Order

  The **last** param given varies fastest, as `itertools.product` does it in PySpark. This is
  worth knowing because two param maps that score identically are separated by nothing but
  their position: the best one is the first maximum.
  """
  @spec param_grid(Estimator.t() | Transformer.t() | Evaluator.t(), keyword([term()])) ::
          [param_map()]
  def param_grid(operator, grid) when is_list(grid), do: param_grid([{operator, grid}])

  @doc """
  See `param_grid/2`. Takes the pairs together, so the product spans several operators.
  """
  @spec param_grid([{Estimator.t() | Transformer.t() | Evaluator.t(), keyword([term()])}]) ::
          [param_map()]
  defdelegate param_grid(grids), to: Internal

  # One search, both kinds. `assemble` is the only difference: what a fold's scores mean.
  defp search(validator, %DataFrame{} = data, assemble) do
    Internal.covered!(validator)

    with {:ok, scored, kept} <- score_folds(validator, Internal.folds(data, validator)),
         {:ok, best} <- refit(validator, data, scored, kept) do
      {:ok, assemble.(best, scored, if(validator.collect_sub_models, do: kept))}
    end
  end

  # The winner is refit on everything. A failure here has to give back what the folds kept,
  # because nothing else will — the caller is holding an error, not a model.
  defp refit(validator, data, scored, kept) do
    index = best_of(fold_metrics(validator, scored), validator.evaluator)
    param_map = Enum.at(validator.param_maps, index)

    case fit(Internal.apply_params(validator.estimator, param_map), data) do
      {:ok, model} ->
        {:ok, model}

      {:error, error} ->
        release(collected(kept))
        {:error, error}
    end
  end

  defp fold_metrics(%CrossValidator{}, scored), do: scored |> transpose() |> Enum.map(&mean/1)
  defp fold_metrics(%TrainValidationSplit{}, scored), do: hd(scored)

  defp score_folds(validator, folds) do
    folds
    |> Enum.reduce_while({:ok, [], []}, fn {train, validation}, {:ok, scored, kept} ->
      case score_fold(validator, train, validation) do
        {:ok, metrics, models} ->
          {:cont, {:ok, [metrics | scored], [models | kept]}}

        {:error, error, models} ->
          {:halt, {:error, error, collected([models | kept])}}
      end
    end)
    |> case do
      {:ok, scored, kept} ->
        {:ok, Enum.reverse(scored), Enum.reverse(kept)}

      {:error, error, orphans} ->
        release(orphans)
        {:error, error}
    end
  end

  # One fold, every param map. The two frames are persisted for the duration: each is read once
  # per param map, and re-deriving a filtered random draw that many times is the one avoidable
  # cost in a search. PySpark does the same, for the same reason.
  defp score_fold(validator, train, validation) do
    with {:ok, train} <- Latu.persist(train),
         {:ok, validation} <- Latu.persist(validation) do
      result = score_grid(validator, train, validation)

      Latu.unpersist(train)
      Latu.unpersist(validation)

      case result do
        {:ok, metrics, kept} -> {:ok, Enum.reverse(metrics), Enum.reverse(kept)}
        other -> other
      end
    else
      {:error, error} -> {:error, error, []}
    end
  end

  defp score_grid(validator, train, validation) do
    Enum.reduce_while(validator.param_maps, {:ok, [], []}, fn param_map, {:ok, metrics, kept} ->
      searched = Internal.apply_params(validator.estimator, param_map)

      case fit(searched, train) do
        {:ok, model} ->
          case evaluate(validator.evaluator, transform(model, validation)) do
            {:ok, metric} ->
              # A sub-model nobody asked for is a cache entry with no owner, so it goes back
              # the moment its metric is read. A 5-by-6 search then holds one at a time rather
              # than thirty. `collect_sub_models: true` keeps them, and they become the
              # caller's — `delete/1` on the returned model releases them.
              if validator.collect_sub_models do
                {:cont, {:ok, [metric | metrics], [model | kept]}}
              else
                release(model)
                {:cont, {:ok, [metric | metrics], kept}}
              end

            {:error, error} ->
              release(model)
              {:halt, {:error, error, kept}}
          end

        {:error, error} ->
          {:halt, {:error, error, kept}}
      end
    end)
  end

  defp collected(kept), do: kept |> List.flatten() |> Enum.flat_map(&cached_models/1)

  defp transpose([]), do: []
  defp transpose(rows), do: rows |> Enum.zip() |> Enum.map(&Tuple.to_list/1)

  defp mean(values), do: Enum.sum(values) / length(values)

  defp population_std(values) do
    average = mean(values)

    :math.sqrt(Enum.sum(Enum.map(values, &((&1 - average) * (&1 - average)))) / length(values))
  end

  @doc """
  Which param map won, as an index into the validator's `param_maps`.

  Pure. The metric is maximised or minimised according to `larger_better?/1` on the evaluator
  that scored it, and ties go to the **first** — so the order `param_grid/2` built the grid in
  is what separates two param maps that scored the same.
  """
  @spec best_index(CrossValidatorModel.t() | TrainValidationSplitModel.t()) :: non_neg_integer()
  def best_index(%CrossValidatorModel{avg_metrics: metrics, validator: validator}) do
    best_of(metrics, validator.evaluator)
  end

  def best_index(%TrainValidationSplitModel{validation_metrics: metrics, validator: validator}) do
    best_of(metrics, validator.evaluator)
  end

  defp best_of(metrics, evaluator) do
    # Strictly better, so an equal metric never displaces the one already held: `np.argmax`
    # and `np.argmin` both return the first, and this is a search whose answer people compare
    # against PySpark's.
    better = if larger_better?(evaluator), do: &Kernel.>/2, else: &Kernel.</2

    metrics
    |> Enum.with_index()
    |> Enum.reduce(fn {metric, index}, {best, at} ->
      if better.(metric, best), do: {metric, index}, else: {best, at}
    end)
    |> elem(1)
  end

  @doc """
  A k-fold cross-validation, as data. See `Latu.ML.CrossValidator`.

  A builder, not an action: nothing here reaches a server and no session is needed, because a
  validator has no wire form at all. `fit/2` is what searches.

      cv =
        ML.cross_validator(
          estimator: lr,
          param_maps: ML.param_grid(lr, reg_param: [0.1, 0.01]),
          evaluator: Evaluation.binary_classification_evaluator(),
          num_folds: 3,
          seed: 42
        )

  `estimator:`, `param_maps:` and `evaluator:` are required. `num_folds:` defaults to 3,
  `seed:` to none — and with no seed the fold cut differs between builds, so pass one for a
  search you can repeat. `fold_col:` names a column that already holds a fold number, in which
  case no draw is added at all. `collect_sub_models: true` keeps every fold's models instead of
  releasing them as they are scored.
  """
  @spec cross_validator(keyword()) :: CrossValidator.t()
  def cross_validator(opts) when is_list(opts) do
    {required, rest} = validator_options!(opts, [:num_folds, :fold_col, :collect_sub_models])

    struct!(
      CrossValidator,
      Map.merge(required, Map.new(rest))
      |> Map.put(:uid, Internal.uid("org.apache.spark.ml.tuning.CrossValidator"))
    )
  end

  @doc """
  A single train/validation split, as data. See `Latu.ML.TrainValidationSplit`.

  The same builder as `cross_validator/1` with one split instead of k folds, so each param map
  is scored once rather than averaged. `train_ratio:` defaults to 0.75.

      tvs =
        ML.train_validation_split(
          estimator: lr,
          param_maps: ML.param_grid(lr, reg_param: [0.1, 0.01]),
          evaluator: Evaluation.binary_classification_evaluator()
        )
  """
  @spec train_validation_split(keyword()) :: TrainValidationSplit.t()
  def train_validation_split(opts) when is_list(opts) do
    {required, rest} = validator_options!(opts, [:train_ratio, :collect_sub_models])

    struct!(
      TrainValidationSplit,
      Map.merge(required, Map.new(rest))
      |> Map.put(:uid, Internal.uid("org.apache.spark.ml.tuning.TrainValidationSplit"))
    )
  end

  # The three every validator needs, plus `seed:`, plus whatever else that kind takes. Checked
  # here rather than left to `struct!/2` so a missing evaluator says what is missing and an
  # option meant for the other kind says which one it belongs to.
  defp validator_options!(opts, extra) do
    allowed = [:estimator, :param_maps, :evaluator, :seed | extra]

    for {name, _value} <- opts, name not in allowed do
      raise ArgumentError,
            "#{inspect(name)} is not an option here. This one takes: " <>
              Enum.map_join(allowed, ", ", &inspect/1)
    end

    required =
      Map.new([:estimator, :param_maps, :evaluator], fn name ->
        case Keyword.fetch(opts, name) do
          {:ok, value} -> {name, value}
          :error -> raise ArgumentError, "a validator needs #{inspect(name)}, and this has none"
        end
      end)

    stage!(required.estimator)

    {required, Keyword.drop(opts, [:estimator, :param_maps, :evaluator])}
  end

  @doc """
  Assign a cluster to every vertex of an affinity matrix — Power Iteration Clustering.

  A lazy builder, not an action: the frame it hands back carries a `Fetch` on the server's own
  helper object and nothing has run. What goes in is an edge list — `src`, `dst` and `weight`
  by default — and what comes back is `id` and `cluster`.

  `PowerIterationClustering` is neither fitted nor applied; this is the one thing it does, and
  that is why it is a verb here rather than a model you keep. Its params ride the call as
  positional arguments, so every one is sent whether you set it or not.

      pic = Latu.ML.Clustering.power_iteration_clustering(k: 2, max_iter: 10)
      clusters = Latu.ML.assign_clusters(pic, edges)
  """
  @spec assign_clusters(Helper.t(), DataFrame.t()) :: DataFrame.t()
  def assign_clusters(%Helper{} = pic, %DataFrame{} = data) do
    helper_operator(pic, data, "org.apache.spark.ml.clustering.PowerIterationClustering")
  end

  @doc """
  Find the frequent sequential patterns in a frame of sequences — PrefixSpan.

  `assign_clusters/2`'s twin in shape: a lazy builder over the helper object, for the other
  operator that is neither fitted nor applied. The input column is `sequence`, an
  `array<array<T>>`; what comes back is `sequence` and `freq`.

      ps = Latu.ML.FPM.prefix_span(min_support: 0.5, max_pattern_length: 5)
      patterns = Latu.ML.find_frequent_sequential_patterns(ps, sequences)
  """
  @spec find_frequent_sequential_patterns(Helper.t(), DataFrame.t()) :: DataFrame.t()
  def find_frequent_sequential_patterns(%Helper{} = span, %DataFrame{} = data) do
    helper_operator(span, data, "org.apache.spark.ml.fpm.PrefixSpan")
  end

  # The class check one layer before the server's, as the generated accessors do it: calling
  # `assign_clusters/2` on a `PrefixSpan` names the verb that would have taken it rather than
  # sending a call whose arguments happen to line up.
  defp helper_operator(%Helper{class: class} = operator, data, class) do
    owner = class |> String.split(".") |> List.last()

    case Registry.helpers_of(owner) do
      [row] ->
        %DataFrame{
          session: data.session,
          plan: Internal.helper_relation_plan(row, Internal.operator_values!(row, operator, data))
        }

      other ->
        raise ArgumentError,
              "#{class} has #{length(other)} helper methods, and this verb assumes one"
    end
  end

  defp helper_operator(%Helper{class: actual}, _data, expected) do
    raise ArgumentError, "that verb is #{expected}'s, and this operator is a #{actual}"
  end

  @doc """
  Read a model attribute whose answer is a value — a coefficient, an intercept, a count.

  An action. A `Vector` or `Matrix` comes back as an `Nx.Tensor`, or a
  `Latu.ML.SparseVector` where densifying would be this package's decision rather than yours.

  **Either spelling.** `Latu.ML.attributes/1` lists snake_case names and the server's allowlist
  is camelCase, so `:area_under_roc` and `"areaUnderROC"` both work and the table is a list of
  things you can actually ask for. A name on the allowlist is taken as written, so a wire name
  never means anything else.

  A name neither spelling accounts for goes to the server as written and comes back as
  `CONNECT_ML.ATTRIBUTE_NOT_ALLOWED`: a typo is its error and not this package's, and that is
  also what keeps a class the registry does not know reachable at all. The generated accessor
  modules — `Latu.ML.Classification.LogisticRegressionModel` and its 41 siblings — are that
  allowlist as named functions and are what to reach for first; this is the layer underneath.
  """
  @spec attribute(Model.t() | Summary.t(), atom() | String.t()) ::
          {:ok, term()} | {:error, Error.t()}
  def attribute(holder, name), do: attribute(holder, name, [])

  @doc """
  Read an attribute that takes arguments — `predict`, `fMeasureByLabel`, `recommendForAllUsers`.

  An action, and the same `Fetch` as `attribute/2` with the arguments attached to its last
  method. **The types are not yours to state**: `priv/ml_attributes.exs` records them from
  PySpark's own annotations, so `predict` knows it wants a `Vector` and `recommendForAllUsers`
  an `int`, and a wrong count or a wrong kind is refused here rather than on the server.

  The generated accessors are these too, and are what to reach for first —
  `Latu.ML.Classification.LogisticRegressionModel.predict/2` is this call with the class
  checked and the arguments named.

      {:ok, label} = Latu.ML.attribute(model, "predict", [Nx.tensor([1.0, 0.0])])
      {:ok, f1} = Latu.ML.attribute(summary, "fMeasureByLabel", [1.0])

  A DataFrame argument is passed as one and rides the relation arm — `Latu.ML.attribute(model,
  "evaluateEachIteration", [frame, "logLoss"])`. Attributes that answer with a *frame* go to
  `attribute_frame/3` instead. The `evaluate` **attribute** — a model's, not the evaluator verb
  `evaluate/2` — answers with another cached object rather than a value, and has no verb of its
  own: asking for it here reaches the server and comes back as a protocol error naming the arm
  it answered on.

  Needs the model's class, since that is what the types are recorded against. A model built by
  `Latu.ML.Estimator.new/2` without one has nothing to look them up in.
  """
  @spec attribute(Model.t() | Summary.t(), atom() | String.t(), [term()]) ::
          {:ok, term()} | {:error, Error.t()}
  def attribute(holder, name, args)

  def attribute(%Model{} = model, name, args) when is_list(args) do
    fetch(model, name, args)
  end

  def attribute(%Summary{} = summary, name, args) when is_list(args) do
    case fetch(summary, name, args) do
      # The one error this package handles rather than reports. A summary is dropped from the
      # cache rather than offloaded, and the caller did nothing to deserve it — so rebuild the
      # summary from the frame it was fitted on and ask once more, which is what PySpark does.
      #
      # Except where there is no frame: a model from `load/3` never had one, and a saved model
      # carries no summary, so there is nothing to rebuild and the server's refusal stands.
      {:error, %Error{error_class: @summary_lost}} when not is_nil(summary.dataset) ->
        with {:ok, _execution} <-
               Latu.Client.execute_command(
                 summary.session,
                 Internal.create_summary_plan(summary)
               ) do
          fetch(summary, name, args)
        end

      answer ->
        answer
    end
  end

  defp fetch(holder, name, args) do
    wire = Internal.wire_name(holder, name)
    plan = Internal.fetch_plan(holder, wire, Internal.arguments!(holder, wire, args))

    with {:ok, execution} <- Latu.Client.execute_command(holder.session, plan) do
      case Internal.answer_shape(holder, wire) do
        # The server packs several cache ids into one `ObjectRef` for these, and each is an
        # entry of its own — so they come back as models, and the caller owns them.
        :operators -> Internal.operators(holder, wire, execution)
        _value_or_unknown -> Result.param(execution)
      end
    end
  end

  @doc "See `attribute/2`. Raises instead of returning an error."
  @spec attribute!(Model.t() | Summary.t(), atom() | String.t()) :: term()
  def attribute!(holder, name), do: bang!(attribute(holder, name))

  @doc "See `attribute/3`. Raises instead of returning an error."
  @spec attribute!(Model.t() | Summary.t(), atom() | String.t(), [term()]) :: term()
  def attribute!(holder, name, args), do: bang!(attribute(holder, name, args))

  @doc """
  Read a model attribute whose answer is a DataFrame — `gaussiansDF`, `itemFactors` and their
  kind.

  A lazy builder, not an action: the same `Fetch` `attribute/2` sends, wrapped as a relation
  rather than a command, so this hands back a `Latu.DataFrame` and nothing has run.

  Which attributes answer this way is not a judgement call — the registry records it from
  PySpark's own `try_remote_attribute_relation` decorator, and the generated accessors are
  split by it. Asking for a literal-valued attribute here builds a relation the server will
  refuse.
  """
  @spec attribute_frame(Model.t() | Summary.t(), atom() | String.t()) :: DataFrame.t()
  def attribute_frame(holder, name), do: attribute_frame(holder, name, [])

  @doc """
  A frame-valued attribute that takes arguments — `describeTopics`, `recommendForAllUsers`,
  `approxNearestNeighbors`.

  `attribute_frame/2` and `attribute/3` in one: still a lazy builder, still typed from
  `priv/ml_attributes.exs` rather than from the caller.

      users = Latu.ML.attribute_frame(model, "recommendForAllUsers", [5])
      near = Latu.ML.attribute_frame(model, "approxNearestNeighbors", [df, key, 3, "distance"])
  """
  @spec attribute_frame(Model.t() | Summary.t(), atom() | String.t(), [term()]) :: DataFrame.t()
  def attribute_frame(holder, name, args)
      when (is_struct(holder, Model) or is_struct(holder, Summary)) and is_list(args) do
    wire = Internal.wire_name(holder, name)

    %DataFrame{
      session: holder.session,
      plan: Internal.fetch_relation(holder, wire, Internal.arguments!(holder, wire, args))
    }
  end

  @doc """
  Call a method on the server's own helper object.

  The escape hatch under `Latu.ML.Stat`, `Latu.ML.Feature.load_default_stop_words/2` and the
  generated `Latu.ML.Feature.StringIndexerModel.from_labels/3` and its siblings — the same
  relationship `attribute/3` has to the generated accessors.

  `ConnectHelper` is a singleton the server resolves by name: there is no model behind it,
  nothing is cached and nothing can be deleted. Its arguments are **positional and complete** —
  every declared one is sent, including the `uid` the three model-building methods take, which
  the named constructors generate and this does not.

      {:ok, locale} = ML.helper(session, :stop_words_remover_get_default_or_us)
      {:ok, words} = ML.helper(session, :stop_words_remover_load_default_stop_words, ["german"])

  A method whose answer is a DataFrame goes to `helper_frame/2` instead; asking for one here
  raises rather than building a relation the server would refuse. `Latu.ML.helpers/0` is the
  table of what there is.
  """
  @spec helper(Latu.Session.t(), atom(), [term()]) :: {:ok, term()} | {:error, Error.t()}
  def helper(session, name, args \\ []) do
    row = Registry.helper!(name)

    case row.returns do
      :frame ->
        raise ArgumentError, "#{row.method} answers with a DataFrame; use helper_frame/2"

      _value_or_model ->
        with {:ok, execution} <-
               Latu.Client.execute_command(session, Internal.helper_plan(row, args)) do
          helper_answer(row, session, args, execution)
        end
    end
  end

  # Three of the eleven build a model out of what was sent and answer with its cache reference,
  # exactly as a fit does — so the caller owns it. The uid is the one that went out with the
  # call, which is how the server's model and this one agree on one.
  defp helper_answer(%{returns: :model} = row, session, args, execution) do
    with {:ok, info} <- Result.operator_info(execution) do
      {:ok,
       %Model{
         ref: info.ref,
         session: session,
         class: row.model_class,
         uid: List.first(args)
       }}
    end
  end

  defp helper_answer(_row, _session, _args, execution), do: Result.param(execution)

  @doc "See `helper/3`. Raises instead of returning an error."
  @spec helper!(Latu.Session.t(), atom(), [term()]) :: term()
  def helper!(session, name, args \\ []), do: bang!(helper(session, name, args))

  @doc """
  Call a helper method whose answer is a DataFrame.

  A lazy builder, where `helper/3` is an action — the same split `attribute_frame/2` and
  `attribute/2` have, and the same `Fetch` underneath. Five of the eleven answer this way, and
  every one of them takes the frame it works on as an argument, so the session comes from
  there and there is none to pass.

      pairs = ML.helper_frame(:correlation, [frame, "features", "pearson"])
  """
  @spec helper_frame(atom(), [term()]) :: DataFrame.t()
  def helper_frame(name, args) do
    row = Registry.helper!(name)

    if row.returns != :frame do
      raise ArgumentError, "#{row.method} answers with a #{row.returns}; use helper/3"
    end

    plan = Internal.helper_relation_plan(row, args)

    %DataFrame{session: Internal.frame_of!(row, args).session, plan: plan}
  end

  @doc """
  The training summary a fitted model carries.

  A lazy builder: Spark has no separate cache key for a summary, so this composes the reference
  from the model's own and nothing is sent. Its attributes are on the generated summary module
  for the class — `Latu.ML.Classification.BinaryLogisticRegressionSummary` and its kind.

  Raises for a model class that has no summary, naming the ten that do. Ten of the 43 is not an
  oversight: a summary is what an estimator recorded *while fitting*, and most do not record
  one.

      summary = Latu.ML.summary(model)
      {:ok, iterations} = Latu.ML.attribute(summary, :totalIterations)

  A summary is the one object the server's cache **drops** rather than offloading, and a fetch
  that finds it gone answers `CONNECT_ML.MODEL_SUMMARY_LOST`. `attribute/2` recovers from that
  on its own, with the frame this struct carries; nothing is asked of the caller.
  """
  @spec summary(Model.t()) :: Summary.t()
  def summary(%Model{} = model), do: Internal.summary(model)

  @doc """
  Write a model or an unfitted operator to a path, in Spark's own on-disk format.

  An action. The path is the **server's**, not this machine's: the write happens where the
  session is, so a bare path is a path on the driver, and anything a second machine has to
  read wants a URL the cluster's filesystem understands (`s3://…`, `hdfs://…`).

  What comes out is Spark's own format, so PySpark and Scala read it as readily as `load/3`
  does — and the reverse, which is what the interop check in `dev/README.md` measures rather
  than assumes.

  ## Options

    * `:overwrite` — replace what is already at the path. Defaults to `false`, where the
      server refuses and names the path.
    * `:options` — the writer's own options, a map of strings, handed to Spark untouched.
    * `:session` — **required for an unfitted operator, and refused for a model.** An
      estimator, a transformer and an evaluator are inert data with no session behind them; a
      model carries its own, and a second one beside it would be a way to write to the wrong
      server.
    * `:sub_models` — **a search only.** Whether to write the fold models beside the winner.
      Defaults to whether the search collected any, which is PySpark's default too, though it
      reaches it through a lowercased key in the writer's option map rather than an option of
      its own. Asking for them where the search kept none is refused rather than silently
      writing nothing.

  ## A search

  A `Latu.ML.CrossValidator` and its model lay out a directory rather than a single operator:
  `metadata` with the validator's params and its grid, then `estimator/` and `evaluator/`, and
  for a fitted one `bestModel/` and the metrics. `subModels/foldN/M/` where they were kept — a
  split has no `foldN` level, because it has one.

      :ok = ML.save(searched, "/tmp/cv", overwrite: true, sub_models: true)

  ## Examples

      :ok = ML.save(model, "/tmp/lr", overwrite: true)
      :ok = ML.save(lr, "/tmp/lr-unfitted", session: session)
  """
  @spec save(persistable(), String.t(), keyword()) :: :ok | {:error, Error.t()}
  def save(target, path, opts \\ [])

  def save(%Model{} = model, path, opts) when is_binary(path) do
    write(model.session, model, path, Keyword.validate!(opts, overwrite: false, options: %{}))
  end

  def save(operator, path, opts)
      when is_binary(path) and
             (is_struct(operator, Estimator) or is_struct(operator, Transformer) or
                is_struct(operator, Evaluator)) do
    opts = Keyword.validate!(opts, [:session, overwrite: false, options: %{}])

    write(opts[:session] || no_session!(operator), operator, path, opts)
  end

  # A pipeline is not a server operator, so nothing here goes out as an `MlCommand.Write`: this
  # package writes the directory itself. `Latu.ML.Layout` is that format.
  def save(%CrossValidator{} = validator, path, opts) when is_binary(path) do
    save_search(validator, :cross_validator, path, opts, %{})
  end

  def save(%TrainValidationSplit{} = validator, path, opts) when is_binary(path) do
    save_search(validator, :train_validation_split, path, opts, %{})
  end

  def save(%CrossValidatorModel{} = model, path, opts) when is_binary(path) do
    save_searched(model, :cross_validator_model, path, opts, %{
      "avgMetrics" => model.avg_metrics,
      "stdMetrics" => model.std_metrics
    })
  end

  def save(%TrainValidationSplitModel{} = model, path, opts) when is_binary(path) do
    save_searched(model, :train_validation_split_model, path, opts, %{
      "validationMetrics" => model.validation_metrics
    })
  end

  def save(%Pipeline{} = pipeline, path, opts) when is_binary(path) do
    opts = Keyword.validate!(opts, [:session, overwrite: false, options: %{}])
    session = opts[:session] || no_session!(pipeline)

    save_stages(session, :pipeline, pipeline.uid, pipeline.stages, path, opts)
  end

  def save(%PipelineModel{} = model, path, opts) when is_binary(path) do
    opts = Keyword.validate!(opts, overwrite: false, options: %{})

    save_stages(model.session, :pipeline_model, model.uid, model.stages, path, opts)
  end

  def save(_target, path, _opts) when not is_binary(path) do
    raise ArgumentError, "a path is a string, not #{inspect(path)}"
  end

  def save(other, _path, _opts) do
    raise ArgumentError,
          "#{inspect(other)} is not something this package can save. A model, an estimator, " <>
            "a transformer or an evaluator is."
  end

  @doc "See `save/3`. Raises instead of returning an error."
  @spec save!(persistable(), String.t(), keyword()) :: :ok
  def save!(target, path, opts \\ []) do
    case save(target, path, opts) do
      :ok -> :ok
      {:error, error} -> raise error
    end
  end

  # A `Write` answers with an empty `MlCommandResult` — no arm set and nothing to read — so a
  # clean execution is the whole answer. PySpark ignores the result too.
  defp write(session, target, path, opts) do
    plan = Internal.save_plan(target, path, opts[:overwrite], opts[:options])

    with {:ok, _execution} <- Latu.Client.execute_command(session, plan), do: :ok
  end

  # A grid entry names its operator by uid and its param by this package's spelling; the file
  # wants the wire spelling. Both come off the operator itself rather than a global table,
  # because two stages of a pipeline can have params of the same name.
  defp wire_of(validator) do
    owners = owners_of(validator)

    fn uid, name ->
      operator = Map.fetch!(owners, uid)

      case Enum.find(operator.known, &(&1.name == name)) do
        nil -> raise ArgumentError, "#{uid} has no #{inspect(name)} to save"
        param -> param.wire
      end
    end
  end

  # And back: the file's wire spelling to this package's, against the operator the loaded
  # directory actually holds. A param the estimator does not have is a refusal, not a setting
  # quietly dropped on the floor.
  defp name_of(estimator, evaluator) do
    owners = owners_of(estimator, evaluator)

    fn uid, wire ->
      case Map.fetch(owners, uid) do
        :error ->
          raise ArgumentError,
                "the saved grid sets a param on #{uid}, which is not the estimator at this " <>
                  "path or any stage of it. The directory is inconsistent with itself."

        {:ok, operator} ->
          case Enum.find(operator.known, &(&1.wire == wire)) do
            nil -> raise ArgumentError, "#{uid} has no #{wire} to load"
            param -> param.name
          end
      end
    end
  end

  defp owners_of(%{estimator: estimator, evaluator: evaluator}) do
    owners_of(estimator, evaluator)
  end

  defp owners_of(estimator, evaluator) do
    [evaluator | stages_of(estimator)]
    |> Enum.filter(&is_map_key(&1, :uid))
    |> Map.new(&{&1.uid, &1})
  end

  defp stages_of(%Pipeline{stages: stages}), do: Enum.flat_map(stages, &stages_of/1)
  defp stages_of(operator), do: [operator]

  # An unfitted search: metadata, then the two operators it is made of.
  defp save_search(validator, kind, path, opts, extra) do
    opts = Keyword.validate!(opts, [:session, :sub_models, overwrite: false, options: %{}])
    session = opts[:session] || no_session!(validator)

    with :ok <- clear(session, path, opts[:overwrite]),
         :ok <-
           Layout.write_metadata(
             session,
             path,
             kind,
             validator.uid,
             Layout.search_params(validator, kind, wire_of(validator)),
             extra
           ),
         :ok <- save_part(session, validator.estimator, path, :estimator, opts) do
      save_part(session, validator.evaluator, path, :evaluator, opts)
    end
  end

  # A fitted one: everything above, plus the winner and optionally the fold models.
  defp save_searched(model, kind, path, opts, metrics) do
    opts = Keyword.validate!(opts, [:session, :sub_models, overwrite: false, options: %{}])
    session = opts[:session] || model.session
    keeping = keeping!(model, opts[:sub_models])
    extra = Map.put(metrics, "persistSubModels", keeping)

    # The session has to go down with the options: a validator is inert data with none of its
    # own, and the one this save uses came off the *model*. Without this, writing the metadata
    # asks the validator for a session it has never had.
    opts = Keyword.put(opts, :session, session)

    with :ok <- save_search(%{model.validator | uid: model.uid}, kind, path, opts, extra),
         :ok <- save_part(session, model.best_model, path, :best_model, opts) do
      if keeping, do: save_sub_models(session, model, path, opts), else: :ok
    end
  end

  # PySpark's default is "persist them if the search collected any", and asking for what was
  # never collected is an error there too — its message is the one worth copying, because the
  # fix is a refit rather than anything about the save.
  defp keeping!(model, nil), do: model.sub_models != nil
  defp keeping!(_model, false), do: false
  defp keeping!(%{sub_models: sub_models}, true) when sub_models != nil, do: true

  defp keeping!(_model, true) do
    raise ArgumentError,
          "sub_models: true, but this search kept none. Only a search fitted with " <>
            "collect_sub_models: true has them, and they cannot be recovered from a result — " <>
            "refit with collect_sub_models: true to save them."
  end

  # A cross-validation indexes by fold and then by param map; a split has one fold and no
  # `foldN` level at all. Both are PySpark's own layout.
  defp save_sub_models(session, %CrossValidatorModel{} = model, path, opts) do
    model.sub_models
    |> Enum.with_index()
    |> Enum.reduce_while(:ok, fn {fold, index}, :ok ->
      case save_fold(session, fold, path, index, opts) do
        :ok -> {:cont, :ok}
        {:error, error} -> {:halt, {:error, error}}
      end
    end)
  end

  defp save_sub_models(session, %TrainValidationSplitModel{} = model, path, opts) do
    save_fold(session, model.sub_models, path, nil, opts)
  end

  defp save_fold(session, models, path, fold, opts) do
    models
    |> Enum.with_index()
    |> Enum.reduce_while(:ok, fn {model, index}, :ok ->
      where = Layout.sub_model_path(path, fold, index)

      case save_stage(session, model, where, opts) do
        :ok -> {:cont, :ok}
        {:error, error} -> {:halt, {:error, error}}
      end
    end)
  end

  defp save_part(session, operator, path, part, opts) do
    save_stage(session, operator, Layout.part_path(path, part), opts)
  end

  # The metadata, then every stage under `stages/`, which is PySpark's own order — the metadata
  # is what names the stage uids, so a directory with no metadata has nothing to load from.
  defp save_stages(session, kind, uid, stages, path, opts) do
    with :ok <- clear(session, path, opts[:overwrite]),
         :ok <- Layout.write_metadata(session, path, kind, uid, stage_map(stages)) do
      stages
      |> Enum.with_index()
      |> Enum.reduce_while(:ok, fn {stage, index}, :ok ->
        where = Layout.stage_path(path, index, stage.uid, length(stages))

        case save_stage(session, stage, where, opts) do
          :ok -> {:cont, :ok}
          {:error, error} -> {:halt, {:error, error}}
        end
      end)
    end
  end

  # `overwrite:` is not passed down: the directory was cleared once above, and a stage that
  # asked for it again would be asking the server to delete a path this package just made.
  defp save_stage(_session, %Model{} = model, path, opts) do
    save(model, path, options: opts[:options])
  end

  defp save_stage(_session, %PipelineModel{} = model, path, opts) do
    save(model, path, options: opts[:options])
  end

  defp save_stage(session, stage, path, opts) do
    save(stage, path, session: session, options: opts[:options])
  end

  # `stageUids` is what every reader walks. `language` is PySpark's own marker: the 4.2 Connect
  # reader ignores it, and the classic JVM-backed one routes to `JavaMLReader` without it. Both
  # are `Latu.ML.Layout`'s compatibility claim — see its notes.
  defp stage_map(stages) do
    %{"stageUids" => Enum.map(stages, & &1.uid), "language" => "Python"}
  end

  defp clear(_session, _path, false), do: :ok

  defp clear(session, path, true) do
    # The one thing `MlCommand.Write`'s own `should_overwrite` cannot do: a pipeline writes a
    # directory rather than one operator, so the directory is what has to go.
    with {:ok, _cleared} <- helper(session, :handle_overwrite, [path, true]), do: :ok
  end

  defp no_session!(%Pipeline{}) do
    raise ArgumentError,
          "a pipeline has no session of its own until it is fitted. Pass session: to say " <>
            "which server to write it to."
  end

  defp no_session!(operator) do
    raise ArgumentError,
          "#{operator.class} has not been fitted, so it has no session of its own. Pass " <>
            "session: to say which server to write it to."
  end

  @doc """
  Read a model or an operator back from a path.

  An action, and the other half of `save/3`. What is at the path is **named rather than
  discovered**: Spark's metadata carries the class, but a `Read` has to say which class to
  load before the server will look, so the caller states what they expect and a mismatch is
  the server's refusal.

  Three spellings, one per way a caller knows what they saved:

    * an operator name from the registry — `:logistic_regression` for the estimator,
      `:logistic_regression_model` for the model it fits;
    * a generated model module — `Latu.ML.Classification.LogisticRegressionModel`;
    * `{class, kind}` for a JVM class this package does not know, where `kind` is
      `:estimator`, `:transformer`, `:evaluator` or `:model`.

  A **model** comes back holding a fresh cache reference: reading one costs a cache entry
  exactly as fitting one does, and it is yours to `delete/1`. It carries no training frame, so
  `summary/1` on it has nothing to rebuild from — and nothing to rebuild either, because a
  saved model does not carry its summary. Everything else comes back as inert data with
  nothing cached, carrying the uid and the params the metadata held.

  ## Examples

      {:ok, model} = ML.load(session, :logistic_regression_model, "/tmp/lr")
      scored = ML.transform(model, features)

      {:ok, lr} = ML.load(session, :logistic_regression, "/tmp/lr-unfitted")
  """
  @spec load(Latu.Session.t(), loadable(), String.t(), keyword()) ::
          {:ok, persistable()} | {:error, Error.t()}
  def load(session, name, path, opts \\ []) when is_binary(path) do
    case Internal.loadable!(name) do
      # A meta-algorithm has no `Read`: what is at the path is a directory this package wrote,
      # so this package is what reads it back. The guard matters — `loadable!({class, kind})`
      # answers with nil in the third slot too, so without it every operator stage lands here.
      {_class, kind, nil} when kind in @meta_kinds ->
        load_meta(session, kind, path, opts)

      resolved ->
        load_operator(session, resolved, path)
    end
  end

  defp load_operator(session, {class, kind, _row} = resolved, path) do
    with {:ok, execution} <-
           Latu.Client.execute_command(session, Internal.load_plan(class, kind, path)),
         {:ok, info} <- Result.operator_info(execution),
         :ok <- cached?(info, kind, class) do
      {:ok, Internal.loaded(session, resolved, info)}
    end
  end

  # A directory says what it holds, and there are two shapes of it now.
  defp load_meta(session, kind, path, opts) do
    case Layout.shape(kind) do
      :stages -> load_pipeline(session, kind, path)
      :search -> load_search(session, kind, path, opts)
    end
  end

  # A saved search, read back. The estimator and evaluator come first, because the grid can
  # only be decoded against the operators that own its params.
  defp load_search(session, kind, path, opts) do
    opts = Keyword.validate!(opts, sub_models: false)

    with {:ok, metadata} <- Layout.read_metadata(session, path),
         {:ok, estimator} <- load_stage(session, Layout.part_path(path, :estimator)),
         {:ok, evaluator} <- load_evaluator(session, path, estimator) do
      params = Map.get(metadata, "paramMap", %{})

      names = name_of(estimator, evaluator)
      validator = rebuilt_search(kind, metadata["uid"], estimator, evaluator, params, names)

      if Layout.fitted?(kind) do
        load_searched(session, kind, path, validator, metadata, opts)
      else
        {:ok, validator}
      end
    end
  end

  # The estimator is cached by the time the evaluator is asked for — a pipeline estimator can
  # carry a fitted stage — so a refusal here gives back what the estimator brought.
  defp load_evaluator(session, path, estimator) do
    case load_stage(session, Layout.part_path(path, :evaluator)) do
      {:ok, evaluator} ->
        {:ok, evaluator}

      {:error, error} ->
        release(collected([estimator]))
        {:error, error}
    end
  end

  defp load_searched(session, kind, path, validator, metadata, opts) do
    with {:ok, best_model} <- load_stage(session, Layout.part_path(path, :best_model)) do
      # The winner is already cached by the time the sub-models are asked for, so a refusal
      # here — `sub_models: true` against a directory saved without them, most likely — has to
      # give it back. A `Read` caches, so every error path below one owns what it read — the
      # others are `load_evaluator/3`, `load_stages/3`, `read_folds/4` and `read_fold/4`.
      case load_sub_models(session, kind, path, validator, metadata, opts) do
        {:ok, sub_models} ->
          {:ok, searched_from(kind, session, validator, metadata, best_model, sub_models)}

        {:error, error} ->
          release(collected([best_model]))
          {:error, error}
      end
    end
  end

  # **Not PySpark's default.** PySpark loads every persisted sub-model; a five-by-six search is
  # then thirty `Read` commands and thirty cache entries, on a load whose caller may only have
  # wanted the winner. Same reasoning as the fit, where a sub-model nobody asked for goes back
  # as soon as it is scored: they stay on disk until asked for, and the directory is still
  # there for a second call. `docs/deviations.md`.
  defp load_sub_models(_session, _kind, _path, _validator, _metadata, sub_models: false) do
    {:ok, nil}
  end

  defp load_sub_models(session, kind, path, validator, metadata, _opts) do
    if Map.get(metadata, "persistSubModels") do
      read_sub_models(session, kind, path, length(validator.param_maps), validator)
    else
      {:error,
       %Error{
         kind: :decode,
         message:
           "sub_models: true, but #{path} was saved without them — its metadata says " <>
             "persistSubModels is false. Only a search saved with sub_models: true has any."
       }}
    end
  end

  defp read_sub_models(session, :cross_validator_model, path, width, validator) do
    read_folds(session, path, Enum.to_list(0..(validator.num_folds - 1)), width)
  end

  defp read_sub_models(session, :train_validation_split_model, path, width, _validator) do
    with {:ok, models} <- read_fold(session, path, nil, width), do: {:ok, models}
  end

  defp read_folds(session, path, folds, width) do
    folds
    |> Enum.reduce_while({:ok, []}, fn fold, {:ok, done} ->
      case read_fold(session, path, fold, width) do
        {:ok, models} -> {:cont, {:ok, [models | done]}}
        {:error, error} -> {:halt, {:error, error, List.flatten(done)}}
      end
    end)
    |> case do
      {:ok, folds} ->
        {:ok, Enum.reverse(folds)}

      {:error, error, orphans} ->
        release(collected(orphans))
        {:error, error}
    end
  end

  # A `Read` caches, so a partial load orphans what it already read. Third place this shape
  # appears, after the pipeline fold and `load_stages`.
  defp read_fold(session, path, fold, width) do
    0..(width - 1)
    |> Enum.reduce_while({:ok, []}, fn index, {:ok, done} ->
      case load_stage(session, Layout.sub_model_path(path, fold, index)) do
        {:ok, model} -> {:cont, {:ok, [model | done]}}
        {:error, error} -> {:halt, {:error, error, done}}
      end
    end)
    |> case do
      {:ok, models} ->
        {:ok, Enum.reverse(models)}

      {:error, error, orphans} ->
        release(collected(orphans))
        {:error, error}
    end
  end

  defp rebuilt_search(kind, uid, estimator, evaluator, params, name_of) do
    grid = Layout.decode_grid(Map.get(params, "estimatorParamMaps", []), name_of)
    common = [estimator: estimator, param_maps: grid, evaluator: evaluator]

    built =
      case kind do
        model when model in [:cross_validator, :cross_validator_model] ->
          cross_validator(
            common ++
              [num_folds: Map.get(params, "numFolds", 3)] ++
              maybe(params, "seed", :seed) ++ fold_col(params)
          )

        _split ->
          train_validation_split(
            common ++
              [train_ratio: Map.get(params, "trainRatio", 0.75)] ++ maybe(params, "seed", :seed)
          )
      end

    %{built | uid: uid, collect_sub_models: Map.get(params, "collectSubModels", false)}
  end

  # An empty `foldCol` is PySpark's way of spelling "none", so it comes back as nil rather than
  # as a column named "".
  defp fold_col(params) do
    case Map.get(params, "foldCol") do
      empty when empty in [nil, ""] -> []
      column -> [fold_col: column]
    end
  end

  defp maybe(params, wire, name) do
    case Map.get(params, wire) do
      nil -> []
      value -> [{name, value}]
    end
  end

  defp searched_from(:cross_validator_model, session, validator, metadata, best_model, subs) do
    %CrossValidatorModel{
      uid: metadata["uid"],
      session: session,
      best_model: best_model,
      avg_metrics: Map.fetch!(metadata, "avgMetrics"),
      std_metrics: Map.get(metadata, "stdMetrics"),
      sub_models: subs,
      validator: validator
    }
  end

  defp searched_from(_split, session, validator, metadata, best_model, sub_models) do
    %TrainValidationSplitModel{
      uid: metadata["uid"],
      session: session,
      best_model: best_model,
      validation_metrics: Map.fetch!(metadata, "validationMetrics"),
      sub_models: sub_models,
      validator: validator
    }
  end

  defp load_pipeline(session, kind, path) do
    with {:ok, metadata} <- Layout.read_metadata(session, path),
         {:ok, uids} <- stage_uids(metadata, path),
         {:ok, stages} <- load_stages(session, path, uids) do
      {:ok, rebuilt(kind, metadata["uid"], session, stages)}
    end
  end

  defp stage_uids(%{"paramMap" => %{"stageUids" => uids}}, _path) when is_list(uids) do
    {:ok, uids}
  end

  defp stage_uids(_metadata, path) do
    {:error,
     %Error{
       kind: :decode,
       message: "the metadata at #{path} names no stageUids, so it is not a saved pipeline"
     }}
  end

  defp load_stages(session, path, uids) do
    uids
    |> Enum.with_index()
    |> Enum.reduce_while({:ok, []}, fn {uid, index}, {:ok, done} ->
      case load_stage(session, Layout.stage_path(path, index, uid, length(uids))) do
        {:ok, stage} -> {:cont, {:ok, [stage | done]}}
        {:error, error} -> {:halt, {:error, error, done}}
      end
    end)
    |> case do
      {:ok, stages} ->
        {:ok, Enum.reverse(stages)}

      # A `Read` caches, so the stages that did load are entries nothing asked for and nothing
      # else will give back. Same shape as a fit that fails partway, and released the same way.
      {:error, error, done} ->
        done |> Enum.flat_map(&cached_models/1) |> release()

        {:error, error}
    end
  end

  # A stage says what it is in its own metadata, which is the only thing that does: a directory
  # of stages is not typed. A stage the server wrote names a JVM class; a nested pipeline names
  # PySpark's, and is read the same way this one was.
  defp load_stage(session, path) do
    with {:ok, metadata} <- Layout.read_metadata(session, path) do
      class = metadata["class"]

      case Layout.kind(class) do
        nil -> load_operator_stage(session, class, path)
        kind -> load_meta(session, kind, path, [])
      end
    end
  end

  defp load_operator_stage(session, class, path) do
    case Registry.operator_of_class(class) do
      nil ->
        {:error,
         %Error{
           kind: :decode,
           message:
             "the stage at #{path} says it is a #{class}, which is not an operator this " <>
               "package knows. Latu.ML.operators/1 lists what there is."
         }}

      operator ->
        # The **row**, not `{class, kind}`. `loadable!({class, kind})` answers with a nil row,
        # and `Internal.loaded/3` reads the operator's param table off that row — so going
        # round through `load/4` here gave every stage inside a pipeline `known: []`, and
        # nothing noticed until a saved search needed to look a wire name up in one.
        load_operator(session, {class, operator.kind, operator}, path)
    end
  end

  defp rebuilt(:pipeline, uid, _session, stages), do: %Pipeline{uid: uid, stages: stages}

  defp rebuilt(:pipeline_model, uid, session, stages) do
    %PipelineModel{uid: uid, session: session, stages: stages}
  end

  @doc "See `load/4`. Raises instead of returning an error."
  @spec load!(Latu.Session.t(), loadable(), String.t(), keyword()) :: persistable()
  def load!(session, name, path, opts \\ []), do: bang!(load(session, name, path, opts))

  # Reading a model registers it, so the answer has to name where. Anything else is answered
  # by class name and caches nothing, which is why only the model arm is checked.
  defp cached?(%{ref: ref}, :model, _class) when is_binary(ref), do: :ok

  defp cached?(_info, :model, class) do
    {:error,
     %Error{
       kind: :protocol,
       message: "reading #{class} answered with no cache reference, where a loaded model belongs"
     }}
  end

  defp cached?(_info, _kind, _class), do: :ok

  @doc """
  Release a model, or a list of them, from the server's ML cache.

  An action, and the explicit half of the resource rule. The server answers with the references
  it dropped rather than staying silent, so a failure here is a real one.

  **Everything goes in one `Delete`**, whatever shape it arrives in. A list is what
  `attribute(model, "trees")` needs — the server registers every sub-model it hands back, so a
  hundred-tree forest is a hundred cache entries and giving them back one at a time is a hundred
  round trips. A composite is flattened: a `PipelineModel` owns the models inside it, nested
  pipelines included, and a search result owns its winner and the fold models where it was asked
  to keep them. And a **mixed list** of any of those is fine, because a script that fitted a
  pipeline and a search writes `delete([fitted, best])`.

  Inside a composite, a stage that holds nothing on the server is skipped rather than refused,
  so deleting a pipeline of transformers is `:ok` and sends nothing. **At the top level the
  contract is strict**: anything that is not one of the four is refused by name, because a
  silent `:ok` for an argument this cannot release is the worse answer.
  """
  @spec delete(deletable() | [deletable()]) :: :ok | {:error, Error.t()}
  def delete(model_or_models)

  def delete(%Model{} = model), do: delete([model])
  def delete(%PipelineModel{} = model), do: delete([model])
  def delete(%CrossValidatorModel{} = model), do: delete([model])
  def delete(%TrainValidationSplitModel{} = model), do: delete([model])

  def delete(models) when is_list(models) do
    Enum.each(models, &deletable!/1)

    case cached_models(models) do
      # Every element held nothing on the server — a pipeline of transformers, or a search
      # whose models have already been released.
      [] ->
        :ok

      cached ->
        session = hd(cached).session
        refs = Enum.map(cached, & &1.ref)

        with {:ok, _execution} <- Latu.Client.execute_command(session, Plan.delete(refs)) do
          :ok
        end
    end
  end

  def delete(other), do: raise(ArgumentError, not_deletable(other))

  defp deletable!(model)
       when is_struct(model, Model) or is_struct(model, PipelineModel) or
              is_struct(model, CrossValidatorModel) or
              is_struct(model, TrainValidationSplitModel),
       do: :ok

  defp deletable!(other), do: raise(ArgumentError, not_deletable(other))

  # A refusal names the fix, and an inspected fitted model is not a fix — it is several
  # kilobytes of params and a plan. So the argument is trimmed hard and the contract is spelled
  # out instead.
  defp not_deletable(other) do
    "#{inspect(other, limit: 2, printable_limit: 80)} is not something the ML cache holds. " <>
      "delete/1 takes a %Latu.ML.Model{}, a %Latu.ML.PipelineModel{}, a search result, or a " <>
      "list of those."
  end

  defp cached_models(%PipelineModel{stages: stages}), do: Enum.flat_map(stages, &cached_models/1)

  # An *unfitted* pipeline holds cache entries too: `Latu.ML.Pipeline.stage/0` admits a model
  # already fitted, so a pipeline loaded as a search's estimator can arrive owning one.
  defp cached_models(%Pipeline{stages: stages}), do: Enum.flat_map(stages, &cached_models/1)

  defp cached_models(%CrossValidatorModel{} = model) do
    cached_models(model.best_model) ++ Enum.flat_map(model.sub_models || [], &cached_models/1)
  end

  defp cached_models(%TrainValidationSplitModel{} = model) do
    cached_models(model.best_model) ++ Enum.flat_map(model.sub_models || [], &cached_models/1)
  end

  defp cached_models(models) when is_list(models), do: Enum.flat_map(models, &cached_models/1)
  defp cached_models(%Model{} = model), do: [model]
  defp cached_models(_stage), do: []

  @doc "See `delete/1`. Raises instead of returning an error."
  @spec delete!(deletable() | [deletable()]) :: :ok
  def delete!(model_or_models) do
    case delete(model_or_models) do
      :ok -> :ok
      {:error, error} -> raise error
    end
  end

  @doc """
  What the server thinks a model weighs, in bytes.

  An action, and a *server* estimate — `Model.estimatedSize`, which nothing client-side can
  compute. It is the number the cache's budgets are spent against, so it is what explains a
  `CONNECT_ML.MODEL_SIZE_OVERFLOW_EXCEPTION` on a fit that was refused, and what to measure
  before fitting a hundred of something.

      {:ok, bytes} = Latu.ML.model_size(model)
  """
  @spec model_size(Model.t()) :: {:ok, non_neg_integer()} | {:error, Error.t()}
  def model_size(%Model{} = model) do
    with {:ok, execution} <-
           Latu.Client.execute_command(model.session, Plan.get_model_size(model.ref)) do
      Result.param(execution)
    end
  end

  @doc "See `model_size/1`. Raises instead of returning an error."
  @spec model_size!(Model.t()) :: non_neg_integer()
  def model_size!(%Model{} = model), do: bang!(model_size(model))

  @doc """
  Everything the session's ML cache is holding: one entry per model and summary.

  An action. Each entry is `%{id: ..., class: ..., size: ...}` — the reference a
  `%Latu.ML.Model{}` carries, the JVM class, and the size in bytes the cache charged it. The
  server renders these as JSON strings and this decodes them, so a shape the server changes
  arrives as `{:error, %Latu.Error{kind: :protocol}}` rather than as a surprise.

  `size` is `0` for a summary, and `0` for a model too when the session has memory control
  turned off: `MLCache` only measures what it is going to budget.

      {:ok, held} = Latu.ML.cache_info(session)
      Enum.map(held, & &1.class)

  Takes a session rather than a model — the cache belongs to the session, and the point of
  asking is usually that you have lost track of what is in it.
  """
  @spec cache_info(Latu.Session.t()) :: {:ok, [cache_entry()]} | {:error, Error.t()}
  def cache_info(session) do
    with {:ok, execution} <- Latu.Client.execute_command(session, Plan.get_cache_info()),
         {:ok, entries} <- Result.param(execution) do
      decode_cache_info(entries)
    end
  end

  @doc "See `cache_info/1`. Raises instead of returning an error."
  @spec cache_info!(Latu.Session.t()) :: [cache_entry()]
  def cache_info!(session), do: bang!(cache_info(session))

  defp decode_cache_info(entries) when is_list(entries) do
    Enum.reduce_while(entries, {:ok, []}, fn entry, {:ok, done} ->
      case decode_entry(entry) do
        {:ok, decoded} -> {:cont, {:ok, [decoded | done]}}
        {:error, error} -> {:halt, {:error, error}}
      end
    end)
    |> case do
      {:ok, decoded} -> {:ok, Enum.reverse(decoded)}
      {:error, error} -> {:error, error}
    end
  end

  defp decode_cache_info(other) do
    {:error,
     %Error{
       kind: :protocol,
       message: "GetCacheInfo answered with #{inspect(other)}, where a list of strings belongs"
     }}
  end

  defp decode_entry(entry) when is_binary(entry) do
    case JSON.decode(entry) do
      {:ok, %{"id" => id, "class" => class, "size" => size}} ->
        {:ok, %{id: id, class: class, size: size}}

      _other ->
        {:error,
         %Error{
           kind: :protocol,
           message:
             "a GetCacheInfo entry is #{inspect(entry)}, where " <>
               ~s({"id": ..., "class": ..., "size": ...} belongs)
         }}
    end
  end

  defp decode_entry(other) do
    {:error,
     %Error{
       kind: :protocol,
       message: "a GetCacheInfo entry is #{inspect(other)}, where a JSON string belongs"
     }}
  end

  @doc """
  Drop everything the session's ML cache holds, and say how many objects that was.

  An action, and a blunt one: `MLCache.clear` empties the map and the offload directory
  together, so **every** model and summary in the session goes at once and every
  `%Latu.ML.Model{}` still in hand becomes a reference to nothing — the next fetch on one
  answers `CONNECT_ML.CACHE_INVALID`.

  `delete/1` and `with_model/3` are the ordinary way to give a model back. This is for a
  session you are reusing and want to start clean in, and for the end of a long-lived one.
  """
  @spec clean_cache(Latu.Session.t()) :: {:ok, non_neg_integer()} | {:error, Error.t()}
  def clean_cache(session) do
    with {:ok, execution} <- Latu.Client.execute_command(session, Plan.clean_cache()) do
      Result.param(execution)
    end
  end

  @doc "See `clean_cache/1`. Raises instead of returning an error."
  @spec clean_cache!(Latu.Session.t()) :: non_neg_integer()
  def clean_cache!(session), do: bang!(clean_cache(session))

  @doc """
  Fit, use, release — the bracket, in `File.open/3`'s shape.

  The model is deleted in an `after`, so it goes even if the function raises. A delete that
  itself fails is logged rather than raised: the useful result is the one in hand, and the
  session's end bounds the leak anyway.

  Returns whatever the function returned, or the fit's error if there was never a model.

      {:ok, coefficients} =
        Latu.ML.with_model(lr, features, fn model ->
          Latu.ML.attribute(model, :coefficients)
        end)
  """
  @spec with_model(
          Estimator.t() | Pipeline.t(),
          DataFrame.t(),
          (Model.t() | PipelineModel.t() -> result)
        ) :: result | {:error, Error.t()}
        when result: var
  def with_model(estimator, %DataFrame{} = data, fun)
      when is_function(fun, 1) and
             (is_struct(estimator, Estimator) or is_struct(estimator, Pipeline)) do
    with {:ok, model} <- fit(estimator, data) do
      try do
        fun.(model)
      after
        release(model)
      end
    end
  end

  @doc "See `with_model/3`. Raises where the fit would have returned an error."
  @spec with_model!(
          Estimator.t() | Pipeline.t(),
          DataFrame.t(),
          (Model.t() | PipelineModel.t() -> result)
        ) :: result
        when result: var
  def with_model!(estimator, %DataFrame{} = data, fun)
      when is_function(fun, 1) and
             (is_struct(estimator, Estimator) or is_struct(estimator, Pipeline)) do
    model = fit!(estimator, data)

    try do
      fun.(model)
    after
      release(model)
    end
  end

  # The five `CONNECT_ML` classes 4.2.0 raises, and what each one actually means. The **dotted**
  # spelling is what `error_class` carries — `dev/probe_ml.exs` printed it — where Spark's own
  # docs and this package's roadmap write them bare, so a matcher written from prose would
  # never fire. That is the whole reason this table exists rather than a note telling callers
  # to compare strings.
  @error_kinds %{
    "CONNECT_ML.CACHE_INVALID" => :cache_invalid,
    "CONNECT_ML.ATTRIBUTE_NOT_ALLOWED" => :attribute_not_allowed,
    @summary_lost => :summary_lost,
    "CONNECT_ML.MODEL_SIZE_OVERFLOW_EXCEPTION" => :model_too_large,
    "CONNECT_ML.ML_CACHE_SIZE_OVERFLOW_EXCEPTION" => :cache_full
  }

  @doc """
  Which ML failure this is, as an atom, or `nil` for anything that is not one.

  Match on this rather than on `error_class`, and never on the message. The class strings carry
  a `CONNECT_ML.` prefix that Spark's own documentation leaves off, so a matcher written from
  the docs compares against a string the server never sends — and the messages are worse: see
  `hint/1`.

      case Latu.ML.attribute(model, :coefficients) do
        {:ok, value} -> value
        {:error, error} ->
          case Latu.ML.error_kind(error) do
            :cache_invalid -> refit()
            _other -> raise error
          end
      end

  The five: `:cache_invalid` (the object is gone), `:attribute_not_allowed` (the name is not on
  this class's allowlist), `:summary_lost` (the cache dropped a summary), `:model_too_large`
  and `:cache_full` (a fit refused against a budget). A reference that never existed at all
  arrives as a bare `INTERNAL_ERROR` with no class, and answers `nil` here.
  """
  @spec error_kind(Error.t()) :: atom() | nil
  def error_kind(%Error{error_class: class}), do: Map.get(@error_kinds, class)
  def error_kind(_other), do: nil

  @doc """
  What to do about an ML failure, in this package's words, or `nil` if it has nothing to add.

  Spark's own message reaches the caller untouched — that is Latu's rule, and its text is
  usually the better one. These five are the exception, because their text is **wrong** in ways
  that send a reader the wrong way:

    * `CACHE_INVALID` describes the missing object as a *Summary object* even for a model,
      blames a 15-minute idle eviction even for a model deleted a moment earlier, and
      recommends `model.evaluate(dataset)`. All three measured false.
    * every one of them opens "Generic Spark Connect ML error".

  So this is a second opinion beside the server's, never a replacement for it.

      {:error, error} = Latu.ML.attribute(deleted, :coefficients)
      Latu.ML.hint(error)
  """
  @spec hint(Error.t()) :: String.t() | nil
  def hint(error), do: hint_for(error_kind(error))

  defp hint_for(:cache_invalid) do
    "That reference names nothing in the session's ML cache: it was deleted, or the session " <>
      "ended. Nothing brings it back — fit again. Spark's own text for this calls the object " <>
      "a Summary and blames an idle timeout; neither is reliable."
  end

  defp hint_for(:attribute_not_allowed) do
    "The server keeps a per-class allowlist of the names a Fetch may invoke, and this one is " <>
      "not on it for this class. Latu.ML.attributes/1 lists what is."
  end

  defp hint_for(:summary_lost) do
    "The cache offloaded the model to disk, and a model's save format does not carry its " <>
      "summary. Latu.ML.attribute/2 recovers from this on its own by sending CreateSummary, " <>
      "so seeing it here means either that the recovery failed — most likely the model is " <>
      "gone too — or that this model came from Latu.ML.load/3, which has no training frame " <>
      "to rebuild from and no summary on disk to have lost."
  end

  defp hint_for(:model_too_large) do
    "The fit was refused before anything was cached: the model is over " <>
      "spark.connect.session.connectML.mlCache.memoryControl.maxModelSize, 1 GB by default. " <>
      "That conf is read per fit, so raising it takes effect on the next one."
  end

  defp hint_for(:cache_full) do
    "The session's ML cache is over " <>
      "spark.connect.session.connectML.mlCache.memoryControl.maxStorageSize, 10 GB by " <>
      "default. Latu.ML.cache_info/1 says what is in it; delete/1 and clean_cache/1 empty it."
  end

  defp hint_for(_unknown), do: nil

  @doc """
  Every operator this package knows, as `Latu.ML.Operator` structs.

  The table the constructors are generated from, handed back so you can look at it. Filters are
  `and`-ed, and an unknown filter key raises rather than quietly matching nothing.

  ## Options

    * `:kind` — `:estimator`, `:transformer`, `:evaluator` or `:model`.
    * `:group` — PySpark's own module grouping: `:classification`, `:clustering`,
      `:evaluation`, `:feature`, `:fpm`, `:recommendation`, `:regression`.
    * `:status` — `:probed`, `:built` or `:missing`. See `t:Latu.ML.Operator.status/0`.

  ## Example

      iex> Latu.ML.operators(kind: :evaluator) |> Enum.map(& &1.name)
      [:binary_classification_evaluator, :clustering_evaluator,
       :multiclass_classification_evaluator, :multilabel_classification_evaluator,
       :ranking_evaluator, :regression_evaluator]
  """
  @spec operators(keyword()) :: [Operator.t()]
  def operators(filters \\ []) do
    filters = Keyword.validate!(filters, [:kind, :group, :status])

    Enum.filter(Registry.operators(), fn operator ->
      Enum.all?(filters, fn {key, value} -> Map.fetch!(operator, key) == value end)
    end)
  end

  @doc """
  Every method on the server's own helper object, as the registry holds them.

  The `ConnectHelper` route: eleven methods in the same allowlist a model's attributes are in,
  and attributes of nothing.

  Each is a map with `:name`, the `:method` that goes on the wire, how it `:returns`
  (`:value`, `:frame` or `:model`), its `:args` named and typed, and the `:python` call site
  all of that was read from — a helper has no class and so no signature, so the call site is
  the only thing that says what it takes.

  Ten of the eleven were read that way. `handleOverwrite` has no such call site — PySpark
  reaches it through `_call_java` — so its arguments are pinned in the extractor with the call
  site they were read off, and its row carries `read_by_hand: true` rather than passing for
  something derived. `save/3` is what uses it: an operator's own `overwrite:` rides
  `MlCommand.Write`, but a pipeline writes a **directory**, and clearing one is this method.

      iex> Latu.ML.helpers() |> Enum.frequencies_by(& &1.returns) |> Enum.sort()
      [frame: 5, model: 3, value: 3]
  """
  @spec helpers() :: [map()]
  def helpers, do: Registry.helpers()

  @doc """
  One operator by name, or `nil`.

      iex> Latu.ML.operator(:standard_scaler).class
      "org.apache.spark.ml.feature.StandardScaler"
  """
  @spec operator(atom()) :: Operator.t() | nil
  def operator(name) when is_atom(name) do
    case Registry.fetch(name) do
      {:ok, operator} -> operator
      :error -> nil
    end
  end

  @doc "See `operator/1`. Raises on a miss, naming the operators whose spelling is close."
  @spec operator!(atom()) :: Operator.t()
  def operator!(name) when is_atom(name), do: Registry.fetch!(name)

  @doc """
  An operator's param table, for a look in `iex`.

      iex> Latu.ML.params(:vector_assembler) |> Enum.map(& &1.name)
      [:handle_invalid, :input_cols, :output_col]
  """
  @spec params(atom()) :: [Param.t()]
  def params(name) when is_atom(name), do: Registry.fetch!(name).params

  @doc """
  What the server will let you `Fetch` from a class, and how each one answers.

  Takes a generated accessor module, a fitted `Latu.ML.Model`, a `Latu.ML.Summary`, or a JVM
  class name. The server's allowlist is inherited, so this is the union up the class's
  hierarchy rather than what its own entry says.

  Where the class is not known — an `LDA` fit, or a summary whose class the fitted model picks
  — this answers with what **every** class it could be allows, since only those are certain to
  be fetchable. Name the class directly to see its own, wider list.

      iex> Latu.ML.attributes(Latu.ML.Clustering.KMeansModel) |> Enum.map(& &1.name)
      [:cluster_center_matrix, :has_summary, :num_features, :predict, :summary, :to_string]
  """
  @spec attributes(module() | Model.t() | Summary.t() | String.t() | nil) :: [map()] | nil
  def attributes(holder)

  # A holder whose class is decided elsewhere carries every class it could be, and what the
  # server will *certainly* let you fetch is what they all allow — the intersection, not the
  # union. `dev/probe_ml.exs` measured the difference: `LocalLDAModel` and `DistributedLDAModel`
  # are siblings rather than a chain, so four of the union's names refuse on a `LocalLDAModel`
  # with `CONNECT_ML.ATTRIBUTE_NOT_ALLOWED`. Name the class you know you have to see its own
  # list — `Latu.ML.attributes(Latu.ML.Clustering.DistributedLDAModel)`.
  def attributes(%{class: nil, candidates: [first | rest]}) do
    Enum.reduce(rest, attributes(first) || [], fn candidate, kept ->
      allowed = MapSet.new(attributes(candidate) || [], & &1.wire)

      Enum.filter(kept, &MapSet.member?(allowed, &1.wire))
    end)
  end

  def attributes(%Model{class: class}), do: attributes(class)
  def attributes(%Summary{class: class}), do: attributes(class)

  def attributes(class) do
    case Registry.class(class) do
      nil -> nil
      row -> row.attributes
    end
  end

  defp release(model) do
    case delete(model) do
      :ok -> :ok
      {:error, error} -> Logger.warning(released(model) <> Exception.message(error))
    end
  end

  defp released(%PipelineModel{uid: uid}), do: "could not delete the models in #{uid}: "
  defp released(%Model{ref: ref}), do: "could not delete model #{ref}: "

  defp released(models) when is_list(models),
    do: "could not delete the #{length(models)} model(s) a failed fit had already cached: "

  defp bang!({:ok, value}), do: value
  defp bang!({:error, error}), do: raise(error)
end
