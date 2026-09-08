defmodule Latu.ML.Internal do
  @moduledoc false
  # The mapping from this package's structs to the protos, in one place. It lives here rather
  # than on `Latu.ML` so the golden tests can build exactly what the facade sends without
  # sending it — a `Fit` is a command, so there is no lazy form to inspect — and rather than on
  # `Latu.ML.Plan`, which takes protocol values and knows nothing about estimators.

  alias Latu.Column
  alias Latu.DataFrame
  alias Latu.ML.CrossValidator
  alias Latu.ML.Estimator
  alias Latu.ML.Evaluator
  alias Latu.ML.Helper
  alias Latu.ML.Model
  alias Latu.ML.Operator
  alias Latu.ML.Param
  alias Latu.ML.Pipeline
  alias Latu.ML.Plan
  alias Latu.ML.Registry
  alias Latu.ML.Summary
  alias Latu.ML.TrainValidationSplit
  alias Latu.ML.Transformer

  @kinds [:estimator, :transformer, :evaluator, :model]

  @doc """
  Whether a bigger metric is a better one, for this evaluator. `Latu.ML.larger_better?/1`.

  Pure, and here rather than on the facade because there is nothing to send: the answer is a
  registry row resolved against the evaluator's own `metric_name`, and `dev/check_offline.exs`
  can only reach it here.
  """
  def larger_better?(%Evaluator{} = evaluator) do
    operator = Registry.operator_of_class(evaluator.class)

    case operator && operator.larger_better do
      true ->
        true

      {:only, metrics} ->
        metric_name(evaluator, operator) in metrics

      {:except, metrics} ->
        metric_name(evaluator, operator) not in metrics

      nil ->
        raise ArgumentError,
              "there is no isLargerBetter for #{evaluator.class}: it is not one of the " <>
                "evaluators this registry knows. Latu.ML.operators(:evaluator) lists them."
    end
  end

  # Set, or the evaluator's own default. A set param is `{wire, type, value}` and is found by
  # its wire name, which comes off the same registry row as the default — so neither spelling
  # is written out here. Only reached for the two shapes that name metrics, and every evaluator
  # with one of those has a `metric_name`.
  defp metric_name(%Evaluator{params: set}, operator) do
    row = Enum.find(operator.params, &(&1.name == :metric_name))

    case List.keyfind(set, row.wire, 0) do
      {_wire, _type, value} -> to_string(value)
      nil -> row.default
    end
  end

  @typedoc """
  One combination from a grid: which operator's param, by uid, and what to set it to.

  A uid rather than a struct because the operator being tuned is often a stage of a pipeline,
  and because this is how Spark's saved format keys them (`parent` and `name`).
  """
  @type param_map :: [{String.t(), atom(), term()}]

  @doc """
  The Cartesian product of a grid, as param maps. `Latu.ML.param_grid/1`.

  Pure and here rather than on the facade for the same reason `larger_better?/1` is.
  """
  def param_grid(grids) when is_list(grids) do
    pairs =
      for {operator, grid} <- grids,
          {name, values} <- grid do
        {operator, name, checked_values!(operator, name, values)}
      end

    # `itertools.product` over the keys in the order they were given, so the **last** param
    # varies fastest. Order is not cosmetic: the best index is the first maximum, so two param
    # maps that score identically are separated by where they sit in this list.
    Enum.reduce(pairs, [[]], fn {operator, name, values}, maps ->
      for map <- maps, value <- values, do: map ++ [{uid_of!(operator), name, value}]
    end)
  end

  defp checked_values!(operator, name, values) when is_list(values) do
    known = Map.new(operator.known, &{&1.name, &1})

    if not Map.has_key?(known, name), do: unknown!(name, operator.known)
    if values == [], do: raise(ArgumentError, empty_grid(name))

    values
  end

  defp checked_values!(_operator, name, values) do
    raise ArgumentError,
          "the grid for #{inspect(name)} is #{inspect(values)}, and a grid is a list of " <>
            "values to try: param_grid(lr, #{name}: [#{inspect(values)}])"
  end

  defp empty_grid(name) do
    "the grid for #{inspect(name)} is empty, so there is nothing to try. Leave the param out " <>
      "to use the estimator's own value."
  end

  # A grid entry says which operator's param it sets, because a pipeline's stages each have
  # their own and PySpark's saved format keys on exactly this. A helper has no uid and cannot
  # be tuned; nothing else is missing one.
  defp uid_of!(%{uid: uid}) when is_binary(uid), do: uid

  defp uid_of!(other) do
    raise ArgumentError,
          "#{inspect(other.__struct__)} has no uid, so its params cannot be tuned: a grid " <>
            "entry has to say which operator it belongs to."
  end

  @doc """
  The `{train, validation}` pairs a validator searches over.

  Pure: every pair is a plan and nothing has run, so `test/latu/ml_test.exs` can read the cut
  off the relations with no session at all. It needs a real `Latu` for the column expressions,
  so `dev/check_offline.exs` is not where it is checked — and the cut is the part of a search
  that goes wrong quietly, so it is checked twice: once on the plans and once on the rows, by
  `dev/probe_ml.exs` against PySpark's own fold.

  The cut is Spark's own and is **not** a `random_split`: a `rand(seed)` column is added once,
  and each fold is a range of it. Adding it once is what makes the two filters agree — a bare
  `rand()` inside a predicate is redrawn per evaluation, so train and validation would overlap.
  """
  def folds(%DataFrame{} = frame, %CrossValidator{fold_col: nil} = validator) do
    drawn = with_draw(frame, validator.uid, validator.seed)
    draw = draw_column(validator.uid)
    height = 1.0 / validator.num_folds

    for index <- 0..(validator.num_folds - 1) do
      held =
        Column.all([
          Column.greater_equal(draw, index * height),
          Column.less(draw, (index + 1) * height)
        ])

      {Latu.filter(drawn, Column.not_(held)), Latu.filter(drawn, held)}
    end
  end

  # A column that already holds a fold number. PySpark also asks the server to raise on a
  # number outside `0..num_folds-1` and to refuse an empty fold, which is three actions per
  # fold before any fitting happens; Latu lets the fit fail on its own rows instead. See
  # `docs/deviations.md`.
  def folds(%DataFrame{} = frame, %CrossValidator{fold_col: fold_col} = validator) do
    for index <- 0..(validator.num_folds - 1) do
      {Latu.filter(frame, Column.not_equal(fold_col, index)),
       Latu.filter(frame, Column.equal(fold_col, index))}
    end
  end

  def folds(%DataFrame{} = frame, %TrainValidationSplit{} = validator) do
    drawn = with_draw(frame, validator.uid, validator.seed)
    held = Column.greater_equal(draw_column(validator.uid), validator.train_ratio)

    [{Latu.filter(drawn, Column.not_(held)), Latu.filter(drawn, held)}]
  end

  # PySpark names it `<uid>_rand`, and the name matters: it is what both filters read. The atom
  # is derived from a uid this package generated, so the set is bounded by the validators
  # fitted rather than by anything a caller can spell.
  defp draw_column(uid), do: String.to_atom(uid <> "_rand")

  defp with_draw(%DataFrame{} = frame, uid, seed) do
    draw = if seed, do: Latu.Functions.rand(seed), else: Latu.Functions.rand()

    Latu.select(frame, [Column.star(), {draw_column(uid), draw}])
  end

  @doc """
  One combination from a grid, applied to the thing being searched.

  A param map says which operator each setting belongs to, by uid, so applying one to a
  pipeline reaches the stage that owns it and leaves the rest alone. That is the whole reason
  a grid entry carries a uid: tuning a classifier at the end of a feature pipeline is the
  ordinary case, not the exotic one.

  Every uid in the map has to belong to something here — see `covered!/1`, which says so once
  before a search starts rather than letting a grid built for the wrong estimator run to
  completion and report a winner that was never varied.
  """
  def apply_params(%Pipeline{} = pipeline, param_map) do
    %{pipeline | stages: Enum.map(pipeline.stages, &apply_params(&1, param_map))}
  end

  def apply_params(%{uid: uid, known: known, params: params} = operator, param_map) do
    case for {^uid, name, value} <- param_map, do: {name, value} do
      [] -> operator
      mine -> %{operator | params: overridden(params, params!(mine, known))}
    end
  end

  # Nothing in this map is for this operator, and a helper has no uid to match on anyway.
  def apply_params(operator, _param_map), do: operator

  # Replace by wire name and append the rest, so setting a param the constructor already set
  # replaces it rather than sending it twice.
  defp overridden(existing, overrides) do
    wires = MapSet.new(overrides, fn {wire, _type, _value} -> wire end)

    Enum.reject(existing, fn {wire, _type, _value} -> wire in wires end) ++ overrides
  end

  @doc """
  Refuse a grid whose params belong to something the search will never touch.

  PySpark makes this check when a validator is **saved**, which is late: a search runs to the
  end first and names a best model that every param map scored identically, because none of
  them changed anything. Checked here before the first fit instead.
  """
  def covered!(validator) do
    theirs = MapSet.new(uids_of(validator.estimator))

    stray =
      for param_map <- validator.param_maps,
          {uid, name, _value} <- param_map,
          not MapSet.member?(theirs, uid),
          do: {uid, name}

    case Enum.uniq(stray) do
      [] ->
        :ok

      [{uid, name} | _rest] ->
        raise ArgumentError,
              "the grid sets #{inspect(name)} on #{uid}, which is not the estimator being " <>
                "searched or any stage of it. A grid has to be built from the same operators."
    end
  end

  defp uids_of(%Pipeline{stages: stages}), do: Enum.flat_map(stages, &uids_of/1)
  defp uids_of(%{uid: uid}) when is_binary(uid), do: [uid]
  defp uids_of(_operator), do: []

  @meta_algorithms [
    :pipeline,
    :pipeline_model,
    :cross_validator,
    :cross_validator_model,
    :train_validation_split,
    :train_validation_split_model
  ]

  @type saveable :: Model.t() | Estimator.t() | Transformer.t() | Evaluator.t()
  @type loadable :: atom() | {String.t(), Plan.kind()}

  @doc """
  The names `Latu.ML.load/4` takes that name no operator: the pipelines and the searches.

  Every one of them is a directory this package laid out rather than something the server
  reads, which is what `Latu.ML`'s dispatch guards on — a `{class, kind}` pair answers with a
  nil row too, so nil alone cannot be the test.
  """
  @spec meta_algorithms() :: [atom()]
  def meta_algorithms, do: @meta_algorithms

  @doc """
  Build the operator a generated constructor stands for.

  Every constructor in `Latu.ML.Classification` and its siblings is one call to this: the
  registry holds the class, the kind, the model class and the param table, so the generated
  function is a name, a doc and a `@spec` and nothing else. Which struct comes back is the
  operator's kind, and that is the whole difference between the three.
  """
  @spec build(atom(), keyword()) :: Estimator.t() | Transformer.t() | Evaluator.t()
  def build(name, opts) when is_atom(name) and is_list(opts) do
    operator = Registry.fetch!(name)

    fields = %{
      class: operator.class,
      uid: uid(operator.class),
      known: operator.params,
      params: params!(opts, operator.params)
    }

    case operator.kind do
      :estimator ->
        struct!(
          Estimator,
          Map.merge(fields, %{
            model_class: operator.model_class,
            model_candidates: operator.model_classes
          })
        )

      :transformer ->
        struct!(Transformer, fields)

      :evaluator ->
        struct!(Evaluator, fields)

      # No uid: a helper call sends its params as positional arguments and no `MlOperator` at
      # all, so there is nothing on the wire for one to ride on.
      :helper ->
        struct!(Helper, Map.delete(fields, :uid))
    end
  end

  @doc """
  The summary a fitted model carries, composed rather than fetched.

  Spark has no separate cache key for a summary: a `Fetch` names the model and puts `summary`
  at the head of its method chain. So there is nothing to send, and this is a builder rather
  than an action.

  The frame it keeps for `CreateSummary` recovery is per model class, and the registry says
  which: seven of the ten want the raw training frame, and the three clustering models want the
  model's own transform of it. Getting that wrong would only ever show up under eviction.
  """
  @spec summary(Model.t()) :: Summary.t()
  def summary(model)

  def summary(%Model{} = model) do
    row = model.class && Registry.class(model.class)

    case row && row.summary_dataset do
      nil ->
        raise ArgumentError, no_summary(model)

      kind ->
        %Summary{
          ref: model.ref,
          session: model.session,
          class: summary_class(row.summary_classes),
          candidates: row.summary_classes,
          dataset: summary_dataset(kind, model)
        }
    end
  end

  @doc "What a generated `summary/1` calls: the class check, then `summary/1`."
  @spec summary(Model.t(), String.t()) :: Summary.t()
  def summary(%Model{} = model, class) do
    ensure_class!(model, class, "summary")

    summary(model)
  end

  defp summary_class([class]), do: class

  # `LogisticRegressionModel` and `RandomForestClassificationModel` answer with the binary
  # summary or the multiclass one depending on what the *fitted* model found. Nothing
  # client-side knows that, so nothing client-side claims it — the candidates are kept instead.
  defp summary_class(_candidates), do: nil

  # A model that came from `Latu.ML.load/3` has no training frame — and no summary either, since
  # a model's save format does not carry one. Nothing to rebuild from, and nothing to rebuild.
  defp summary_dataset(_kind, %Model{dataset: nil}), do: nil
  defp summary_dataset(:training_frame, model), do: model.dataset
  defp summary_dataset(:transformed, model), do: transform_relation(model, model.dataset)

  defp no_summary(%Model{class: nil}) do
    "this model has no class, so nothing here knows whether it has a summary. " <>
      "Latu.ML.Estimator.new/2 leaves it unset; pass model_class: to set it."
  end

  defp no_summary(%Model{class: class}) do
    with_summaries =
      Registry.classes()
      |> Enum.filter(& &1.summary_dataset)
      |> Enum.map_join(", ", & &1.python)

    "#{class} has no training summary. The ones that do: #{with_summaries}."
  end

  @doc """
  Read an attribute through a generated accessor, having checked the model is the right class.

  The server checks the *name* against the class's allowlist and refuses with
  `CONNECT_ML.ATTRIBUTE_NOT_ALLOWED`. This checks the *class*, one layer earlier, so calling
  `LogisticRegressionModel.coefficients/1` on a `KMeansModel` names the module that would have
  taken it rather than costing a round trip to be told a name is not allowed.
  """
  @spec attribute(Model.t() | Summary.t(), String.t(), String.t(), [term()]) ::
          {:ok, term()} | {:error, Latu.Error.t()}
  def attribute(holder, class, wire, args \\ []) do
    ensure_class!(holder, class, wire)

    Latu.ML.attribute(holder, wire, args)
  end

  @doc "See `attribute/4` above. The frame half, so a lazy builder rather than an action."
  @spec attribute_frame(Model.t() | Summary.t(), String.t(), String.t(), [term()]) ::
          Latu.DataFrame.t()
  def attribute_frame(holder, class, wire, args \\ []) do
    ensure_class!(holder, class, wire)

    Latu.ML.attribute_frame(holder, wire, args)
  end

  # A model built by `Latu.ML.Estimator.new/2` without a `model_class` has nothing to check
  # against; the server's own refusal is then the only one, which is what that escape hatch
  # promises.
  defp ensure_class!(%{class: nil}, _class, _wire), do: :ok
  defp ensure_class!(%{class: class}, class, _wire), do: :ok

  defp ensure_class!(%{class: actual}, expected, wire) do
    raise ArgumentError,
          "#{wire} is an attribute of #{expected}, and this model is a #{actual}. " <>
            home(actual)
  end

  defp home(class) do
    case Registry.class(class) do
      nil -> "Latu.ML.attribute/2 takes any class."
      row -> "Try #{inspect(row.module)}."
    end
  end

  @doc """
  Check a caller's keyword list against an operator's param table, and hand back the entries
  `Latu.ML.Plan.params/1` wants.

  Names only. A value's kind is checked when it becomes a literal, and its range is Spark's
  `ParamValidators` to refuse — with a better message than this package could write.
  """
  @spec params!(keyword(), [Param.t()]) :: [Latu.ML.Plan.param()]
  def params!(opts, known) when is_list(opts) and is_list(known) do
    table = Map.new(known, &{&1.name, &1})

    Enum.map(opts, fn {name, value} ->
      param = Map.get(table, name) || unknown!(name, known)

      {param.wire, param.type, value}
    end)
  end

  defp unknown!(name, known) do
    raise ArgumentError,
          "unknown param #{inspect(name)}, expected one of " <>
            inspect(Enum.map(known, & &1.name))
  end

  @doc """
  A uid in PySpark's shape: the class's short name, an underscore, twelve hex digits.

  The server does not assign one — a `Fit` answers with an `obj_ref` and nothing else — so the
  uid a model carries is whatever the client gave its estimator. PySpark copies it the same way.
  """
  @spec uid(String.t()) :: String.t()
  def uid(class) when is_binary(class) do
    short = class |> String.split(".") |> List.last()

    short <> "_" <> Base.encode16(:crypto.strong_rand_bytes(6), case: :lower)
  end

  @doc "The `Fit` plan `Latu.ML.fit/2` sends."
  @spec fit_plan(Estimator.t(), Plan.relation()) :: Plan.command()
  def fit_plan(%Estimator{} = estimator, data) do
    Plan.fit(
      Plan.operator(estimator.class, estimator.uid, :estimator),
      Plan.params(estimator.params),
      data
    )
  end

  @doc "The `Evaluate` plan `Latu.ML.evaluate/2` sends."
  @spec evaluate_plan(Evaluator.t(), Plan.relation()) :: Plan.command()
  def evaluate_plan(%Evaluator{} = evaluator, data) do
    Plan.evaluate(
      Plan.operator(evaluator.class, evaluator.uid, :evaluator),
      Plan.params(evaluator.params),
      data
    )
  end

  @doc """
  The `Write` plan `Latu.ML.save/3` sends.

  A model is written by reference, because the server already holds it; everything else is
  written by class and uid, because the server holds nothing and builds the instance from this
  message. Either way the params ride along — for a model they are laid over a copy of what
  the cache holds, which is how PySpark makes a saved model carry the values its estimator was
  given.
  """
  @spec save_plan(saveable(), String.t(), boolean(), Plan.options()) :: Plan.command()
  def save_plan(target, path, overwrite, options)

  def save_plan(%Model{} = model, path, overwrite, options) do
    Plan.write(model.ref, Plan.params(model.params), path, overwrite, options)
  end

  def save_plan(operator, path, overwrite, options) when is_struct(operator) do
    Plan.write(
      Plan.operator(operator.class, operator.uid, kind!(operator)),
      Plan.params(operator.params),
      path,
      overwrite,
      options
    )
  end

  @doc """
  The `Read` plan `Latu.ML.load/3` sends.

  The uid is empty on purpose: a `Read` names a class rather than an instance, and the uid is
  in the saved metadata — it comes back in the answer.
  """
  @spec load_plan(String.t(), Plan.kind(), String.t()) :: Plan.command()
  def load_plan(class, kind, path) do
    Plan.read(Plan.operator(class, "", kind), path)
  end

  @doc """
  What the caller named, as a class, a kind, and the registry row behind it if there is one.

  Four spellings, one per way a caller knows what they saved: the registry's own name for an
  operator or a model class (`:logistic_regression`, `:logistic_regression_model`), the
  generated accessor module for a model class, a `{class, kind}` pair for anything on the
  server's classpath this package never heard of, or `:pipeline` / `:pipeline_model` — which
  are neither operators nor classes, because a pipeline has no server-side anything.
  """
  @spec loadable!(loadable()) :: {String.t(), Plan.kind() | atom(), Operator.t() | nil}
  def loadable!(name)

  def loadable!({class, kind}) when is_binary(class) and kind in @kinds, do: {class, kind, nil}

  def loadable!(name) when name in @meta_algorithms do
    {Latu.ML.Layout.class(name), name, nil}
  end

  def loadable!(name) when is_atom(name) and not is_nil(name) do
    case Registry.fetch(name) do
      {:ok, operator} -> {operator.class, operator.kind, operator}
      :error -> from_module!(name)
    end
  end

  def loadable!(other) do
    raise ArgumentError,
          "#{inspect(other)} names nothing to load. Pass an operator name from " <>
            "Latu.ML.operators/1, a generated model module, :pipeline or :pipeline_model, " <>
            "or {class, kind} for a class this package does not know."
  end

  # An accessor module is an atom too, so this is where an atom lands that is not an operator
  # name. A summary module is the near miss worth naming: it looks exactly like a model module
  # and there is no such thing as a saved summary.
  defp from_module!(module) do
    case Registry.class(module) do
      %{kind: :model, class: class} ->
        {class, :model, nil}

      %{kind: :summary, python: python} ->
        raise ArgumentError,
              "#{python} is a training summary, and a summary is not saved or loaded — it " <>
                "belongs to the model that made it, and a saved model does not carry one."

      nil ->
        raise ArgumentError,
              "no operator or model class #{inspect(module)}. " <>
                "Latu.ML.operators/1 lists what there is."
    end
  end

  @doc """
  The struct a `Read` answer becomes.

  A model carries the cache reference the read registered, and no training frame: a saved
  model does not carry its summary, so there is nothing for one to be rebuilt from. Everything
  else carries no reference at all — the server answered with data and cached nothing.
  """
  @spec loaded(Latu.Session.t(), {String.t(), Plan.kind(), Operator.t() | nil}, map()) ::
          Model.t() | Estimator.t() | Transformer.t() | Evaluator.t()
  def loaded(session, resolved, info)

  def loaded(session, {class, :model, _row}, info) do
    %Model{
      ref: info.ref,
      session: session,
      class: class,
      uid: info.uid,
      dataset: nil,
      params: info.params
    }
  end

  def loaded(_session, {class, :estimator, row}, info) do
    %Estimator{
      class: class,
      uid: info.uid || uid(class),
      params: info.params,
      known: known(row),
      model_class: row && row.model_class,
      model_candidates: (row && row.model_classes) || []
    }
  end

  def loaded(_session, {class, :transformer, row}, info) do
    %Transformer{
      class: class,
      uid: info.uid || uid(class),
      params: info.params,
      known: known(row)
    }
  end

  def loaded(_session, {class, :evaluator, row}, info) do
    %Evaluator{
      class: class,
      uid: info.uid || uid(class),
      params: info.params,
      known: known(row)
    }
  end

  defp known(nil), do: []
  defp known(row), do: row.params

  defp kind!(%Estimator{}), do: :estimator
  defp kind!(%Transformer{}), do: :transformer
  defp kind!(%Evaluator{}), do: :evaluator

  defp kind!(other) do
    raise ArgumentError,
          "#{inspect(other)} is not something this package can save. A model, an estimator, " <>
            "a transformer or an evaluator is."
  end

  @doc """
  The relation `Latu.ML.transform/2` builds.

  A model transforms by reference and resends the params it was fitted with, which is what
  PySpark does: `try_remote_fit` copies the estimator's values onto the model, and
  `try_remote_transform_relation` serialises the model's own map.
  """
  @spec transform_relation(Transformer.t() | Model.t(), Plan.relation()) :: Plan.relation()
  def transform_relation(%Transformer{} = transformer, data) do
    Plan.transform(
      Plan.operator(transformer.class, transformer.uid, :transformer),
      data,
      Plan.params(transformer.params)
    )
  end

  def transform_relation(%Model{} = model, data) do
    Plan.transform(model.ref, data, Plan.params(model.params))
  end

  @doc """
  The `Fetch` plan `Latu.ML.attribute/2` sends.

  A summary's chain has `summary` in front of the attribute, and the object it names is the
  model — Spark reaches a summary through the model that owns it rather than by a key of its
  own.
  """
  @spec fetch_plan(Model.t() | Summary.t(), atom() | String.t(), [term()]) :: Plan.command()
  def fetch_plan(holder, name, args \\ [])

  def fetch_plan(%Model{} = model, name, args) do
    Plan.fetch_command(model.ref, [Plan.method(name, args)])
  end

  def fetch_plan(%Summary{} = summary, name, args) do
    Plan.fetch_command(summary.ref, methods(summary, name, args))
  end

  @doc """
  The relation `Latu.ML.attribute_frame/2` builds.

  The same `Fetch` as `fetch_plan/2`, wrapped as a relation instead of a command — that is the
  whole difference between an attribute that answers with a literal and one that answers with a
  DataFrame. `model_summary_dataset` stays nil: it is there to rebuild an evicted *summary*
  from its training frame, and a model is never evicted (`docs/decisions.md`).
  """
  @spec fetch_relation(Model.t() | Summary.t(), atom() | String.t(), [term()]) :: Plan.relation()
  def fetch_relation(holder, name, args \\ [])

  def fetch_relation(%Model{} = model, name, args) do
    Plan.fetch_relation(model.ref, [Plan.method(name, args)])
  end

  # A summary's frame attribute is the one place `model_summary_dataset` is not nil: the server
  # can rebuild a dropped summary from it while it plans, so this arm needs no retry of its own.
  def fetch_relation(%Summary{} = summary, name, args) do
    Plan.fetch_relation(summary.ref, methods(summary, name, args), summary.dataset)
  end

  @doc "The `CreateSummary` plan that recovers a summary the cache has dropped."
  @spec create_summary_plan(Summary.t()) :: Plan.command()
  def create_summary_plan(%Summary{} = summary) do
    Plan.create_summary(summary.ref, summary.dataset)
  end

  # Only the last link takes arguments — the leading ones are the path to the object.
  defp methods(%Summary{methods: leading}, name, args) do
    Enum.map(leading, &Plan.method/1) ++ [Plan.method(name, args)]
  end

  @doc """
  The wire name for whatever the caller wrote.

  `Latu.ML.attributes/1` lists snake_case names and the server's allowlist is camelCase, so both
  have to reach a `Fetch` or the two halves of one surface disagree — `:area_under_roc` is what
  the table advertises and `"areaUnderROC"` is what `MLUtils` will answer to.

  The order matters. A name that is already on the class's allowlist is taken as written, so a
  wire name never means anything else; only then is it looked for among the rows' snake names.
  And a name the rows do not know **passes through unchanged**, because a class this package has
  no registry for has to stay reachable and the server's `ATTRIBUTE_NOT_ALLOWED` is the better
  error than a guess (`docs/deviations.md`).
  """
  @spec wire_name(Model.t() | Summary.t(), atom() | String.t()) :: String.t()
  def wire_name(holder, name) do
    text = to_string(name)

    case attribute_row(holder, text) do
      nil -> snake_wire(holder, text)
      _allowed -> text
    end
  end

  # Mirrors `attribute_row/2`'s handling of a holder whose class is not decided yet: every class
  # it could be is asked, and the first that names the attribute answers.
  defp snake_wire(%{class: nil, candidates: [_ | _] = candidates}, text) do
    Enum.find_value(candidates, text, &wire_for_snake(&1, text))
  end

  defp snake_wire(%{class: nil}, text), do: text
  defp snake_wire(%{class: class}, text), do: wire_for_snake(class, text) || text

  defp wire_for_snake(class, text) do
    case Registry.class(class) do
      nil ->
        nil

      row ->
        Enum.find_value(row.attributes, fn attribute ->
          if to_string(attribute.name) == text, do: attribute.wire
        end)
    end
  end

  @doc """
  How an attribute answers, per the registry: `:value`, `:frame`, `:summary` or `:operators`.

  `nil` where the class is unknown or the name is not on its allowlist — the server is then the
  only judge, which is what an unnamed class buys you.
  """
  @spec answer_shape(Model.t() | Summary.t(), String.t()) :: atom() | nil
  def answer_shape(holder, wire) do
    case attribute_row(holder, wire) do
      nil -> nil
      row -> row.returns
    end
  end

  @doc """
  Turn the ids an `:operators` answer carried into models, one per sub-model.

  The class is the registry's `:element_class`, read off PySpark's `List[X]` annotation. It is
  not guessable: a **GBT classifier's** trees are `DecisionTreeRegressionModel`s, because
  gradient boosting fits regression trees to residuals whatever it is classifying, while a
  random forest classifier's are `DecisionTreeClassificationModel`s.

  Each one is a cache entry of its own — `MLHandler` registers every sub-model on the way out —
  so these come back as models the caller owns and must give back, exactly like a
  `Latu.ML.fit/2` does.
  """
  @spec operators(Model.t() | Summary.t(), String.t(), struct()) ::
          {:ok, [Model.t()]} | {:error, Latu.Error.t()}
  def operators(holder, wire, execution) do
    with {:ok, refs} <- Latu.ML.Result.operators(execution) do
      class = attribute_row(holder, wire).element_class

      {:ok, Enum.map(refs, &%Model{ref: &1, session: holder.session, class: class})}
    end
  end

  @doc """
  Turn a caller's argument list into the literals and relations a `Fetch` method carries.

  The types are not guessed and not asked of the caller: `priv/ml_attributes.exs` records them
  from PySpark's own annotations, so `predict` knows it wants a `Vector`, `recommendForAllUsers`
  an `int` and `evaluate` a whole frame. That is also why this needs the holder's class — a
  model built by `Latu.ML.Estimator.new/2` without one has nothing to look the types up in.

  A frame is the one argument that is not a literal at all: it rides `Fetch.Method.Args`'s
  `input` arm as a relation, which is how the server gets a dataset into an attribute call.
  """
  @spec arguments!(Model.t() | Summary.t(), String.t(), [term()]) :: [term()]
  def arguments!(holder, wire, values) when is_list(values) do
    case attribute_row(holder, wire) do
      nil when values == [] ->
        []

      nil ->
        raise ArgumentError,
              "#{wire} takes no recorded arguments here, so #{length(values)} cannot be " <>
                "checked. #{unknown_class(holder)}"

      %{args: nil} when values == [] ->
        []

      %{args: nil} ->
        raise ArgumentError, "#{wire} takes no arguments; use Latu.ML.attribute/2"

      %{args: declared} when length(declared) != length(values) ->
        raise ArgumentError,
              "#{wire} takes #{length(declared)} arguments — " <>
                Enum.map_join(declared, ", ", &"#{&1.name} (#{inspect(&1.type)})") <>
                " — and #{length(values)} were given"

      %{args: declared} ->
        declared |> Enum.zip(values) |> Enum.map(fn {arg, value} -> argument(arg, value) end)
    end
  end

  # `is_struct/2` rather than a `%Latu.DataFrame{}` pattern: a pattern would need the struct at
  # compile time, and `dev/check_offline.exs` builds this module without Latu itself.
  defp argument(%{type: :frame}, frame) when is_struct(frame, Latu.DataFrame), do: frame.plan

  defp argument(%{type: :frame, name: name}, value) do
    raise ArgumentError, "#{name} is a DataFrame argument, and #{inspect(value)} is not one"
  end

  defp argument(%{type: type}, value), do: Plan.literal(value, type)

  # A holder whose class is decided elsewhere is looked up in every class it could be, in
  # PySpark's own order. Two shapes have that: a summary whose class the *fitted* model picks
  # (`BinaryLogisticRegressionTrainingSummary` or `LogisticRegressionTrainingSummary`), and an
  # `LDA` model, which is a `LocalLDAModel` or a `DistributedLDAModel` depending on its
  # `optimizer`.
  #
  # The **union**, deliberately, where `Latu.ML.attributes/1` reports the intersection: they
  # answer different questions. This one takes a name the caller has already chosen and asks
  # what type its argument is, and a name means the same thing in whichever candidate has it.
  # Listing is the question that must not over-promise; looking a type up is not.
  defp attribute_row(%{class: nil, candidates: [_ | _] = candidates}, wire) do
    Enum.find_value(candidates, &row_for(&1, wire))
  end

  defp attribute_row(%{class: nil}, _wire), do: nil
  defp attribute_row(%{class: class}, wire), do: row_for(class, wire)

  defp row_for(class, wire) do
    case Registry.class(class) do
      nil -> nil
      row -> Enum.find(row.attributes, &(&1.wire == wire))
    end
  end

  defp unknown_class(%{class: nil, candidates: [_ | _] = candidates}) do
    "None of #{Enum.join(candidates, " or ")} has it."
  end

  defp unknown_class(%{class: nil}) do
    "This model has no class — Latu.ML.Estimator.new/2 leaves it unset; pass model_class: to " <>
      "set it."
  end

  defp unknown_class(%{class: class}), do: "#{class} has no such attribute."

  @doc """
  The `Fetch` a helper method sends, as a command.

  `ConnectHelper` is reached by the same `Fetch` a model attribute is, and differs in
  everything else: the object is a singleton the server resolves by name, the arguments are
  positional, and three of the eleven answer with a model the call itself built.
  """
  @spec helper_plan(map(), [term()]) :: Plan.command()
  def helper_plan(row, values) do
    Plan.helper(row.method, helper_arguments!(row, values))
  end

  @doc "See `helper_plan/2`. The DataFrame-valued half, so a relation rather than a command."
  @spec helper_relation_plan(map(), [term()]) :: Plan.relation()
  def helper_relation_plan(row, values) do
    Plan.helper_relation(row.method, helper_arguments!(row, values))
  end

  @doc """
  A caller's values as the literals and relations a helper `Fetch` carries.

  `arguments!/3`'s twin, and the same rule: the types are the registry's, read off PySpark's
  own call sites rather than asked of the caller. The difference is that a helper has no class
  to look them up in — the method name is the key, because the object is a singleton.
  """
  @spec helper_arguments!(map(), [term()]) :: [term()]
  def helper_arguments!(row, values) when is_list(values) do
    declared = row.args

    if length(declared) != length(values) do
      raise ArgumentError,
            "#{row.method} takes #{length(declared)} arguments — " <>
              Enum.map_join(declared, ", ", &"#{&1.name} (#{inspect(&1.type)})") <>
              " — and #{length(values)} were given"
    end

    declared |> Enum.zip(values) |> Enum.map(fn {arg, value} -> argument(arg, value) end)
  end

  @doc """
  The whole positional argument list for an operator whose one method is a helper call.

  A helper's arguments are positional, so **every one is sent** whether the caller set it or
  not — where an `MlParams` map carries only what was set and the server fills the rest. Each
  value is the caller's, then the param's own default, then the fallback PySpark's call site
  records for a param Spark gives no default at all. `PowerIterationClustering.weightCol` is
  the only one on 4.2.0 that needs the third, and the call site says what it sends rather than
  this package deciding.

  A param with none of the three is refused by name: the alternative is inventing a value for
  something the caller never mentioned.
  """
  @spec operator_values!(map(), Helper.t(), Latu.DataFrame.t()) :: [term()]
  def operator_values!(row, %Helper{} = operator, frame) do
    [dataset | rest] = row.args

    if dataset.type != :frame do
      raise ArgumentError, "#{row.method} does not take a frame first, so nothing can send one"
    end

    set = Map.new(operator.params, fn {wire, _type, value} -> {wire, value} end)
    known = Map.new(operator.known, &{&1.name, &1})

    [frame | Enum.map(rest, &operator_value!(&1, set, known, row))]
  end

  defp operator_value!(argument, set, known, row) do
    param = Map.get(known, argument.name)

    cond do
      param && Map.has_key?(set, param.wire) -> Map.fetch!(set, param.wire)
      param && written_down?(param.default) -> param.default
      Map.has_key?(argument, :unset) -> argument.unset
      true -> no_value!(argument, row)
    end
  end

  # Four defaults in the registry are markers rather than values — a uid suffix, one only a
  # running server knows, one that is not the same twice, and NaN. Sending a marker would be
  # worse than refusing.
  defp written_down?(default) do
    not (is_nil(default) or default in [:varies, :server_supplied, :nan] or
           (is_tuple(default) and elem(default, 0) == :uid_suffix))
  end

  defp no_value!(argument, row) do
    raise ArgumentError,
          "#{row.method} sends every argument, and #{argument.name} has no value: set it on " <>
            "the operator, because Spark gives it no default this package can write down."
  end

  @doc """
  What a generated `Latu.ML.Feature.StringIndexerModel.from_labels/3` and its two siblings
  call: a uid, then `Latu.ML.helper/3`.

  Three of the eleven helper methods build a model server-side out of what you pass. The uid is
  the **client's** — the call's first declared argument — and sending it is how the model the
  server registers and the `%Latu.ML.Model{}` the caller holds agree on one; PySpark does the
  same. The generic `Latu.ML.helper/3` takes it like every other declared argument, because an
  escape hatch that quietly supplied one would be an escape hatch that surprises; a named
  constructor is the other way round, and this is where the uid is made.

  `opts` are params set on the model afterwards, as PySpark's own `model.setInputCol(...)`
  does, checked against that model class's own table.

  The send is `Latu.ML`'s, not this module's — the same split `attribute/4` follows, and the
  layering test is what holds it.
  """
  @spec model_from_helper(atom(), Latu.Session.t(), [term()], keyword()) ::
          {:ok, Model.t()} | {:error, Latu.Error.t()}
  def model_from_helper(name, session, values, opts) do
    row = Registry.helper!(name)

    known =
      case Registry.operator_of_class(row.model_class) do
        %{params: params} -> params
        nil -> []
      end

    with {:ok, %Model{} = model} <-
           Latu.ML.helper(session, name, [uid(row.model_class) | values]) do
      {:ok, %Model{model | params: params!(opts, known)}}
    end
  end

  @doc "The frame a helper's arguments carry, which is where its session comes from."
  @spec frame_of!(map(), [term()]) :: Latu.DataFrame.t()
  def frame_of!(row, values) do
    row.args
    |> Enum.zip(values)
    |> Enum.find_value(fn {argument, value} -> argument.type == :frame && value end) ||
      raise ArgumentError,
            "#{row.method} takes no frame, so there is no session to reach the server with"
  end

  @doc "The `Delete` plan `Latu.ML.delete/1` sends."
  @spec delete_plan(Model.t()) :: Plan.command()
  def delete_plan(%Model{} = model), do: Plan.delete([model.ref])
end
