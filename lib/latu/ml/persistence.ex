defmodule Latu.ML.Persistence do
  @moduledoc false
  # `Latu.ML.save/3` and `load/4`, and everything under them. The docstrings stay on `Latu.ML`,
  # where `h` reads a contract; this is what runs. `docs/decisions.md`, 2026-09-11.

  alias Latu.Error
  alias Latu.ML.Cache
  alias Latu.ML.CrossValidator
  alias Latu.ML.CrossValidatorModel
  alias Latu.ML.Estimator
  alias Latu.ML.Evaluator
  alias Latu.ML.Internal
  alias Latu.ML.Layout
  alias Latu.ML.Model
  alias Latu.ML.Pipeline
  alias Latu.ML.PipelineModel
  alias Latu.ML.Registry
  alias Latu.ML.Result
  alias Latu.ML.TrainValidationSplit
  alias Latu.ML.TrainValidationSplitModel
  alias Latu.ML.Transformer

  # Every kind that is a directory this package wrote rather than something the server reads.
  @meta_kinds Internal.meta_algorithms()

  # =============================================
  # Saving
  # =============================================

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
    |> Cache.each_ok(fn {fold, index} -> save_fold(session, fold, path, index, opts) end)
  end

  defp save_sub_models(session, %TrainValidationSplitModel{} = model, path, opts) do
    save_fold(session, model.sub_models, path, nil, opts)
  end

  defp save_fold(session, models, path, fold, opts) do
    models
    |> Enum.with_index()
    |> Cache.each_ok(fn {model, index} ->
      save_stage(session, model, Layout.sub_model_path(path, fold, index), opts)
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
      |> Cache.each_ok(fn {stage, index} ->
        where = Layout.stage_path(path, index, stage.uid, length(stages))
        save_stage(session, stage, where, opts)
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
    with {:ok, _cleared} <- Latu.ML.helper(session, :handle_overwrite, [path, true]), do: :ok
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

  # =============================================
  # Loading
  # =============================================

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
        Cache.release(Cache.collected([estimator]))
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
          Cache.release(Cache.collected([best_model]))
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
    Cache.traverse(folds, &read_fold(session, path, &1, width))
  end

  defp read_fold(session, path, fold, width) do
    Cache.traverse(0..(width - 1), &load_stage(session, Layout.sub_model_path(path, fold, &1)))
  end

  defp rebuilt_search(kind, uid, estimator, evaluator, params, name_of) do
    grid = Layout.decode_grid(Map.get(params, "estimatorParamMaps", []), name_of)
    common = [estimator: estimator, param_maps: grid, evaluator: evaluator]

    built =
      case kind do
        model when model in [:cross_validator, :cross_validator_model] ->
          Latu.ML.cross_validator(
            common ++
              [num_folds: Map.get(params, "numFolds", 3)] ++
              maybe(params, "seed", :seed) ++ fold_col(params)
          )

        _split ->
          Latu.ML.train_validation_split(
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
    |> Cache.traverse(fn {uid, index} ->
      load_stage(session, Layout.stage_path(path, index, uid, length(uids)))
    end)
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
end
