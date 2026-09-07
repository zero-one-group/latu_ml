defmodule Latu.ML.Layout do
  @moduledoc false
  # Spark's on-disk layout for the parts of MLlib the server does not save.
  #
  # Everything with a server-side object goes out through `MlCommand.Write`, and the *server*
  # writes both its metadata and its data. A meta-algorithm has no server-side object at all —
  # a pipeline is a list this package holds — so the client writes the directory itself:
  #
  #     <path>/metadata            one JSON object, as a one-row text file
  #     <path>/stages/00_uid/      each stage, saved the ordinary way
  #
  # This module is that format, and nothing else: the JSON, the paths, and the two IO calls that
  # put them there. Which stages to save and in what order is the caller's business.
  #
  # ## The metadata is a compatibility claim
  #
  # `class` and `paramMap.language` say **PySpark**, and that one string is the whole claim.
  # Read against Scala's writer, the two formats differ by it and nothing else: `uid`,
  # `paramMap.stageUids` and the zero-padded stage directory names are identical in both.
  #
  # The two readers are not symmetric, which is what makes it a choice rather than a detail:
  #
  #   * PySpark 4.2's `PipelineReader.load` reads `uid` and `paramMap.stageUids` and stops. It
  #     never looks at `class` or `language` — those matter only to the *classic*, JVM-backed
  #     reader, which routes to `JavaMLReader` when `language` is missing.
  #   * Scala's `DefaultParamsReader.loadMetadata` does
  #     `require(className == expectedClassName, ...)`, so `Pipeline.load` on the JVM refuses a
  #     directory whose `class` names a Python module.
  #
  # So PySpark's string is readable by PySpark only, and Scala's would be readable by both.
  # Latu writes PySpark's anyway, because it is the one that can be *checked*:
  # `dev/pyspark_oracle.py --interop` reads what this wrote with a client already in the loop,
  # and claiming the Scala name would mean a JVM and a `Pipeline.load` smoke test to say
  # anything true about it. `docs/decisions.md` has that trade in full; `docs/deviations.md`
  # tells the user the resulting limitation, which is PySpark's own limitation too.
  #
  # Who actually wrote the file goes in a top-level `latuMl` key, beside `class`. Both readers
  # take the JSON apart by name — PySpark's `DefaultParamsReader` returns a plain dict and pulls
  # named keys; Scala's `parseMetadata` extracts six fields with json4s paths into a `Metadata`
  # case class — so an extra key is inert to both. Checked, not assumed.

  @metadata "metadata"
  @stages "stages"
  @sub_models "subModels"

  # What PySpark writes for each. `Latu.ML.load/4` dispatches on it and so does Scala's reader,
  # which `require`s an exact match; PySpark 4.2's own pipeline reader ignores it entirely.
  @classes %{
    pipeline: "pyspark.ml.pipeline.Pipeline",
    pipeline_model: "pyspark.ml.pipeline.PipelineModel",
    cross_validator: "pyspark.ml.tuning.CrossValidator",
    cross_validator_model: "pyspark.ml.tuning.CrossValidatorModel",
    train_validation_split: "pyspark.ml.tuning.TrainValidationSplit",
    train_validation_split_model: "pyspark.ml.tuning.TrainValidationSplitModel"
  }

  # Which of them lay out a directory of stages, and which lay out a search. Both are read by
  # `class`, so something has to say them apart — ML5a could assume every known class here was
  # a pipeline and no longer can.
  @shapes %{
    pipeline: :stages,
    pipeline_model: :stages,
    cross_validator: :search,
    cross_validator_model: :search,
    train_validation_split: :search,
    train_validation_split_model: :search
  }

  @doc "The `class` string a kind of meta-algorithm is saved under."
  def class(kind), do: Map.fetch!(@classes, kind)

  @doc "Which kind a `class` string names, or nil for anything that is not one of ours."
  def kind(class) do
    Enum.find_value(@classes, fn {kind, name} -> if name == class, do: kind end)
  end

  @doc """
  What a kind lays out on disk: `:stages` for a pipeline, `:search` for a validator.

  A directory says what it holds in its own `class`, and reading it back means dispatching on
  that. Two shapes, and nothing outside this module should be matching class strings to tell
  them apart.
  """
  def shape(kind), do: Map.fetch!(@shapes, kind)

  @doc "Whether a kind is a fitted thing (it has models in it) or an unfitted one."
  def fitted?(kind) do
    kind in [:pipeline_model, :cross_validator_model, :train_validation_split_model]
  end

  @doc """
  Write `<path>/metadata`.

  One row, one column, coalesced to one file, as PySpark does it — a directory of part files
  would still read back, but `Latu.first/2` would then depend on which part came first.
  """
  def write_metadata(session, path, kind, uid, param_map, extra \\ %{}) do
    with {:ok, version} <- Latu.spark_version(session),
         json = metadata(kind, uid, param_map, version, extra),
         {:ok, frame} <-
           Latu.create_dataframe(session, [value: [json]], schema: "value STRING") do
      frame
      |> Latu.coalesce(1)
      |> Latu.write(format: "text", path: Path.join(path, @metadata))
    end
  end

  @doc "Read `<path>/metadata` back, as the map it was written from."
  def read_metadata(session, path) do
    frame = Latu.read(session, format: "text", path: Path.join(path, @metadata))

    with {:ok, row} <- Latu.first(frame), do: decode(row, path)
  end

  defp decode(%{value: json}, path) when is_binary(json) do
    case JSON.decode(json) do
      {:ok, %{"class" => _class, "uid" => _uid} = metadata} -> {:ok, metadata}
      _other -> {:error, unreadable(path, "does not hold Spark's metadata JSON")}
    end
  end

  defp decode(nil, path), do: {:error, unreadable(path, "is empty")}
  defp decode(_row, path), do: {:error, unreadable(path, "has no `value` column")}

  defp unreadable(path, what) do
    %Latu.Error{
      kind: :decode,
      message: "#{Path.join(path, @metadata)} #{what}, so there is nothing to load here"
    }
  end

  @doc """
  The metadata JSON, exactly as `DefaultParamsWriter._get_metadata_to_save` shapes it — plus the
  one key that says who wrote it.

  Key *order* is not PySpark's: JSON is unordered, both readers take it apart by name, and
  Elixir's encoder sorts. Nothing compares these bytes.
  """
  def metadata(kind, uid, param_map, spark_version, extra \\ %{}) do
    # `extra` is PySpark's `extraMetadata`: a search's `avgMetrics`, `stdMetrics` and
    # `persistSubModels` sit at the top level beside `class`, not inside `paramMap`, and its
    # reader looks for them there.
    JSON.encode!(
      Map.merge(extra, %{
        "class" => class(kind),
        "timestamp" => System.system_time(:millisecond),
        "sparkVersion" => spark_version,
        "uid" => uid,
        "paramMap" => param_map,
        "defaultParamMap" => %{},
        "latuMl" => %{
          "version" => version(),
          "note" =>
            "Written by latu_ml, an Elixir client for Spark Connect. Nothing here was written " <>
              "by PySpark: the `class` and `paramMap.language` fields above are PySpark's " <>
              "spelling of this format, which is what makes the directory readable by PySpark " <>
              "and not by Scala Spark. See https://hexdocs.pm/latu_ml for what that costs."
        }
      })
    )
  end

  defp version do
    case Application.spec(:latu_ml, :vsn) do
      nil -> "unknown"
      vsn -> to_string(vsn)
    end
  end

  @doc """
  Where one stage of `count` goes, under `<path>/stages`.

  `00_uid`, zero-padded to the width of the largest index — PySpark's `getStagePath`, and the
  reason a nineteen-stage pipeline and a hundred-stage one name their first stage differently.
  """
  def stage_path(path, index, uid, count) do
    width = count |> Integer.to_string() |> String.length()
    name = index |> Integer.to_string() |> String.pad_leading(width, "0")

    Path.join([path, @stages, name <> "_" <> uid])
  end

  @doc """
  Where the parts of a saved search go.

  Flat names, PySpark's own: the estimator being searched, the evaluator that scored it, and —
  for a fitted one — the winner. A sub-model is indexed by param map, under its fold where
  there are folds and directly where there is one split.
  """
  @parts %{estimator: "estimator", evaluator: "evaluator", best_model: "bestModel"}

  def part_path(path, part), do: Path.join(path, Map.fetch!(@parts, part))

  def sub_model_path(path, fold, index) when is_integer(fold) do
    Path.join([path, @sub_models, "fold#{fold}", Integer.to_string(index)])
  end

  def sub_model_path(path, nil, index) do
    Path.join([path, @sub_models, Integer.to_string(index)])
  end

  @doc """
  A grid, as Spark's saved metadata spells it.

  Each entry is `parent` (the uid of the operator whose param it is), `name` (the **wire**
  name, `regParam` not `reg_param`), `value` and `isJson`. Latu's own grid entries already
  carry the uid, so this is a rename rather than a lookup.

  ## The value is encoded twice

  PySpark writes `json.dumps(v)` into the `value` field of a document that is itself JSON, and
  reads it back with `json.loads`. So `0.5` is stored as the **string** `"0.5"`, not the number
  `0.5`, and a file written with the number in it does not load. That is not a guess: the
  writer is `jsonParam["value"] = json.dumps(v)` and the reader is
  `value = json.loads(jsonParam["value"])`. Latu matches it, because the point of this format
  is that PySpark reads it.

  `isJson: false` means the value was itself an estimator, written to its own `epm_*`
  directory. No param in this registry takes an operator — `OneVsRest`, the one that does, is
  in neither extractor — so writing one is refused by name rather than half-implemented.
  """
  def encode_grid(param_maps, wire_of) do
    Enum.map(param_maps, fn param_map ->
      Enum.map(param_map, fn {uid, name, value} ->
        # The value first: a map literal evaluates in order, so checking it here rather than
        # inline means an operator-valued param is refused as such, instead of failing
        # wherever `wire_of` happens to look it up.
        encodable!(name, value)

        %{
          "parent" => uid,
          "name" => wire_of.(uid, name),
          "value" => JSON.encode!(value),
          "isJson" => true
        }
      end)
    end)
  end

  defp encodable!(name, %{__struct__: module}) do
    raise ArgumentError,
          "the grid sets #{inspect(name)} to a #{inspect(module)}. Spark saves an " <>
            "operator-valued param to its own directory beside the metadata, and no param in " <>
            "this registry takes one, so this package does not write that shape."
  end

  defp encodable!(_name, _value), do: :ok

  @doc """
  A saved grid, back to the `{uid, name, value}` entries `Latu.ML.param_grid/2` builds.

  `name_of` maps a uid and a wire name back to this package's own spelling, which is a registry
  lookup on the operator the uid belongs to — so an entry naming a param the estimator at this
  path does not have is a refusal rather than a silently dropped setting.
  """
  def decode_grid(encoded, name_of) when is_list(encoded) do
    Enum.map(encoded, fn param_map ->
      Enum.map(param_map, fn entry ->
        uid = Map.fetch!(entry, "parent")

        {uid, name_of.(uid, Map.fetch!(entry, "name")), decoded!(entry)}
      end)
    end)
  end

  defp decoded!(%{"isJson" => false} = entry) do
    raise ArgumentError,
          "the saved grid sets #{entry["name"]} from #{entry["value"]}, a directory holding " <>
            "an operator. This package does not read that shape; see Latu.ML.Layout."
  end

  defp decoded!(entry), do: JSON.decode!(Map.fetch!(entry, "value"))

  @doc """
  A validator's own params, as the saved `paramMap` spells them.

  PySpark's `extractJsonParams` with `estimator`, `evaluator` and `estimatorParamMaps` skipped,
  and then the grid put back. A nil is left out rather than written as null: an absent key
  means the reader keeps its own default, where a null would be a value.

  ## A fitted search carries fewer keys than an unfitted one

  Not tidiness. `DefaultParamsReader.getAndSetParams` calls `getParam(name)` for **every** key
  in `paramMap` and raises on one the instance does not have — and `CrossValidatorModel` has no
  `collectSubModels`, where `CrossValidator` does. PySpark never trips on this because it
  extracts from the instance in front of it; a client writing the map itself has to know.
  Writing that key on a model makes `CrossValidatorModel.load` raise in PySpark and nowhere
  else, so nothing here would ever have caught it.
  """
  def search_params(validator, kind, wire_of) do
    own =
      case validator do
        %Latu.ML.CrossValidator{} ->
          %{"numFolds" => validator.num_folds, "foldCol" => validator.fold_col || ""}

        %Latu.ML.TrainValidationSplit{} ->
          %{"trainRatio" => validator.train_ratio}
      end

    own
    |> collecting(validator, kind)
    |> Map.put("estimatorParamMaps", encode_grid(validator.param_maps, wire_of))
    |> then(fn map -> if validator.seed, do: Map.put(map, "seed", validator.seed), else: map end)
  end

  defp collecting(map, validator, kind) do
    if fitted?(kind) do
      map
    else
      Map.put(map, "collectSubModels", validator.collect_sub_models)
    end
  end
end
