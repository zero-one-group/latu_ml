# Probe: which of the 111 registered operators the server actually loads, and which of the 669
# allowlisted attributes actually answer.
#
#     docker compose up -d spark-connect
#     mix run dev/probe_ml.exs                 # report, change nothing
#     mix run dev/probe_ml.exs -- --write      # rewrite priv/ml_probe.exs, then `mix compile`
#     mix run dev/probe_ml.exs -- --recovery   # also watch a dropped summary come back (~80 s)
#     mix run dev/probe_ml.exs -- --interop    # also save a model where PySpark can read it
#
# `probe_dtypes.exs`'s job, for ML. The extractors read PySpark and the server's own Scala, and
# both can only be wrong in the same way PySpark is; **the server is the only oracle for
# behaviour**, and this is the thing that asks it. What it passes is what the docs are allowed
# to call supported: `Latu.ML.Registry` reads `priv/ml_probe.exs` and every constructor's `@doc`
# opens with the status it finds there.
#
# ## What a status means
#
#   * `:probed` — an estimator fitted, a transformer applied and executed, or an evaluator run
#     to a number, on the fixture below; and every zero-argument allowlisted attribute of the
#     resulting model answered —
#     and, for the ten models that record one, every attribute of its training summary too.
#     Where the fit decides the class (`LDA`, and the two model-dependent summaries) that means
#     the attributes allowed on *every* class it could be, which is what `Latu.ML.attributes/1`
#     promises; the ones only one branch allows are not reached from here.
#   * `:built` — it did not, and the reason is kept. That is not a verdict on the operator: far
#     more likely the fixture below is the wrong shape for it, which is the first thing to look
#     at when reading a `:built` reason.
#   * `:missing` — the server would not load the class at all. This is the one that matters for
#     the registry, because it is the one PySpark cannot tell us.
#
# ## What is not probed, and why
#
#   * **Models**, directly. A model is reached through the estimator that fits it, which is how
#     its attributes get probed.
#   * **The two `:helper` operators**, in this loop. `PowerIterationClustering` and `PrefixSpan`
#     are neither fitted nor applied, so there is nothing here to run them with; the helper
#     section near the end runs both, along with the other eight methods on that object.
#   * **Attributes that take arguments**, in the per-operator loop. `Latu.ML.attribute/3` sends
#     them and a section near the end asks one per recorded argument type, but a hundred of
#     them need a hundred plausible arguments and that is not what a status is for.
#   * **The values anything returns.** Never this package's claim: the server is the oracle for
#     what it *accepts and answers*, and for nothing beyond that.
#
# ## The fixture
#
# Four small frames, built with `sql/2` so the probe needs no Arrow, no Explorer and no data
# files. Their columns carry **Spark's own default names** — `features`, `label`, `user`,
# `item`, `rating`, `items` — so most operators need no params at all, and the per-operator
# table below is only the ones that do. Every entry in it is a guess about shape that the run
# either confirms or turns into a reason.
#
# Each operator is run in its own `try`. One refusal costs one row.

alias Latu.ML
alias Latu.ML.Feature
alias Latu.ML.Registry

defmodule ProbeML do
  @moduledoc false

  # Which fixture frame an operator wants, and what it needs told beyond the defaults. An
  # operator absent from here gets `:features` and no params, which is right for every
  # predictor and every clusterer — they read `featuresCol` and `labelCol`, and the frame
  # already carries both under those names.
  @config %{
    # --- clustering: the defaults would ask for more clusters than there are rows -----------
    k_means: {:features, [k: 2]},
    bisecting_k_means: {:features, [k: 2]},
    gaussian_mixture: {:features, [k: 2]},
    lda: {:features, [k: 2, max_iter: 2]},

    # --- classification and regression: only the ones with a required column of their own ---
    aft_survival_regression: {:features, [censor_col: :censor, label_col: :survival]},
    multilayer_perceptron_classifier: {:features, [layers: [2, 2, 2], max_iter: 2]},

    # --- feature operators over the Vector column -------------------------------------------
    standard_scaler: {:features, [input_col: :features, output_col: :scaled]},
    min_max_scaler: {:features, [input_col: :features, output_col: :scaled]},
    max_abs_scaler: {:features, [input_col: :features, output_col: :scaled]},
    robust_scaler: {:features, [input_col: :features, output_col: :scaled]},
    normalizer: {:features, [input_col: :features, output_col: :scaled]},
    pca: {:features, [input_col: :features, output_col: :pcs, k: 1]},
    dct: {:features, [input_col: :features, output_col: :dct]},
    polynomial_expansion: {:features, [input_col: :features, output_col: :poly, degree: 2]},
    idf: {:features, [input_col: :features, output_col: :idf]},
    vector_indexer:
      {:features, [input_col: :features, output_col: :indexed, max_categories: 2]},
    variance_threshold_selector: {:features, [features_col: :features, output_col: :selected]},
    chi_sq_selector:
      {:features, [features_col: :features, output_col: :selected, num_top_features: 1]},
    univariate_feature_selector:
      {:features,
       [
         features_col: :features,
         label_col: :label,
         output_col: :selected,
         feature_type: "continuous",
         label_type: "categorical",
         selection_mode: "numTopFeatures",
         selection_threshold: 1.0
       ]},
    vector_slicer: {:features, [input_col: :features, output_col: :sliced, indices: [0]]},
    vector_size_hint: {:features, [input_col: :features, size: 2]},
    elementwise_product:
      {:features,
       [input_col: :features, output_col: :scaled, scaling_vec: Nx.tensor([1.0, 2.0])]},
    bucketed_random_projection_lsh:
      {:features, [input_col: :features, output_col: :hashes, bucket_length: 2.0]},
    min_hash_lsh: {:features, [input_col: :features, output_col: :hashes]},
    vector_assembler: {:features, [input_cols: [:x1, :x2], output_col: :assembled]},
    interaction: {:features, [input_cols: [:x1, :x2], output_col: :interacted]},

    # --- feature operators over scalars ------------------------------------------------------
    binarizer: {:features, [input_col: :x1, output_col: :above, threshold: 1.0]},
    bucketizer:
      {:features,
       [
         input_col: :x1,
         output_col: :bucket,
         splits: [-1.0e9, 1.0, 1.0e9],
         handle_invalid: "keep"
       ]},
    quantile_discretizer: {:features, [input_col: :x1, output_col: :bucket, num_buckets: 2]},
    imputer: {:features, [input_cols: [:x1], output_cols: [:x1_imputed]]},
    one_hot_encoder: {:features, [input_cols: [:category_index], output_cols: [:category_hot]]},
    string_indexer: {:features, [input_col: :category, output_col: :category_idx]},
    index_to_string:
      {:features,
       [input_col: :category_index, output_col: :category_back, labels: ["a", "b", "c"]]},
    target_encoder:
      {:features,
       [input_cols: [:category_index], output_cols: [:category_te], label_col: :label]},
    feature_hasher: {:features, [input_cols: [:x1, :x2], output_col: :hashed]},
    r_formula:
      {:features, [formula: "label ~ x1 + x2", features_col: :rf_features, label_col: :rf_label]},
    sql_transformer: {:features, [statement: "SELECT * FROM __THIS__"]},

    # --- text --------------------------------------------------------------------------------
    tokenizer: {:text, [input_col: :sentence, output_col: :tokens]},
    regex_tokenizer: {:text, [input_col: :sentence, output_col: :tokens]},
    stop_words_remover: {:text, [input_col: :words, output_col: :kept]},
    ngram: {:text, [input_col: :words, output_col: :grams, n: 2]},
    count_vectorizer: {:text, [input_col: :words, output_col: :counts]},
    hashing_tf: {:text, [input_col: :words, output_col: :tf]},
    word2vec:
      {:text, [input_col: :words, output_col: :vec, vector_size: 2, min_count: 0]},

    # --- evaluators: each reads a *scored* frame, which nothing above produces -----------------
    #
    # An evaluator is the one kind that takes predictions rather than features, so its fixtures
    # are hand-written rather than fitted: three frames, chosen so that between them they cover
    # the three shapes the six want — scalars, a Vector with a cluster beside it, and arrays.
    binary_classification_evaluator: {:scored, []},
    multiclass_classification_evaluator: {:scored, []},
    regression_evaluator: {:scored, []},
    clustering_evaluator: {:clustered, []},
    multilabel_classification_evaluator: {:ranked, []},
    ranking_evaluator: {:ranked, []},

    # --- the two frames that exist for one operator each ---------------------------------------
    als: {:ratings, [rank: 2, max_iter: 2, num_item_blocks: 1, num_user_blocks: 1]},
    fp_growth: {:transactions, [min_support: 0.1]}
  }

  # The fixture frames `frames/1` builds, and the only names `@config` may point at.
  @frames [:features, :text, :ratings, :transactions, :scored, :clustered, :ranked]

  def config, do: @config

  @doc """
  Check the table above against the registry before reaching for a server.

  Every key must be a real operator, and every param name a real param of it. A stale entry
  would otherwise turn into a `:built` reason that reads like the server's refusal and is not,
  which is the worst kind of noise in a file whose whole job is to be read.
  """
  def validate! do
    problems =
      for {name, {frame, params}} <- Enum.sort(@config) do
        case Registry.fetch(name) do
          :error ->
            "#{name}: no such operator"

          {:ok, operator} ->
            known = MapSet.new(operator.params, & &1.name)
            unknown = for {key, _value} <- params, not MapSet.member?(known, key), do: key

            cond do
              frame not in @frames ->
                "#{name}: no fixture frame #{inspect(frame)}"

              unknown != [] ->
                "#{name}: no param #{inspect(unknown)} — see Latu.ML.params(#{inspect(name)})"

              operator.kind not in [:estimator, :transformer, :evaluator] ->
                "#{name}: a #{operator.kind} is not run"

              true ->
                nil
            end
        end
      end
      |> Enum.reject(&is_nil/1)

    if problems != [] do
      IO.puts("the probe's own table is stale:")
      for problem <- problems, do: IO.puts("  " <> problem)
      System.halt(1)
    end
  end

  # =============================================
  # The fixture
  # =============================================

  def frames(session) do
    # **Every numeric column is cast.** `VALUES (0.0)` is a `DECIMAL(2,1)`, not a double, and
    # while `VectorAssembler` and the predictors cast it on the way in, an operator reading a
    # raw column does not: `Binarizer` answers "Data type DecimalType(2,1) is not supported".
    # `survival` is positive because `AFTSurvivalRegression` takes the log of its label.
    base =
      Latu.sql!(session, """
      SELECT
        CAST(label AS DOUBLE) AS label,
        CAST(x1 AS DOUBLE) AS x1,
        CAST(x2 AS DOUBLE) AS x2,
        CAST(weight AS DOUBLE) AS weight,
        CAST(censor AS DOUBLE) AS censor,
        category,
        CAST(category_index AS DOUBLE) AS category_index,
        CAST(survival AS DOUBLE) AS survival
      FROM VALUES
        (0.0, 0.0, 1.1, 1.0, 1.0, 'a', 0.0, 1.2),
        (0.0, 0.1, 1.0, 1.0, 1.0, 'b', 1.0, 2.4),
        (1.0, 2.0, 0.1, 1.0, 0.0, 'a', 0.0, 0.8),
        (1.0, 2.1, 0.2, 1.0, 1.0, 'c', 2.0, 3.1)
      AS t(label, x1, x2, weight, censor, category, category_index, survival)
      """)

    assembler = Feature.vector_assembler(input_cols: [:x1, :x2], output_col: :features)

    %{
      base: base,
      features: ML.transform(assembler, base),
      text:
        Latu.sql!(session, """
        SELECT sentence, split(sentence, ' ') AS words FROM VALUES
          ('a b c'), ('b c d'), ('a c d'), ('b d e')
        AS t(sentence)
        """),
      ratings:
        Latu.sql!(session, """
        SELECT * FROM VALUES
          (0, 0, 4.0), (0, 1, 2.0), (1, 0, 1.0), (1, 1, 5.0)
        AS t(user, item, rating)
        """),
      transactions:
        Latu.sql!(session, """
        SELECT split(line, ' ') AS items FROM VALUES
          ('a b c'), ('a b'), ('a c'), ('b c')
        AS t(line)
        """),

      # An evaluator reads a frame something has already scored, so these are written rather
      # than fitted — a fixture that depended on a fit would make an evaluator's status a
      # statement about the classifier in front of it.
      #
      # `rawPrediction` is a double here, not a Vector: `BinaryClassificationEvaluator` takes
      # either, and a double is one fewer moving part.
      scored:
        Latu.sql!(session, """
        SELECT
          CAST(label AS DOUBLE) AS label,
          CAST(prediction AS DOUBLE) AS prediction,
          CAST(raw AS DOUBLE) AS rawPrediction
        FROM VALUES (0.0, 0.0, -1.2), (0.0, 1.0, 0.4), (1.0, 1.0, 1.5), (1.0, 1.0, 2.0)
        AS t(label, prediction, raw)
        """),

      # `ClusteringEvaluator` is the one that wants features *and* a cluster, because a
      # silhouette is about the geometry rather than about a label.
      clustered:
        ML.transform(
          assembler,
          Latu.sql!(session, """
          SELECT
            CAST(x1 AS DOUBLE) AS x1,
            CAST(x2 AS DOUBLE) AS x2,
            CAST(prediction AS DOUBLE) AS prediction
          FROM VALUES (0.0, 1.1, 0.0), (0.1, 1.0, 0.0), (2.0, 0.1, 1.0), (2.1, 0.2, 1.0)
          AS t(x1, x2, prediction)
          """)
        ),

      # The two evaluators whose columns are *arrays* of doubles — a set of predicted labels
      # against a set of true ones. The casts are inside the arrays, for the same reason every
      # other numeric column here is cast.
      ranked:
        Latu.sql!(session, """
        SELECT
          array(CAST(a AS DOUBLE), CAST(b AS DOUBLE)) AS prediction,
          array(CAST(c AS DOUBLE), CAST(d AS DOUBLE)) AS label
        FROM VALUES (0.0, 1.0, 0.0, 2.0), (1.0, 2.0, 1.0, 2.0)
        AS t(a, b, c, d)
        """)
    }
  end

  # =============================================
  # One operator
  # =============================================

  def probe(operator, frames) do
    {frame_name, params} = Map.get(@config, operator.name, {:features, []})
    frame = Map.fetch!(frames, frame_name)

    try do
      built = apply_constructor(operator, params)

      case operator.kind do
        :estimator -> fit_and_fetch(built, frame)
        :transformer -> apply_and_run(built, frame)
        :evaluator -> evaluate_to_a_number(built, frame)
      end
    rescue
      error -> {:built, trim(Exception.message(error)), %{}}
    catch
      kind, value -> {:built, trim("#{kind}: #{inspect(value)}"), %{}}
    end
  end

  defp apply_constructor(%{constructor: {module, name, 1}}, params) do
    apply(module, name, [params])
  end

  defp fit_and_fetch(estimator, frame) do
    case ML.fit(estimator, frame) do
      {:ok, model} ->
        try do
          attributes = fetch_all(model)
          failed = for {name, {:error, _} = result} <- attributes, do: {name, result}

          if failed == [] do
            {:probed, nil, summarise(attributes)}
          else
            {:built, "fitted, but " <> refusals(failed), summarise(attributes)}
          end
        after
          ML.delete(model)
        end

      {:error, error} ->
        {status(error), reason(error), %{}}
    end
  end

  defp apply_and_run(transformer, frame) do
    transformed = ML.transform(transformer, frame)

    # `to_arrow/2` rather than `collect/2`: it bypasses the decoder and the dtype guard, so a
    # Vector output column does not turn a working transformer into a refusal about UDTs. What
    # is under test is whether the server loads the operator and runs it.
    case Latu.to_arrow(Latu.limit(transformed, 1)) do
      {:ok, _batches} -> {:probed, nil, %{}}
      {:error, error} -> {status(error), reason(error), %{}}
    end
  end

  # An evaluator has no model behind it and no attributes to fetch: it answers with a number or
  # it does not answer. `Latu.ML.evaluate/2` is what ML4 added, and this is the first thing to
  # send an `Evaluate` — the command itself was goldened against PySpark a milestone earlier.
  defp evaluate_to_a_number(evaluator, frame) do
    case ML.evaluate(evaluator, frame) do
      {:ok, metric} when is_float(metric) -> {:probed, nil, %{}}
      {:ok, other} -> {:built, "answered #{inspect(other)}, where a metric belongs", %{}}
      {:error, error} -> {status(error), reason(error), %{}}
    end
  end

  # Every zero-argument attribute the class is allowed, by the registry's own reckoning. A
  # value is fetched; a frame is resolved with `schema/1`, which is a real round trip through
  # the server's planner without decoding a Vector column this package cannot read yet.
  defp fetch_all(model) do
    for holder <- [model | summary_of(model)],
        attribute <- ML.attributes(holder) || [],
        attribute.arity == 0,
        attribute.returns in [:value, :frame],
        into: [] do
      {label(holder, attribute), fetch_one(holder, attribute)}
    end
  end

  # Ten of the 43 model classes record a training summary. Getting one sends nothing — the
  # reference is composed from the model's own — so it costs nothing to ask, and the 370
  # attributes behind it are the largest part of the surface no server had seen.
  defp summary_of(model) do
    [ML.summary(model)]
  rescue
    ArgumentError -> []
  end

  defp label(%Latu.ML.Summary{}, attribute), do: :"summary.#{attribute.name}"
  defp label(_model, attribute), do: attribute.name

  defp fetch_one(holder, %{returns: :value} = attribute) do
    case ML.attribute(holder, attribute.wire) do
      {:ok, _value} -> :ok
      {:error, error} -> {:error, reason(error)}
    end
  rescue
    # A decode that raises is this package's bug, not the server's, and it is worth exactly one
    # attribute: `NaiveBayesModel.sigma` is a 0x0 Matrix on a multinomial model, and the first
    # run of this probe is how `Latu.ML.Linalg` found out it could not read one.
    error -> {:error, trim(Exception.message(error))}
  end

  defp fetch_one(holder, %{returns: :frame} = attribute) do
    case Latu.schema(ML.attribute_frame(holder, attribute.wire)) do
      {:ok, _schema} -> :ok
      {:error, error} -> {:error, reason(error)}
    end
  rescue
    error -> {:error, trim(Exception.message(error))}
  end

  # Named, not counted. A run that says "1 attribute refused" is a run you have to do again
  # with more printing, which is the whole cost of a probe against a live server.
  defp refusals(failed) do
    Enum.map_join(failed, "; ", fn {name, {:error, reason}} -> "#{name}: #{reason}" end)
  end

  # A sorted keyword list rather than a map: the file is committed, and a map's inspect order is
  # not something to bet a clean regenerate-and-diff on.
  defp summarise(attributes), do: Enum.sort(attributes)

  # The one distinction the registry cares about: an operator the server would not load at all,
  # against one that loaded and then went wrong. Spark says the first with a message naming the
  # class; there is no error class for it, so the text is what there is.
  defp status(%Latu.Error{} = error) do
    message = error.message || ""

    if String.contains?(message, "Unsupported ML operator") or
         String.contains?(message, "ClassNotFoundException") or
         String.contains?(message, "not registered") do
      :missing
    else
      :built
    end
  end

  defp reason(%Latu.Error{} = error) do
    trim(Enum.join(Enum.reject([error.error_class, error.message], &is_nil/1), ": "))
  end

  @doc "The first line of a server's message, for a report that has to stay readable."
  def first_line(text), do: trim(text)

  defp trim(text) do
    text
    |> String.split("\n")
    |> hd()
    |> String.slice(0, 240)
    |> String.trim()
  end

  # =============================================
  # Reporting and writing
  # =============================================

  def report(results) do
    by_status = Enum.frequencies_by(results, fn {_name, {status, _reason, _attrs}} -> status end)

    IO.puts("")

    for status <- [:probed, :built, :missing], count = Map.get(by_status, status, 0), count > 0 do
      IO.puts("  #{String.pad_leading(to_string(count), 4)}  #{status}")
    end

    unclean =
      for {name, {status, reason, _attrs}} <- results, status != :probed, do: {name, reason}

    if unclean != [] do
      IO.puts("\nnot probed:")

      for {name, reason} <- Enum.sort(unclean) do
        IO.puts("  " <> to_string(name))
        IO.puts("      " <> (reason || "(no reason recorded)"))
      end
    end
  end

  def render(results, spark) do
    rows =
      results
      |> Enum.sort_by(fn {name, _} -> name end)
      |> Enum.map_join(",\n", fn {name, {status, reason, attributes}} ->
        """
            #{name}: %{
              status: :#{status},
              reason: #{inspect(reason)},
              attributes: #{inspect(attributes, limit: :infinity)}
            }\
        """
      end)

    """
    # Generated by dev/probe_ml.exs against a live Spark #{spark} server. Do not hand-edit.
    #
    # What the server accepted and answered, operator by operator. `Latu.ML.Registry` reads this
    # and it is where every constructor's status line comes from; delete it and everything falls
    # back to `:built`, which is what a package with no probe result is entitled to claim.
    #
    #     mix run dev/probe_ml.exs -- --write
    #
    # A `:built` reason is far more often this fixture being the wrong shape for the operator
    # than the operator being broken. Read dev/probe_ml.exs's own table first.
    %{
      spark: "#{spark}",
      operators: %{
    #{rows}
      }
    }
    """
  end
end

# =============================================
# Run
# =============================================

url = System.get_env("SPARK_REMOTE", "sc://localhost:15002")
write? = "--write" in System.argv()

ProbeML.validate!()

session = Latu.connect!(url)
spark = Latu.spark_version!(session)

IO.puts("Spark #{spark} — ML operator probe (#{url})")

frames = ProbeML.frames(session)

runnable =
  Enum.filter(Registry.operators(), &(&1.kind in [:estimator, :transformer, :evaluator]))

IO.puts(
  "#{length(runnable)} operators to run; models and the two helper operators are not — " <>
    "see the header\n"
)

results =
  for operator <- runnable do
    {status, reason, attributes} = ProbeML.probe(operator, frames)

    IO.write(
      case status do
        :probed -> "."
        :built -> "x"
        :missing -> "?"
      end
    )

    {operator.name, {status, reason, attributes}}
  end

ProbeML.report(results)

# =============================================
# The cache verbs
# =============================================
#
# Measured 2026-09-06, and this settles the `[]` ML1 left open: `GetCacheInfo` answers correctly
# on the session that holds the model — one entry, with the same byte count `model_size/1`
# reports — so the empty list ML1 saw was an empty cache, not a broken verb. The raw
# `MlCommandResult` is printed beside the decoded list so a future `[]` can be told apart from
# a decode that lost its elements without another reading session.
#
# Two other things this run says, both worth watching for a change:
#
#   * **A summary costs no cache entry.** `cache_info` does not grow after one is read, because
#     `Fetch` registers a `Summary` only when the *chain ends* on one, and Latu's chains always
#     end on the attribute — `["summary", "totalIterations"]`, never `["summary"]` alone.
#   * `CACHE_INVALID`'s text calls a deleted **model** a *Summary object*. Upstream finding #11,
#     confirmed here on the model path too.
#
# It costs one fit, so it is not behind a flag.

IO.puts("\nthe cache verbs")

cache_say = fn label, result ->
  detail =
    case result do
      {:ok, value} -> "ok — " <> inspect(value, limit: 5, printable_limit: 120)
      {:error, error} -> "#{inspect(error.error_class)} — " <> ProbeML.first_line(error.message)
      other -> inspect(other)
    end

  IO.puts("  " <> String.pad_trailing(label, 34) <> detail)
end

case ML.fit(Latu.ML.Classification.logistic_regression(max_iter: 2), frames.features) do
  {:ok, cached} ->
    cache_say.("model_size, one live model", ML.model_size(cached))
    cache_say.("cache_info, same session", ML.cache_info(session))

    # The raw arm, undecoded: if `cache_info/1` says `[]` this says whether the server sent an
    # empty array or something `Latu.Result.Literal` read as one.
    raw =
      with {:ok, execution} <- Latu.Client.execute_command(session, Latu.ML.Plan.get_cache_info()) do
        {:ok, execution.ml_command_result}
      end

    cache_say.("...and the undecoded result", raw)

    # Measured: this does *not* grow the list. A `Fetch` registers a `Summary` only when the
    # chain ends on one, and Latu's chains end on the attribute. `ML.summary/1` sends nothing
    # at all, hence the attribute read.
    summary = ML.summary(cached)
    ML.attribute(summary, "totalIterations")
    cache_say.("cache_info, after a summary", ML.cache_info(session))

    cache_say.("clean_cache, how many it dropped", ML.clean_cache(session))
    cache_say.("cache_info, after the clean", ML.cache_info(session))

    # The proof that `clean_cache/1` is as blunt as its docs say: this model was never deleted.
    cache_say.("the model, after the clean", ML.attribute(cached, "intercept"))

  {:error, error} ->
    cache_say.("fit", {:error, error})
end

# =============================================
# The hundred attributes that take arguments
# =============================================
#
# `priv/ml_attributes.exs` now records each argument's name and type, read off PySpark's own
# annotations, and `Latu.ML.attribute/3` builds the literals from that. Nothing about the
# recording is verifiable by reading: an annotation that says `float` where the JVM wants a
# `double`, or a `Vector` where it wants a `double`, is a `MethodNotFound` on the server and
# looks exactly like a correct one until it runs.
#
# So this asks one attribute per recorded type — vector, double, int, string, frame, the
# four-argument case, and both arms of 4.2.0's one union — and prints what came back. Six fits,
# no flag.

IO.puts("\nattributes that take arguments")

# Each ask in its own `try`, like the operator loop above: an argument type that is wrong is
# exactly what this section exists to find, and finding one must not cost the other nine lines.
argued = fn label, fun ->
  detail =
    try do
      case fun.() do
        {:ok, value} ->
          "ok — " <> String.replace(inspect(value, limit: 3, printable_limit: 60), "\n", " ")

        {:error, error} ->
          "#{inspect(error.error_class)} — " <> ProbeML.first_line(error.message)

        other ->
          inspect(other)
      end
    rescue
      e -> "raised #{inspect(e.__struct__)} — " <> ProbeML.first_line(Exception.message(e))
    end

  IO.puts("  " <> String.pad_trailing(label, 40) <> detail)
end

# A frame-valued one: built lazily, so nothing is proved until it is collected.
rows = fn frame ->
  case Latu.to_arrow(Latu.limit(frame, 1)) do
    {:ok, _batches} -> {:ok, :collected}
    other -> other
  end
end

with_fit = fn label, operator, data, fun ->
  case ML.fit(operator, data) do
    {:ok, model} ->
      try do
        fun.(model)
      after
        ML.delete(model)
      end

    {:error, error} ->
      argued.(label <> ", fit", fn -> {:error, error} end)
  end
end

key = Nx.tensor([1.0, 1.0])

lr = Latu.ML.Classification.logistic_regression(max_iter: 2)

with_fit.("logistic_regression", lr, frames.features, fn model ->
  argued.("predict — a vector", fn -> ML.attribute(model, "predict", [key]) end)
  argued.("predictRaw — a vector", fn -> ML.attribute(model, "predictRaw", [key]) end)

  # A summary whose class the fitted model decides, so the argument's type is looked up in both
  # candidates. This is the line an earlier run died on, before they were recorded.
  summary = ML.summary(model)
  argued.("fMeasureByLabel — a double", fn -> ML.attribute(summary, "fMeasureByLabel", [1.0]) end)
end)

gbt = Latu.ML.Classification.gbt_classifier(max_iter: 2)

with_fit.("gbt_classifier", gbt, frames.features, fn model ->
  argued.("evaluateEachIteration — a frame", fn ->
    ML.attribute(model, "evaluateEachIteration", [frames.features])
  end)
end)

als = Latu.ML.Recommendation.als(rank: 2, max_iter: 2, num_item_blocks: 1, num_user_blocks: 1)

with_fit.("als", als, frames.ratings, fn model ->
  argued.("recommendForAllUsers — an int", fn ->
    rows.(ML.attribute_frame(model, "recommendForAllUsers", [2]))
  end)

  argued.("recommendForUserSubset — frame, int", fn ->
    rows.(ML.attribute_frame(model, "recommendForUserSubset", [frames.ratings, 2]))
  end)
end)

with_fit.("lda", Latu.ML.Clustering.lda(k: 2, max_iter: 2), frames.features, fn model ->
  argued.("describeTopics — an int", fn ->
    rows.(ML.attribute_frame(model, "describeTopics", [2]))
  end)
end)

w2v = Feature.word2vec(input_col: :words, output_col: :vec, vector_size: 2, min_count: 0)

with_fit.("word2vec", w2v, frames.text, fn model ->
  # 4.2.0's one union-typed argument, asked both ways. The registry records
  # `{:either, [:string, :vector]}` and the value's own shape picks the arm.
  argued.("findSynonyms — the string arm", fn ->
    rows.(ML.attribute_frame(model, "findSynonyms", ["a", 2]))
  end)

  argued.("findSynonyms — the vector arm", fn ->
    rows.(ML.attribute_frame(model, "findSynonyms", [Nx.tensor([0.1, 0.2]), 2]))
  end)
end)

lsh =
  Feature.bucketed_random_projection_lsh(
    input_col: :features,
    output_col: :hashes,
    bucket_length: 2.0
  )

with_fit.("bucketed_random_projection_lsh", lsh, frames.features, fn model ->
  argued.("approxNearestNeighbors — four arguments", fn ->
    rows.(ML.attribute_frame(model, "approxNearestNeighbors", [frames.features, key, 2, "dist"]))
  end)
end)

# The same calls through the **generated accessors**, which is how a caller is meant to make
# them: the class is checked before anything is sent and the arguments are named. Two of them,
# one per answer shape, because what is under test is the generator rather than the wire.
with_fit.("generated accessors", lr, frames.features, fn model ->
  argued.("LogisticRegressionModel.predict/2", fn ->
    Latu.ML.Classification.LogisticRegressionModel.predict(model, key)
  end)

  argued.("...and refuses a model of another class", fn ->
    Latu.ML.Clustering.KMeansModel.predict(model, key)
  end)
end)

with_fit.("generated accessors, frame", als, frames.ratings, fn model ->
  argued.("ALSModel.recommend_for_all_users/2", fn ->
    rows.(Latu.ML.Recommendation.ALSModel.recommend_for_all_users(model, 2))
  end)
end)

# =============================================
# `trees`, the fourth answer shape — and what it costs
# =============================================
#
# ML2 found this shape by being refused: the server answers `trees` on `operator_info` with the
# sub-models' cache ids joined by commas, not on `param`. Two things about it can only be
# measured:
#
#   * whether the split-and-wrap actually yields usable models, and
#   * how many cache entries it just spent. `MLHandler` registers **every** sub-model on the way
#     out, so a fetch of `trees` is not a read — it is an allocation, and `cache_info/1` before
#     and after is what proves it.

IO.puts("\ntrees, and what fetching them costs")

rf = Latu.ML.Classification.random_forest_classifier(num_trees: 3, max_depth: 2)

case ML.fit(rf, frames.features) do
  {:ok, forest} ->
    before = length(ML.cache_info!(session))
    held = fn -> length(ML.cache_info!(session)) end

    # **Once.** An earlier version of this section asked twice — through the accessor and then
    # through `attribute/2` — and leaked the first three trees, which is how we learned the
    # next line's finding the hard way.
    answer = Latu.ML.Classification.RandomForestClassificationModel.trees(forest)

    argued.("RandomForestClassificationModel.trees/1", fn -> answer end)

    case answer do
      {:ok, trees} ->
        IO.puts("  " <> String.pad_trailing("how many came back", 40) <> "#{length(trees)}")

        after_one = held.()

        IO.puts(
          "  " <>
            String.pad_trailing("cache entries, before and after", 40) <>
            "#{before} -> #{after_one}"
        )

        # The server does not dedupe: `MLHandler` registers the sub-models on every `Fetch`, so
        # asking twice costs twice. Worth measuring rather than assuming, because it decides
        # whether `trees` is something a caller may call in a loop (it is not).
        {:ok, again} = ML.attribute(forest, "trees")

        IO.puts(
          "  " <>
            String.pad_trailing("asking a second time costs again", 40) <>
            "#{after_one} -> #{held.()}"
        )

        ML.delete(again)

        argued.("a tree answers its own attributes", fn ->
          ML.attribute(hd(trees), "depth")
        end)

        argued.("...and its class is the right one", fn ->
          {:ok, hd(trees).class |> String.split(".") |> List.last()}
        end)

        argued.("delete/1 takes the whole list", fn -> {:ok, ML.delete(trees)} end)

        IO.puts(
          "  " <>
            String.pad_trailing("cache entries, back to the forest", 40) <> "#{held.()}"
        )

      {:error, error} ->
        argued.("trees", fn -> {:error, error} end)
    end

    ML.delete(forest)

  {:error, error} ->
    argued.("random_forest_classifier, fit", fn -> {:error, error} end)
end

# =============================================
# The five `CONNECT_ML` classes, and the atoms they map to
# =============================================
#
# `Latu.ML.error_kind/1` matches on the **dotted** class, which is what the server sends and not
# what Spark's docs write. Three of the five can be provoked cheaply; the two overflow classes
# would need a model over a gigabyte, so they stay read-only until something needs them.

IO.puts("\nerror classes")

kind = fn label, fun ->
  detail =
    case fun.() do
      {:error, error} ->
        "#{inspect(ML.error_kind(error))} — #{inspect(error.error_class)}"

      other ->
        "no error: " <> inspect(other, limit: 2)
    end

  IO.puts("  " <> String.pad_trailing(label, 34) <> detail)
end

case ML.fit(Latu.ML.Classification.logistic_regression(max_iter: 2), frames.features) do
  {:ok, doomed} ->
    kind.("a name off the allowlist", fn -> ML.attribute(doomed, "notAThing") end)
    ML.delete(doomed)
    kind.("a model that has been deleted", fn -> ML.attribute(doomed, "intercept") end)

  {:error, error} ->
    kind.("setup", fn -> {:error, error} end)
end

# =============================================
# A summary the cache has dropped, and whether it comes back
# =============================================
#
# A summary is dropped rather than offloaded, and the fetch that finds it gone answers
# `CONNECT_ML.MODEL_SUMMARY_LOST`. `Latu.ML.attribute/2` is supposed to send `CreateSummary`
# with the frame the model was fitted on and ask again — on its own, without the caller
# knowing. Nothing above forces that, because nothing above waits.
#
# Off by default because it costs eighty seconds, and eighty seconds is the *floor*. Two facts
# from `MLCache.scala` and `Connect.scala` on 4.2.0 set it, and both are worth knowing before
# reading this or writing anything else that leans on the cache:
#
#   * **`offloadingTimeout` is in minutes** — `.timeConf(TimeUnit.MINUTES)`. `"1s"` does not
#     mean one second, it truncates to **0**, and `expireAfterAccess(0, MINUTES)` means nothing
#     is ever held in memory. A run that sets it low is not testing a fast cache; it is
#     testing no cache, where a model is reloaded from disk — *without its summary* — on every
#     single command, and no `CreateSummary` can ever outlive the next one.
#   * **`offloadingTimeout` and `maxInMemorySize` are read once**, when `MLCache.cachedModel` is
#     first built, and baked into a Guava cache as `expireAfterAccess` and `maximumWeight`.
#     Changing either mid-session does nothing at all. Only `maxModelSize` and `maxStorageSize`
#     are read per call, and it is those that ML1 measured when it recorded that the memory
#     controls are live — the finding is right for two of the four and wrong for the other two.
#
# So: one minute, the smallest value that is not degenerate, set on a fresh session before the
# cache exists, and then a wait longer than it.

if "--recovery" in System.argv() do
  IO.puts("\nsummary recovery — about 80 s, see the header")

  recovery = Latu.connect!(url)
  key = "spark.connect.session.connectML.mlCache.memoryControl.offloadingTimeout"

  say = fn label, result ->
    detail =
      case result do
        {:ok, value} -> "ok — " <> inspect(value, limit: 3, printable_limit: 40)
        {:error, error} -> "#{inspect(error.error_class)} — " <> ProbeML.first_line(error.message)
        other -> inspect(other)
      end

    IO.puts("  " <> String.pad_trailing(label, 30) <> detail)
  end

  with :ok <- Latu.set_conf(recovery, key, "1m"),
       frames = ProbeML.frames(recovery),
       lr = Latu.ML.Classification.logistic_regression(max_iter: 2),
       {:ok, model} <- ML.fit(lr, frames.features) do
    summary = ML.summary(model)

    say.("summary, straight away", ML.attribute(summary, "totalIterations"))

    IO.puts("  waiting out the minute...")
    Process.sleep(70_000)

    # ML1 measured this from the model's side: an offload costs a model nothing. Here it is
    # from the summary's — the same event costs the summary everything, because the model's
    # own save format does not carry one and `MLCache` writes it to a sibling path instead.
    say.("model, after the offload", ML.attribute(model, "intercept"))

    # An `ok` on the next line alone would prove nothing: a summary that was never dropped
    # answers the same as one the retry put back. So ask first *without* the retry — the same
    # `Fetch` the facade sends, straight down `Latu.Client` — and let that line say whether
    # there was anything to recover from.
    bare =
      with {:ok, execution} <-
             Latu.Client.execute_command(
               recovery,
               Latu.ML.Internal.fetch_plan(summary, "totalIterations")
             ) do
        Latu.ML.Result.param(execution)
      end

    say.("summary, no retry", bare)

    # The claim: one fetch, no help. It should fail inside, send `CreateSummary` with the frame
    # the struct carries, and answer.
    say.("summary, asked once more", ML.attribute(summary, "totalIterations"))

    ML.delete(model)
  else
    {:error, error} -> say.("setup", {:error, error})
  end

  Latu.disconnect(recovery, release: true)
else
  IO.puts("\n(summary recovery not run; pass --recovery, it takes about 80 s)")
end

# =============================================
# The helper object
# =============================================
#
# `ConnectHelper` is the one entry in the server's attribute allowlist that is an attribute of
# nothing: a singleton reached by a `Fetch` naming the literal id `______ML_CONNECT_HELPER______`.
# Ten of its eleven methods are reachable, in three shapes — a value, a DataFrame, and a model
# the call itself builds — and the eleventh (`handleOverwrite`) is the meta-algorithms'.
#
# What this asks that nothing else can: whether the server accepts the arguments PySpark's own
# **call sites** say each one takes. Those are the only place the types exist — a helper has no
# class and so no signature — so the extractor read them off `invoke_helper_attr` and
# `invoke_helper_relation` expressions, and this is what says it read them right.

IO.puts("\nthe helper object")

helper_say = fn label, result ->
  detail =
    case result do
      {:ok, %Latu.ML.Model{} = model} -> "ok — model #{String.slice(model.ref, 0, 8)}…"
      {:ok, value} -> "ok — " <> inspect(value, limit: 3, printable_limit: 60)
      {:error, error} -> "#{inspect(error.error_class)} — " <> ProbeML.first_line(error.message)
      other -> inspect(other)
    end

  IO.puts("  " <> String.pad_trailing(label, 38) <> detail)
end

# A frame's worth of a resolved relation is the answer for the five that are lazy: `schema/1`
# is a real round trip through the planner, and it does not decode a Vector this package
# cannot read.
resolved = fn frame ->
  case Latu.schema(frame) do
    {:ok, schema} -> {:ok, Enum.map(schema, & &1.name)}
    other -> other
  end
end

helper_say.("getDefaultOrUS", ML.helper(session, :stop_words_remover_get_default_or_us))

helper_say.(
  "loadDefaultStopWords",
  ML.helper(session, :stop_words_remover_load_default_stop_words, ["german"])
)

helper_say.(
  "chiSquareTest",
  resolved.(Latu.ML.Stat.chi_square_test(frames.features, :features, :label))
)

helper_say.("correlation", resolved.(Latu.ML.Stat.correlation(frames.features, :features)))

helper_say.(
  "kolmogorovSmirnovTest",
  resolved.(Latu.ML.Stat.kolmogorov_smirnov_test(frames.features, :x1, "norm", [0.0, 1.0]))
)

# The two operators. Their params ride the call as positional arguments, so this is also the
# only thing that exercises `Latu.ML.Internal.operator_values!/3` against a server.
edges =
  Latu.sql!(session, """
  SELECT CAST(src AS LONG) AS src, CAST(dst AS LONG) AS dst, CAST(w AS DOUBLE) AS weight
  FROM VALUES (0, 1, 1.0), (1, 2, 1.0), (3, 4, 1.0), (4, 5, 1.0)
  AS t(src, dst, w)
  """)

sequences =
  Latu.sql!(session, """
  SELECT array(array(CAST(a AS INT)), array(CAST(b AS INT))) AS sequence
  FROM VALUES (1, 2), (1, 2), (1, 3), (2, 3)
  AS t(a, b)
  """)

helper_say.(
  "assignClusters",
  resolved.(ML.assign_clusters(Latu.ML.Clustering.power_iteration_clustering(k: 2), edges))
)

helper_say.(
  "findFrequentSequentialPatterns",
  resolved.(
    ML.find_frequent_sequential_patterns(Latu.ML.FPM.prefix_span(min_support: 0.5), sequences)
  )
)

# The three that build a model. Each is a cache entry the caller owns, exactly as a fit's is,
# and the uid the client sent is the one the model carries.
built =
  for {label, result} <- [
        {"stringIndexerModelFromLabels",
         Latu.ML.Feature.StringIndexerModel.from_labels(session, ["a", "b"], input_col: :x)},
        {"stringIndexerModelFromLabelsArray",
         Latu.ML.Feature.StringIndexerModel.from_arrays_of_labels(
           session,
           [["a", "b"], ["c"]],
           input_cols: [:x, :y]
         )},
        {"countVectorizerModelFromVocabulary",
         Latu.ML.Feature.CountVectorizerModel.from_vocabulary(session, ["a", "b"],
           input_col: :words
         )}
      ] do
    helper_say.(label, result)

    with {:ok, model} <- result, do: model
  end

# One of them read back, because a model built from a list should answer with that list.
case built do
  [%Latu.ML.Model{} = indexer | _rest] ->
    helper_say.("...and the labels it was built from", ML.attribute(indexer, "labels"))

  _other ->
    :ok
end

ML.delete(Enum.filter(built, &is_struct(&1, Latu.ML.Model)))

# =============================================
# Save, load, and whether the two clients can read each other
# =============================================
#
# ML4's claim. The path is the **server's**: a write happens where the session is, so these are
# the driver container's own `/tmp` and nothing on this machine has to exist.
#
# The interop half needs both clients and one shared directory, so it is a dance rather than a
# run — `dev/README.md` has the three steps. Each side fits the **same** model on the **same**
# rows with the **same** params, saves it, and then reads the other's and compares
# coefficients. Nothing is passed between the two but the saved model itself: both fits happen
# on one server against one plan, so agreeing to the last bit is a real expectation and not a
# tolerance dressed up as one.
#
# What goes red here is worth reading twice: a coefficient that differs is a *format* being
# read differently by two clients, which is the only thing this section is about.

IO.puts("\nsave and load")

interop_root = "/tmp/latu_ml_interop"

say = fn label, detail -> IO.puts("  " <> String.pad_trailing(label, 34) <> detail) end

outcome = fn
  {:ok, value} ->
    "ok — " <> inspect(value, limit: 4, printable_limit: 90)

  # A refusal this package made itself carries no `error_class` — that field is Spark's — so
  # its `kind` is what says where it came from.
  {:error, %{error_class: nil, kind: kind} = error} ->
    "#{inspect(kind)} — " <> ProbeML.first_line(error.message)

  {:error, error} ->
    "#{inspect(error.error_class)} — " <> ProbeML.first_line(error.message)

  other ->
    inspect(other)
end

coefficients = fn model ->
  case ML.attribute(model, :coefficients) do
    {:ok, tensor} -> {:ok, Nx.to_flat_list(tensor)}
    other -> other
  end
end

# The same estimator and the same rows the oracle uses. `frames.features` assembles [x1, x2]
# from the base frame's first four rows, which is what `--interop` builds on the Python side.
interop_lr = Latu.ML.Classification.logistic_regression(max_iter: 5, reg_param: 0.01)

case ML.fit(interop_lr, frames.features) do
  {:ok, ours} ->
    {:ok, mine} = coefficients.(ours)
    say.("fitted here", inspect(mine))

    saved = interop_root <> "/latu"
    say.("saved for PySpark", outcome.(ML.save(ours, saved, overwrite: true)))

    # The round trip that needs no second client: a load is a *new* cache entry, and what it
    # holds has to score like what was written.
    case ML.load(session, :logistic_regression_model, saved) do
      {:ok, back} ->
        say.("loaded back", outcome.(coefficients.(back)))
        say.("uid survived the trip", inspect(back.uid == ours.uid))

        # A saved model does not carry its summary, and a loaded one has no training frame to
        # rebuild one from. Both halves of that are meant to be visible rather than silent.
        lost = ML.attribute(ML.summary(back), "totalIterations")
        say.("and its summary did not", outcome.(lost))

        ML.delete(back)

      {:error, error} ->
        say.("loaded back", outcome.({:error, error}))
    end

    # The other half. Nothing here stats the server's filesystem — the load is the test for
    # whether the oracle has run, and its refusal says so plainly enough.
    if "--interop" in System.argv() do
      case ML.load(session, :logistic_regression_model, interop_root <> "/pyspark") do
        {:ok, theirs} ->
          matched =
            case coefficients.(theirs) do
              {:ok, ^mine} -> "identical"
              {:ok, other} -> "DIFFER — PySpark's #{inspect(other)}, ours #{inspect(mine)}"
              other -> outcome.(other)
            end

          say.("PySpark's model, read here", matched)
          ML.delete(theirs)

        {:error, error} ->
          say.("PySpark's model, read here", outcome.({:error, error}))
          IO.puts("      (run `python dev/pyspark_oracle.py --interop` first)")
      end
    else
      IO.puts("  (PySpark's half not read; pass --interop, see dev/README.md)")
    end

    ML.delete(ours)

  {:error, error} ->
    say.("setup", outcome.({:error, error}))
end


IO.puts("\npipelines")

# ML5a's claim, and the only place it can be made. A pipeline has no wire form: the fit is a
# fold this client runs and the save is a directory this client lays out around the stage
# writes the server does. So there is no golden file to diff and nothing offline can settle
# it — whether the layout is Spark's own is a question only a Spark that reads it can answer.
#
# The fold is checked here too, because the same objection applies: `check_offline.exs` knows
# the shape of a plan, not whether folding one frame through three stages lands where fitting
# them by hand would.

pipe = fn ->
  ML.pipeline([
    Feature.vector_assembler(input_cols: [:x1, :x2], output_col: :features),
    Latu.ML.Classification.logistic_regression(max_iter: 5, reg_param: 0.01)
  ])
end

# The last stage of a fitted pipeline is the model the fold produced. Its coefficients are what
# every comparison below is about: they are the one number that could only be right if the
# stages went through in order and came back in order.
last_coefficients = fn %Latu.ML.PipelineModel{stages: stages} ->
  coefficients.(List.last(stages))
end

case ML.fit(pipe.(), frames.base) do
  {:ok, fitted} ->
    {:ok, folded} = last_coefficients.(fitted)
    say.("fitted through the fold", inspect(folded))

    # The fold has to land exactly where fitting the same stages by hand would. Nothing about
    # a pipeline is meant to be its own numerical path, and this is what says so.
    {:ok, hand_fitted} = ML.fit(interop_lr, frames.features)
    {:ok, by_hand} = coefficients.(hand_fitted)
    ML.delete(hand_fitted)

    say.(
      "and by hand",
      if(by_hand == folded, do: "identical", else: "DIFFER — #{inspect(by_hand)}")
    )

    saved = interop_root <> "/latu_pipeline"
    say.("saved for PySpark", outcome.(ML.save(fitted, saved, overwrite: true)))

    case ML.load(session, :pipeline_model, saved) do
      {:ok, back} ->
        say.("loaded back", outcome.(last_coefficients.(back)))
        say.("uid survived the trip", inspect(back.uid == fitted.uid))
        same_uids = Enum.map(back.stages, & &1.uid) == Enum.map(fitted.stages, & &1.uid)
        say.("and so did the stage uids", inspect(same_uids))
        ML.delete(back)

      {:error, error} ->
        say.("loaded back", outcome.({:error, error}))
    end

    # An unfitted pipeline is metadata and nothing else: no stage holds a server reference, so
    # this is the one save that is entirely this client's own work.
    unfitted_path = interop_root <> "/latu_pipeline_unfitted"
    written = ML.save(pipe.(), unfitted_path, session: session, overwrite: true)
    say.("an unfitted pipeline saves", outcome.(written))

    case ML.load(session, :pipeline, unfitted_path) do
      {:ok, %Latu.ML.Pipeline{stages: stages}} ->
        say.("and loads as its stages", inspect(Enum.map(stages, & &1.class)))

      other ->
        say.("and loads as its stages", outcome.(other))
    end

    if "--interop" in System.argv() do
      case ML.load(session, :pipeline_model, interop_root <> "/pyspark_pipeline") do
        {:ok, theirs} ->
          matched =
            case last_coefficients.(theirs) do
              {:ok, ^folded} -> "identical"
              {:ok, other} -> "DIFFER — PySpark's #{inspect(other)}, ours #{inspect(folded)}"
              other -> outcome.(other)
            end

          say.("PySpark's pipeline, read here", matched)
          ML.delete(theirs)

        {:error, error} ->
          say.("PySpark's pipeline, read here", outcome.({:error, error}))
          IO.puts("      (run `python dev/pyspark_oracle.py --interop` first)")
      end
    else
      IO.puts("  (PySpark's half not read; pass --interop, see dev/README.md)")
    end

    ML.delete(fitted)

  {:error, error} ->
    say.("setup", outcome.({:error, error}))
end


IO.puts("\ntuning")

# ML5b's claim. A search has no wire form: the fold cut, the grid, the mean and the argmax are
# all this client's own, and the only thing that can check them is a Spark that runs the same
# search. `dev/pyspark_oracle.py --interop` runs it in PySpark against this same server and
# prints the same four numbers, which is the comparison this section exists for.
#
# Everything here is deterministic given the seed, so a difference is a real difference.

# **A discriminating fixture, on purpose.** The first version of this section searched a
# logistic regression on the four-row frame and every param map scored 0.75 — which means the
# comparison would have passed with the grid never applied at all, since three ties make index
# 0 the answer either way. So: twelve rows, a linear relationship, and a `reg_param` range wide
# enough that shrinkage really costs something. `rmse` is also the **minimised** branch of
# `larger_better?`, which the default evaluator would never have exercised.
tuning_rows =
  Latu.sql!(session, """
  SELECT CAST(y AS DOUBLE) AS label, CAST(x1 AS DOUBLE) AS x1, CAST(x2 AS DOUBLE) AS x2
  FROM VALUES
    (1.0, 0.0, 0.1), (2.1, 1.0, 0.2), (2.9, 2.0, 0.1), (4.2, 3.0, 0.3),
    (5.1, 4.0, 0.2), (5.8, 5.0, 0.4), (7.2, 6.0, 0.1), (8.1, 7.0, 0.3),
    (8.8, 8.0, 0.2), (10.3, 9.0, 0.5), (11.1, 10.0, 0.1), (11.9, 11.0, 0.4)
  AS t(y, x1, x2)
  """)

tuning_frame =
  ML.transform(
    Feature.vector_assembler(input_cols: [:x1, :x2], output_col: :features),
    tuning_rows
  )

tuning_lr = Latu.ML.Regression.linear_regression(max_iter: 5)
tuning_grid = ML.param_grid(tuning_lr, reg_param: [0.5, 0.0, 5.0])
tuning_eval = Latu.ML.Evaluation.regression_evaluator()

say.("grid size", inspect(length(tuning_grid)))
say.("larger_better? (rmse: false)", inspect(Latu.ML.Internal.larger_better?(tuning_eval)))

cv =
  ML.cross_validator(
    estimator: tuning_lr,
    param_maps: tuning_grid,
    evaluator: tuning_eval,
    num_folds: 2,
    seed: 42
  )

before_search = length(ML.cache_info!(session))

cached = fn model ->
  Enum.filter([model.best_model | List.flatten(model.sub_models || [])], & &1)
end

# Say whether the number is the right one rather than printing it and hoping. A leak here is a
# small integer among other small integers, which is exactly the kind of wrong that survives a
# read-through — one did.
entries = fn label, expected ->
  held = length(ML.cache_info!(session)) - before_search
  verdict = if held == expected, do: "ok — #{held}", else: "LEAK — #{held}, wanted #{expected}"
  say.(label, verdict)
end

case ML.fit(cv, tuning_frame) do
  {:ok, found} ->
    say.("avg metrics, one per param map", inspect(found.avg_metrics))
    say.("std metrics", inspect(found.std_metrics))
    say.("best index", inspect(ML.best_index(found)))
    say.("and the reg_param that won", inspect(Enum.at(tuning_grid, ML.best_index(found))))

    # If these are all equal the section proves nothing: three ties make index 0 the answer
    # whether or not the grid was ever applied. Said out loud so a degenerate fixture cannot
    # sit here looking green.
    say.(
      "the grid actually discriminates",
      inspect(length(Enum.uniq(found.avg_metrics)) == length(found.avg_metrics))
    )

    # The grid is ordered so the winner is neither first nor last. A best index of 0 is what a
    # broken argmin returns *and* what "take the first" returns, so 1 is the answer that says
    # the comparison actually happened.
    say.("the winner is in the middle", inspect(ML.best_index(found) == 1))

    # The lifecycle claim: `num_folds * grid` models were fitted and every one of them was
    # given back as its metric was read. What is left is the winner and nothing else.
    entries.("cache entries held", 1)

    say.("the winner scores rows", outcome.(coefficients.(found.best_model)))

    # ML5c. A search's directory is laid out entirely by this client — metadata, the grid, the
    # two operators it is made of, the winner — so nothing offline can say whether it is
    # Spark's own shape. Only a Spark that reads it can, which is `--interop`.
    saved_cv = interop_root <> "/latu_cv"
    say.("saved for PySpark", outcome.(ML.save(found, saved_cv, overwrite: true)))

    case ML.load(session, :cross_validator_model, saved_cv) do
      {:ok, back} ->
        say.("loaded back", inspect(back.avg_metrics == found.avg_metrics))
        say.("uid survived the trip", inspect(back.uid == found.uid))
        say.("std metrics too", inspect(back.std_metrics == found.std_metrics))

        # The grid is the part with an encoding of its own, and the part a wrong `parent`
        # would break silently: the search would load and simply have nothing to vary.
        say.("and the grid came back", inspect(back.validator.param_maps == tuning_grid))
        knobs = {back.validator.num_folds, back.validator.seed}
        say.("with its num_folds and seed", inspect(knobs))
        say.("the loaded winner scores the same", outcome.(coefficients.(back.best_model)))
        say.("and the best index agrees", inspect(ML.best_index(back) == ML.best_index(found)))

        # Sub-models were not collected here, so asking for them is a refusal rather than an
        # empty list — the directory has none and saying so beats handing back nil.
        say.(
          "sub-models it has none of",
          outcome.(ML.load(session, :cross_validator_model, saved_cv, sub_models: true))
        )

        ML.delete(back)

      {:error, error} ->
        say.("loaded back", outcome.({:error, error}))
    end

    ML.delete(found)
    entries.("and delete/1 gives it back", 0)

  {:error, error} ->
    say.("cross_validator", outcome.({:error, error}))
end

# Collected sub-models are the caller's, and `delete/1` has to reach all of them — this is the
# one path where a leak would be six entries rather than one.
collecting = %{cv | collect_sub_models: true}

case ML.fit(collecting, tuning_frame) do
  {:ok, found} ->
    say.("collected, folds by grid", inspect(Enum.map(found.sub_models || [], &length/1)))
    entries.("cache entries held", 7)

    # With sub-models, and read back both ways: the default leaves them on disk, which is the
    # one place this package deliberately does not do what PySpark does.
    kept_path = interop_root <> "/latu_cv_subs"
    say.("saved with sub-models", outcome.(ML.save(found, kept_path, overwrite: true)))

    case ML.load(session, :cross_validator_model, kept_path) do
      {:ok, lean} ->
        say.("a plain load leaves them on disk", inspect(lean.sub_models))
        say.("costing one entry, not seven", inspect(length(cached.(lean))))
        ML.delete(lean)

      other ->
        say.("a plain load", outcome.(other))
    end

    case ML.load(session, :cross_validator_model, kept_path, sub_models: true) do
      {:ok, whole} ->
        say.("and asking gets them", inspect(Enum.map(whole.sub_models || [], &length/1)))
        say.("costing seven", inspect(length(cached.(whole))))
        ML.delete(whole)

      other ->
        say.("asking for them", outcome.(other))
    end
    ML.delete(found)
    entries.("and delete/1 reaches all of them", 0)

  {:error, error} ->
    say.("collect_sub_models", outcome.({:error, error}))
end

tvs =
  ML.train_validation_split(
    estimator: tuning_lr,
    param_maps: tuning_grid,
    evaluator: tuning_eval,
    train_ratio: 0.75,
    seed: 42
  )

unfitted_path = interop_root <> "/latu_cv_unfitted"
written = ML.save(cv, unfitted_path, session: session, overwrite: true)
say.("an unfitted search saves", outcome.(written))

case ML.load(session, :cross_validator, unfitted_path) do
  {:ok, back} ->
    say.("and loads with its grid", inspect(back.param_maps == tuning_grid))
    say.("its estimator", inspect(back.estimator.class == tuning_lr.class))
    say.("and its evaluator", inspect(back.evaluator.class == tuning_eval.class))

  other ->
    say.("an unfitted search loads", outcome.(other))
end

case ML.fit(tvs, tuning_frame) do
  {:ok, found} ->
    say.("split metrics, one per param map", inspect(found.validation_metrics))
    say.("best index", inspect(ML.best_index(found)))
    ML.delete(found)

  {:error, error} ->
    say.("train_validation_split", outcome.({:error, error}))
end

# The other half of the dance. A search PySpark wrote, read here: the metrics are its own
# arithmetic and need not match ours, so what is compared is that the directory is legible at
# all and that the grid inside it survived both encodings.
if "--interop" in System.argv() do
  case ML.load(session, :cross_validator_model, interop_root <> "/pyspark_cv") do
    {:ok, theirs} ->
      say.("PySpark's search, read here", inspect(theirs.avg_metrics))

      readable =
        Enum.map(theirs.validator.param_maps, fn map ->
          Enum.map(map, fn {_uid, name, value} -> {name, value} end)
        end)

      say.("and its grid came through", inspect(readable))
      say.("with the estimator it searched", inspect(theirs.validator.estimator.class))
      ML.delete(theirs)

    {:error, error} ->
      say.("PySpark's search, read here", outcome.({:error, error}))
      IO.puts("      (run `python dev/pyspark_oracle.py --interop` first)")
  end
else
  IO.puts("  (PySpark's search not read; pass --interop, see dev/README.md)")
end

# A grid whose params reach nothing is the failure that otherwise completes and reports a
# winner, so the refusal is worth seeing rather than trusting.
stray =
  ML.cross_validator(
    estimator: Latu.ML.Feature.vector_assembler(input_cols: [:x1], output_col: :f),
    param_maps: tuning_grid,
    evaluator: tuning_eval
  )

say.(
  "a grid reaching nothing refused",
  try do
    ML.fit(stray, tuning_frame)
    "NOT REFUSED"
  rescue
    error in ArgumentError -> "ok — " <> ProbeML.first_line(error.message)
  end
)

if write? do
  path = Path.expand("../priv/ml_probe.exs", __DIR__)
  File.write!(path, ProbeML.render(results, spark))
  IO.puts("\nwrote #{path} — run `mix compile` to pick it up")
else
  IO.puts("\n(nothing written; pass --write to rewrite priv/ml_probe.exs)")
end

Latu.disconnect(session, release: true)
