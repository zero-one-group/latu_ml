# Deviations from PySpark

Every place `latu_ml`'s public API departs from `pyspark.ml`, and why. The wire is identical —
that is what `test/latu/ml/wire_test.exs` is for — so everything here is a surface choice.

Latu's own `docs/deviations.md` covers the DataFrame API this builds on.

| PySpark | Latu ML | Why |
|---|---|---|
| `LogisticRegression(maxIter=10)` | `logistic_regression(max_iter: 10)` | Elixir spelling, Spark's meaning. The wire name is the registry's job, not the caller's. |
| `featuresCol="features"` | `features_col: :features` | A column is an atom, as everywhere in Latu. Strings work too. |
| `model.coefficients` raises on failure | `attribute(model, :coefficients)` returns `{:ok, _}` or `{:error, %Latu.Error{}}` | Latu's action convention, with a `!` twin. A fetch is a round trip and can fail for reasons the caller did not cause. |
| `model.areaUnderROC` is the only spelling | `attribute(summary, :area_under_roc)` **or** `"areaUnderROC"` | The server's allowlist is camelCase and `Latu.ML.attributes/1` lists snake_case, so accepting only one would make the table a list of things you cannot ask for. A name already on the allowlist is taken as written; a name in neither spelling goes to the server unchanged, which is what keeps a class the registry does not know reachable. |
| A model is freed by `RemoteModelRef.__del__` | `with_model/3`, or `fit/2` and `delete/1` | BEAM terms have no finalizer, and faking one needs a process or a NIF resource. Latu's rule (M13.5) is to hand the resource out and let the caller say when it ends. |
| `DenseVector` → `numpy.ndarray` | `Nx.Tensor` | The ecosystem's currency. |
| `SparseVector` → `SparseVector` | `%Latu.ML.SparseVector{}`, never densified | Nx has no sparse tensor, and a vector that is sparse is usually sparse on purpose. `to_dense/1` is the caller's decision. |
| `DenseMatrix` keeps `isTransposed` | always row-major | A storage flag is not a shape. Column-major values are reshaped and transposed on the way in. |
| `deserialize_param` raises on an unknown UDT | `{:error, _}` naming the class | A refusal names the fix. |
| `model.summary` is a property that raises if there is none | `Latu.ML.summary/1`, which raises for a class that has no summary and names the ten that do | Ten of the 43 model classes record one. A function that only works for a quarter of its callers should say which quarter. |
| An empty `Matrix` is a `0x0` numpy array | `nil` | `Nx` has no empty tensor: a dimension must be a positive integer. `NaiveBayesModel.sigma` is a 0x0 Matrix on a multinomial model — sigma belongs to the gaussian one — so the answer is "there is nothing here", not a shape nothing can hold. |
| Params are validated by `TypeConverters` *and* `ParamValidators` client-side | kind refused client-side, range refused by the server | Spark's own validator has the better message, and duplicating it is a second source of truth. |
| `model.uid` comes from the model | copied from the estimator | Not a choice: a `Fit` answers with an `obj_ref` and nothing else — no uid, no params, no warning. PySpark's `_resetUid` does the same thing for the same reason. |
| `model.write().overwrite().option("k", "v").save(path)` | `Latu.ML.save(model, path, overwrite: true, options: %{"k" => "v"})` | A builder chain is a Python idiom for what a keyword list already says. One action, one call, and the same three fields on the wire. |
| An option key is lowercased by `MLWriter.option` | passed through as given | Spark says an option name is case-insensitive, so lowercasing is PySpark being helpful about something the server does not mind. This package does not rewrite what the caller wrote. |
| `LogisticRegressionModel.load(path)` | `Latu.ML.load(session, :logistic_regression_model, path)` | A `Read` names a class either way — PySpark's is the class object it is called on. A registry name, a generated model module and `{class, kind}` all spell it here, and the session has nowhere else to come from. |
| `evaluator.evaluate(dataset, params)` | `Latu.ML.evaluate(evaluator, frame)` | An evaluator is inert data, so a second evaluator with the other metric is a value rather than a call-site override — and a param map that only applies to one call is a second place for a param to live. |
| `ChiSquareTest.test`, `Correlation.corr`, `KolmogorovSmirnovTest.test` | `Latu.ML.Stat.chi_square_test/4`, `correlation/3`, `kolmogorov_smirnov_test/4` | Three classes with two `test`s between them do not flatten into one module. The helper method's own name is the tie-breaker, and it reads better than the class would. |
| `KolmogorovSmirnovTest.test(df, col, "norm", 0.0, 1.0)` | `kolmogorov_smirnov_test(df, col, "norm", [0.0, 1.0])` | A trailing list reads better in Elixir than a trailing splat, and the wire carries an array either way. |
| `StringIndexerModel.from_labels(labels, inputCol=...)` | `Latu.ML.Feature.StringIndexerModel.from_labels(session, labels, input_col: ...)` | Same placement, same params set on the model afterwards. The session is positional because nothing else carries one, as in `Latu.ML.load/3`. |
| `StopWordsRemover.loadDefaultStopWords(language)` | `Latu.ML.Feature.load_default_stop_words(session, language)` | `StopWordsRemover` is a transformer, so it has no accessor module to put this on; the group module is the closest home. |
| `pic.assignClusters(df)` | `Latu.ML.assign_clusters(pic, frame)` | Subject first, like every other verb here. `PowerIterationClustering` is `%Latu.ML.Helper{}` — an operator with params, no fit, and no uid, because a helper call sends no `MlOperator` for one to ride on. |
| `Bucketizer(splitsArray=[[-inf, 0.0, inf]])` | finite bounds only | Not a choice: the BEAM has no float infinity, so no Elixir client can send those. Splits only have to increase. |
| `Pipeline(stages=[...])`, `model.stages` | `Latu.ML.pipeline([...])`, `Latu.ML.fit/2` | Same shape, subject first. Neither client has a wire form for a pipeline — `MlCommand.Fit` takes an `MlOperator` and Spark exposes none for `Pipeline` — so both loop over the stages themselves. |
| `PipelineModel` is freed stage by stage | `Latu.ML.delete/1` takes the whole model | A pipeline model owns the models inside it, nested pipelines included, and one `Delete` carries every ref. The stages that hold nothing on the server are skipped rather than refused. |
| `ParamGridBuilder().addGrid(lr.regParam, [...]).build()` | `ML.param_grid(lr, reg_param: [...])` | A builder chain is a Python idiom for what a keyword list already says. Same product, same order — the last param varies fastest, as `itertools.product` does it. |
| `CrossValidator(parallelism=4)` | serial, no option yet | PySpark runs the grid on a thread pool. Whether several processes can drive one session's channel at once is unmeasured here, so the option is not offered rather than offered untested. `docs/decisions.md`. |
| A sub-model lives until the JVM collects it | deleted as its metric is read | The BEAM has no finalizer. `collect_sub_models: true` keeps them, and then `delete/1` on the returned model releases them with the winner. |
| `evaluator.isLargerBetter()` | `ML.larger_better?(evaluator)` | Pure here: the method is not on the server's allowlist, so PySpark overrides it client-side too and the registry extracts those overrides. |
| A validator checks its grid applies when it is **saved** | checked before the first fit | PySpark's check is in `_ValidatorSharedReadWrite.validateParams`, so a search over the wrong estimator completes, ties everywhere, and names a winner. A search you never save never finds out. |
| `CrossValidatorModel.load(path)` loads every persisted sub-model | `load/4` loads the winner; `sub_models: true` asks for the rest | Thirty `Read` commands and thirty cache entries on a load whose caller may only have wanted `best_model`, and no finalizer behind them here. The directory is unchanged, so a second call gets them. |
| `write().option("persistSubModels", "true")` | `save(model, path, sub_models: true)` | PySpark reads it from the writer's lowercased option map. Latu's `options:` means the options Spark's writer reads, and this package does not lowercase what a caller wrote — one keyword, one meaning. |
| `Summarizer.metrics("mean", "count").summary(col, weightCol)` | `Latu.ML.Stat.summary(col, [:mean, :count], weight: w)` | A builder chain is a Python idiom for what a list and a keyword already say. Metric names are atoms in Elixir spelling (`:num_nonzeros`, `:norm_l2`); the struct's fields keep Spark's, because the server names them. |
| `Summarizer.mean(col)` aliases the column `mean(features)` | `Latu.ML.Stat.mean(col)`, named where it is projected | Latu names an expression in `agg`'s keyword list, never on the expression. The wire below the alias is identical. |
| `vector_to_array(col, "float16")` reaches the server | refused here, naming both dtypes | Spark answers `INVALID_PARAMETER_VALUE.DTYPE`, which is a good error a round trip too late. The set is closed and two elements long, so it is checked where the typo is. Arity is still Spark's to refuse. |
| Saved metadata has six keys | a seventh, `latuMl` | Says which client wrote the directory. Both readers extract named keys and ignore the rest, so it costs nothing; nothing branches on it. |

## Not a deviation, but surprising

**A Vector column cannot be collected.** On Spark 4.2.0 the server describes `features`,
`rawPrediction` and `probability` as UDTs with no `sql_type`, so Latu's dtype guard refuses them
rather than guessing a layout. So is a Vector column you build yourself with
`Latu.ML.Functions.array_to_vector/1`: this is how Connect describes the type, not something a
fit does to a column.

Three routes through, and the destination picks one:

* **`Latu.to_nx/2`** (Latu 0.3.0) reads the Arrow bytes, where the type *is* stated, and gives
  one `{rows, width}` `f64` tensor. Use it when the destination is a tensor: measured at 100 MB
  of payload it costs about a twelfth of the BEAM memory of any row-by-row rebuild, and about
  half the wall time (`dev/probe_nx_copies.exs` in Latu).
* **`Latu.ML.Functions.vector_to_array/2`** converts server-side, so the column arrives as an
  ordinary `array<double>` any reader takes. It is PySpark's own answer and the right one when
  the destination is an `Explorer.DataFrame` — the *decode* to a frame is cheap; it is only
  carrying on to a tensor row by row that is not.
* **`select` or `drop` the column**, which is often the answer — a `prediction` column is a
  double and was never the problem. `Latu.to_arrow/2` bypasses the guard too, if you want the
  bytes.

`test/integration/vertical_slice_test.exs` pins the refusal, so the day Spark populates
`sql_type` this stops being true and the suite says so.

**A helper call sends every argument, every time.** `MlParams` carries only what the caller
set and the server fills the rest; the `ConnectHelper` methods take their arguments
**positionally**, so an unset param has to become something. It becomes the param's own
default, or — for `PowerIterationClustering.weightCol`, the one param Spark gives no default at
all — the empty string PySpark's own call site sends. Nothing here is invented: the fallback is
read off that call site along with the types.

**A loaded model has no training summary.** A model's save format does not carry one — `MLCache`
writes a summary to a sibling path when it offloads, and a `Read` does not go looking — so
asking a loaded model for one answers `CONNECT_ML.MODEL_SUMMARY_LOST`, exactly as an offloaded
model's would. The difference is what can be done about it: an offloaded summary is rebuilt from
the frame the model was fitted on, and a loaded model has no such frame and never did. So the
retry that makes an offload invisible does not fire here, and the server's refusal reaches the
caller with `Latu.ML.hint/1` naming both ways to arrive at it.

**A saved grid's value is a string.** `estimatorParamMaps` stores `0.5` as `"0.5"` — PySpark
writes `json.dumps(v)` into a field of a document that is itself JSON, and `json.loads` on the
way back. Writing the number instead produces a file PySpark will not load. Latu matches it,
because the point of the format is that PySpark reads it.

**A fitted search's metadata carries fewer params than an unfitted one's.** PySpark's reader
calls `getParam` for every key in `paramMap` and raises on one the instance lacks;
`CrossValidatorModel` has no `collectSubModels` where `CrossValidator` does. So the key is
written for a validator and omitted for its model. Nothing on this side would notice the
difference — only PySpark's reader does.

**A fold is cut with `rand(seed)` and a range, not `random_split`.** Spark adds one random
column above the data and each fold is a slice of it — `CrossValidator._kFold` and
`TrainValidationSplit` both, despite the name of the latter. Adding the column **once** is what
makes it work: a bare `rand()` inside a predicate is redrawn per evaluation, so train and
validation would overlap and the search would still produce a plausible answer. Latu does the
same, names the column `<validator uid>_rand` as PySpark does, and pins the arrangement in
`test/latu/ml_test.exs` — where the assertion is that both filters read the *same* projection.

With no `seed:` the draw differs between builds, so the search is not repeatable. Latu says
this for every seeded function; a grid search is where it matters most.

**A fold number in a column is trusted.** With `fold_col:`, PySpark asks the server to raise on
a number outside `0..num_folds-1` and refuses an empty fold — three extra actions per fold
before any fitting. Latu filters and lets the fit fail on its own rows, which costs a worse
message for a case the caller created deliberately.

**A saved pipeline is readable by PySpark and not by Scala.** A pipeline's directory is laid out
by the client, and Latu writes PySpark's layout — which means `metadata` says
`class: "pyspark.ml.pipeline.Pipeline"`. Scala's reader does
`require(className == expectedClassName)`, so `Pipeline.load` on the JVM refuses it. This is not
a Latu limitation being inherited quietly: **PySpark's own saved pipelines cannot be read by
Scala either**, for the same reason and the same one key.

What *is* portable: the `stages/NN_uid/` directories underneath are written by the server, so
each stage is in Scala's own format and loads anywhere. Only the wrapper is Python-flavoured.
The rest of the metadata — `uid`, `paramMap.stageUids`, the zero-padded stage directory names —
is byte-identical between the two, and PySpark 4.2's reader never reads `class` at all. So the
whole gap is one string, and `docs/decisions.md` has why it is the string it is and what
claiming the other one would cost.

A bare model is unaffected: its directory is written entirely by the server, so `Latu.ML.save/3`
on a model produces a directory any Spark client reads.

**A deleted model answers `CONNECT_ML.CACHE_INVALID`**, whose message calls the missing object a
"Summary object", blames a 15-minute idle eviction, and suggests `model.evaluate(dataset)` — all
three wrong for a model deleted a moment earlier. Match `error_class`, never the message. A ref
that never existed answers a bare `INTERNAL_ERROR` with no class at all, so the two cases differ.
