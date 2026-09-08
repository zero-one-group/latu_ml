# With Nx and Scholar

`latu_ml` trains on the cluster. This guide is about the other direction: getting numbers back
onto the BEAM, where `Nx`, `Scholar` and `defn` live.

Three things can cross, and they are not equally easy. **Features** come back as tensors.
**Fitted parameters** come back as tensors, and for some model families that is the whole model.
**The model itself** does not cross at all. Knowing which family you are in is most of what this
page is for.

Every fence below runs against a live server as part of the test suite.

## Setting up

```elixir
alias Latu.ML
alias Latu.ML.{Feature, Functions, Regression}

session = Latu.connect!("sc://localhost:15003")

training =
  Latu.sql!(session, """
  SELECT CAST(y AS DOUBLE) AS label, CAST(x1 AS DOUBLE) AS x1, CAST(x2 AS DOUBLE) AS x2
  FROM VALUES
    (7.0, 1.0, 1.0), (12.0, 2.0, 2.0), (17.0, 3.0, 3.0),
    (10.0, 1.0, 2.0), (14.0, 2.0, 3.0), (9.0, 3.0, 0.5)
  AS t(y, x1, x2)
  """)

assembler = Feature.vector_assembler(input_cols: [:x1, :x2], output_col: :features)
features = ML.transform(assembler, training)
```

## Features out

A `features` column is a `Vector`. That is the one thing in Spark that `Latu.collect/2` will not
give you:

```elixir
{:error, refused} = Latu.collect(features)
true = refused.message =~ "UDT that does not say its SQL type"
```

Spark describes the column as a UDT and declines to say what it serialises as. The decoder has
nothing to check it against, so it refuses rather than guess. `docs/deviations.md` has the whole
of it.

`Latu.to_nx/2` reads the Arrow bytes directly, where the type *is* stated, and hands back one
`{rows, width}` tensor:

```elixir
{:ok, %{"features" => rows}} = Latu.to_nx(features, columns: ["features"])

{6, 2} = Nx.shape(rows)
{:f, 64} = Nx.type(rows)
[1.0, 1.0, 2.0, 2.0] = rows |> Nx.slice([0, 0], [2, 2]) |> Nx.to_flat_list()
```

For a single batch this costs no copy: the Arrow buffer *is* the tensor's binary.

**When you want a DataFrame rather than a tensor**, go through Spark instead.
`Functions.vector_to_array/2` turns the Vector into an ordinary `array<double>` server-side, and
Explorer reads that as a list column:

```elixir
arrays = Latu.select(features, [:label, x: Functions.vector_to_array(:features)])

{:ok, frame} = Latu.to_explorer(arrays)
6 = Explorer.DataFrame.n_rows(frame)
```

`vector_to_array` is the right answer for Explorer and the wrong one for `Nx`.

## Parameters out

For the model families where the parameters *are* the model, the fit happens on the cluster and
the scoring happens here. A linear model is the clearest case: a dot product and an intercept.

```elixir
{:ok, model} = ML.fit(Regression.linear_regression(max_iter: 20), features)

{:ok, coefficients} = ML.attribute(model, :coefficients)
{:ok, intercept} = ML.attribute(model, :intercept)

{2} = Nx.shape(coefficients)
{:f, 64} = Nx.type(coefficients)
true = is_float(intercept)
```

### The one thing to get right

`Nx.tensor(2.0)` is **f32**. A bare Elixir float entering `defn` becomes an f32 tensor, and an
f32 intercept costs about 1e-7 of precision. Enough that your predictions and Spark's quietly
disagree in the eighth decimal. `Latu.ML.Linalg` builds the coefficients as f64, so the vector is
already right and only the scalar bites:

```elixir
intercept = Nx.tensor(intercept, type: :f64)
{:f, 64} = Nx.type(intercept)
```

### Scoring

```elixir
defmodule Linear do
  import Nx.Defn

  defn predict(features, coefficients, intercept) do
    Nx.dot(features, coefficients) + intercept
  end
end
```

Read the features and Spark's own predictions in **one** call, so nothing depends on two queries
agreeing about row order. `to_nx/2` hands back a tensor per column, of whatever shape each column
happens to have:

```elixir
scored = ML.transform(model, features)

{:ok, %{"features" => x, "prediction" => theirs}} =
  Latu.to_nx(scored, columns: ["features", "prediction"])

{6, 2} = Nx.shape(x)
{6} = Nx.shape(theirs)
```

Then score, and compare. Both sides used the same coefficients, so a mismatch here is arithmetic
and nothing else. Worth asserting rather than eyeballing:

```elixir
ours = Linear.predict(x, coefficients, intercept)

1 = Nx.to_number(Nx.all_close(ours, theirs, atol: 1.0e-10, rtol: 0.0))
```

The tensors are ordinary BEAM terms, so they outlive the handle they came from. Delete the model
and the scoring still works. That is the point of this seam:

```elixir
:ok = ML.delete(model)
{:error, gone} = ML.attribute(model, :coefficients)
"CONNECT_ML.CACHE_INVALID" = gone.error_class

unseen = Nx.tensor([[1.0, 1.0], [2.0, 2.0]], type: :f64)
{2} = unseen |> Linear.predict(coefficients, intercept) |> Nx.shape()
```

The same shape works for k-means centres, scaler means and variances, PCA components, and a
naive Bayes model's `theta` and `pi`. Anywhere the fitted state is a handful of tensors, you can
train on a cluster and predict in a GenServer.

## Where this stops working

It stops at models that are a *structure* rather than a set of parameters. Trees above all.

What the server's allowlist gives a fitted tree is its shape: `depth`, `num_nodes`,
`feature_importances`, and `to_debug_string`. **No node table, and no thresholds as data.** The
splits cross only inside `to_debug_string`, which is a text dump with no compatibility promise.

Scholar does not close the gap from its side either. It has no decision tree, random forest or
gradient-boosted tree at all. They do not express as `defn`, and Scholar's own README points at
EXGBoost instead.

So there are three honest answers for a tree, and none of them is "port the model":

* **Score on the cluster** with `ML.transform/2`. Not a workaround. Spark already holds the
  structure, and for a tree that structure is the model.
* **`ML.save/3` and read the artefact.** Spark's on-disk format carries the node table as
  Parquet, which Explorer reads. A real route, and a project rather than an afternoon.
* **Train an XGBoost4J booster** and load the saved booster with EXGBoost.

## Scholar, and when to reach for it instead

Scholar and `latu_ml` share their verbs, `fit` and `transform`, and disagree about what a model
*is*. Scholar's is a struct of tensors you can inspect, ship and pattern-match. This package's is
a reference into a server-side cache that offloads under memory pressure and is gone when the
session ends.

Choose Scholar when the training set fits one node. Choose this when it does not, or when the
features already live in a lakehouse and moving them would cost more than the fit. Trained here
and predicting there, the seam above is the road.

## What `to_nx/2` will not read

A tensor has one type and one shape. Anything without both is refused by name rather than guessed
at: nulls, strings, booleans, lists whose rows differ in length, and sparse `Vector`s. Arrow packs
booleans as a bitmap, not one byte per value.

```elixir
strings = Latu.sql!(session, "SELECT 'a' AS s")
{:error, why} = Latu.to_nx(strings)
true = why.message =~ "column s is a string column"
```

A sparse Vector is the one worth planning around. Densify it on the server before you read it:
there is no one width to reshape to.

```elixir
Latu.disconnect(session, release: true)
```
