# Probe: how a vector computed on the BEAM gets back to Spark as a `features` column.
#
#     docker compose up -d spark-connect
#     mix run dev/probe_embeddings.exs
#
# The cookbook wants one recipe for "embed on the BEAM, cluster on the cluster", and the step
# that decides its shape is the way *up*. Reading Spark's source says the obvious route cannot
# work: Polars maps every list to Arrow's `LargeList`, and `ArrowUtils.fromArrowField` has an
# arm for `List` and `ListView` and none for `LargeList` — it does have `LargeUtf8`, which is
# why string columns go up fine. If that reading holds, `create_dataframe/3` cannot carry an
# `array<double>` and the recipe is Parquet. **The server settles it; this asks.**
#
# Not a test. Prints a table, leaves one file in the container's /tmp, deletes what it fits.

alias Latu.ML
alias Latu.ML.Clustering.KMeansModel
alias Latu.ML.{Clustering, Functions}

url = System.get_env("LATU_URL", "sc://localhost:15002")
session = Latu.connect!(url)

# Four rows, two obvious clusters, four dimensions. Standing in for what `Nx.Serving` hands
# back from a Bumblebee embedding model: rows of equal-width floats, and nothing else.
ids = [1, 2, 3, 4]

embeddings = [
  [0.10, 0.11, 0.90, 0.91],
  [0.12, 0.09, 0.88, 0.93],
  [0.91, 0.89, 0.10, 0.12],
  [0.88, 0.92, 0.11, 0.09]
]

# Every route answers `{:ok, term}` or `{:error, the server's own words}`, trimmed so the table
# stays readable. The words matter: a guide fence asserts on a substring of the real message.
brief = fn
  {:ok, value} -> {:ok, value}
  {:error, %Latu.Error{} = e} -> {:error, e.message |> String.replace("\n", " ") |> String.slice(0, 300)}
end

types = fn
  {:ok, df} -> brief.(Latu.dtypes(df))
  {:error, _} = error -> error
end

# =============================================
# The way up
# =============================================

# A. A local relation with a list column, schema inferred. `create_dataframe/3` itself answers
#    happily: the server reads the Arrow schema at the first action, so `dtypes` is the refusal.
route_a = types.(brief.(Latu.create_dataframe(session, id: ids, embedding: embeddings)))

# B. The same with the type spelled out. The schema is a cast applied *after* the server reads
#    the Arrow schema, so it should not save it — worth knowing rather than assuming, because
#    if it does save it, the whole recipe is one line.
route_b =
  types.(
    brief.(
      Latu.create_dataframe(session, [id: ids, embedding: embeddings],
        schema: "id BIGINT, embedding ARRAY<DOUBLE>"
      )
    )
  )

# C. Plan-only fallback: carry the vector as a string and rebuild it server-side. Ugly, but it
#    needs no filesystem and no conf, which is the reason to know whether it works at all.
joined = Enum.map(embeddings, &Enum.map_join(&1, ",", fn f -> Float.to_string(f) end))

route_c =
  with {:ok, strings} <- brief.(Latu.create_dataframe(session, id: ids, emb: joined)) do
    types.(
      brief.(
        Latu.sql(session, "SELECT id, CAST(split(emb, ',') AS ARRAY<DOUBLE>) AS embedding FROM s",
          views: [s: strings]
        )
      )
    )
  end

# D. Parquet through `copy_to_fs/3`. Parquet's LIST is Parquet's own, so Polars' 64-bit offsets
#    never reach Spark's Arrow reader. The route `copy_to_fs/3`'s own docstring shows.
allow = "spark.sql.artifact.copyFromLocalToFs.allowDestLocal"
modifiable = Latu.is_modifiable!(session, allow)
if modifiable, do: :ok = Latu.set_conf(session, allow, "true")

path = "/tmp/latu_ml_embeddings/#{System.unique_integer([:positive])}.parquet"
bytes = Explorer.DataFrame.dump_parquet!(Explorer.DataFrame.new(id: ids, embedding: embeddings))

route_d =
  case Latu.copy_to_fs(session, path, bytes) do
    :ok -> types.({:ok, Latu.read(session, format: "parquet", path: path)})
    {:error, _} = error -> brief.(error)
  end

# =============================================
# The way on: does a surviving route reach MLlib
# =============================================

route_e =
  case route_d do
    {:ok, _} ->
      features =
        session
        |> Latu.read(format: "parquet", path: path)
        |> Latu.select([:id, features: Functions.array_to_vector(:embedding)])

      with {:ok, model} <- ML.fit(Clustering.k_means(k: 2, seed: 1), features),
           {:ok, centres} <- KMeansModel.cluster_center_matrix(model),
           scored = model |> ML.transform(features) |> Latu.select([:id, :prediction]),
           {:ok, rows} <- Latu.collect(scored) do
        :ok = ML.delete(model)
        {:ok, "centres #{inspect(Nx.shape(centres))}, #{inspect(rows)}"}
      else
        {:error, _} = error -> brief.(error)
      end

    _ ->
      {:error, "route D did not produce a frame"}
  end

# =============================================
# Report
# =============================================

IO.puts("\n#{url}  ·  #{allow} modifiable: #{modifiable}\n")

[
  {"A  create_dataframe, inferred", route_a},
  {"B  create_dataframe, ARRAY<DOUBLE>", route_b},
  {"C  string, split and cast", route_c},
  {"D  parquet, copy_to_fs", route_d},
  {"E  array_to_vector -> k_means", route_e}
]
|> Enum.each(fn
  {route, {:ok, value}} -> IO.puts("  ok    #{route}\n          #{inspect(value)}\n")
  {route, {:error, why}} -> IO.puts("  FAIL  #{route}\n          #{why}\n")
end)

{:ok, entries} = ML.cache_info(session)
IO.puts("cache on the way out: #{length(entries)} entries (want 0)\n")

IO.puts("""
What the cookbook needs from this:

  * whether A or B works, and if not, the server's exact words — a guide fence asserts on a
    substring of the real message, never on a guess
  * whether C is worth documenting as the small-data route or is a hack nobody should copy
  * that D produces `array<double>`, and that `array_to_vector` accepts it
  * whether the conf above is modifiable here, since the fence has to set it
""")

Latu.disconnect(session, release: true)
