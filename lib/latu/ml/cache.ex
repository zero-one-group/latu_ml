defmodule Latu.ML.Cache do
  @moduledoc false
  # Who owns what the server's ML cache holds. A `Read` caches, so every error path below one
  # owns what it read; these are the folds that apply that rule and the release that pays for
  # it, shared by `Latu.ML` and `Latu.ML.Persistence`. `docs/decisions.md`, 2026-09-11.

  require Logger

  alias Latu.ML.CrossValidatorModel
  alias Latu.ML.Model
  alias Latu.ML.Pipeline
  alias Latu.ML.PipelineModel
  alias Latu.ML.TrainValidationSplitModel

  def cached_models(%PipelineModel{stages: stages}), do: Enum.flat_map(stages, &cached_models/1)

  # An *unfitted* pipeline holds cache entries too: `Latu.ML.Pipeline.stage/0` admits a model
  # already fitted, so a pipeline loaded as a search's estimator can arrive owning one.
  def cached_models(%Pipeline{stages: stages}), do: Enum.flat_map(stages, &cached_models/1)

  def cached_models(%CrossValidatorModel{} = model) do
    cached_models(model.best_model) ++ Enum.flat_map(model.sub_models || [], &cached_models/1)
  end

  def cached_models(%TrainValidationSplitModel{} = model) do
    cached_models(model.best_model) ++ Enum.flat_map(model.sub_models || [], &cached_models/1)
  end

  def cached_models(models) when is_list(models), do: Enum.flat_map(models, &cached_models/1)
  def cached_models(%Model{} = model), do: [model]
  def cached_models(_stage), do: []

  def collected(kept), do: kept |> List.flatten() |> Enum.flat_map(&cached_models/1)

  # Collect in order, or give back what was already cached and answer with the first error.
  # A `Read` caches, so every error path below one owns what it read. That decision lives here
  # rather than at each fold, because five copies of it is how the sixth came to forget.
  def traverse(items, fun) do
    items
    |> Enum.reduce_while({:ok, []}, fn item, {:ok, done} ->
      case fun.(item) do
        {:ok, value} -> {:cont, {:ok, [value | done]}}
        {:error, error} -> {:halt, {:error, error, done}}
      end
    end)
    |> case do
      {:ok, done} ->
        {:ok, Enum.reverse(done)}

      {:error, error, done} ->
        release(collected(done))
        {:error, error}
    end
  end

  # `:ok`, or the first error. Nothing is cached on this path, so nothing is given back.
  def each_ok(items, fun) do
    Enum.reduce_while(items, :ok, fn item, :ok ->
      case fun.(item) do
        :ok -> {:cont, :ok}
        {:error, error} -> {:halt, {:error, error}}
      end
    end)
  end

  def release(model) do
    case Latu.ML.delete(model) do
      :ok -> :ok
      {:error, error} -> Logger.warning(released(model) <> Exception.message(error))
    end
  end

  defp released(%PipelineModel{uid: uid}), do: "could not delete the models in #{uid}: "
  defp released(%CrossValidatorModel{uid: uid}), do: "could not delete the models in #{uid}: "

  defp released(%TrainValidationSplitModel{uid: uid}),
    do: "could not delete the models in #{uid}: "

  defp released(%Model{ref: ref}), do: "could not delete model #{ref}: "

  defp released(models) when is_list(models),
    do: "could not delete the #{length(models)} model(s) a failed fit had already cached: "
end
