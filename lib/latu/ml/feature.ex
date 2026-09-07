defmodule Latu.ML.Feature do
  @moduledoc """
  Feature transformers, and the models the fitted ones produce.

  The half of MLlib that shapes data rather than predicting from it: assembling a `features`
  Vector, scaling it, indexing strings, hashing text. Most are transformers and reach no
  server when applied; the ones that must learn something first — a vocabulary, a mean, a set
  of quantiles — are estimators, and `Latu.ML.fit/2` is what turns those into models.

  Every constructor here is generated from PySpark 4.2.0's own param table, and every
  accessor module from the server's own attribute allowlist — see `Latu.ML.operators/1` for
  the table it comes from. The two functions below are the exception, and they are here for
  the reason PySpark puts them on `StopWordsRemover`: they answer questions about that
  transformer's defaults, which only a running server can answer.
  """

  require Latu.ML.Generate

  Latu.ML.Generate.constructors(:feature)

  @doc """
  The locale a `StopWordsRemover` uses when you do not name one.

  An action, and the reason `stop_words_remover/1`'s `locale` param has no default this
  package can print: it is the JVM's own default locale, or `en_US` where that one has no
  stop-word list. Only the server knows which.

      {:ok, locale} = Latu.ML.Feature.default_locale(session)
  """
  @spec default_locale(Latu.Session.t()) :: {:ok, String.t()} | {:error, Latu.Error.t()}
  def default_locale(session) do
    Latu.ML.helper(session, :stop_words_remover_get_default_or_us)
  end

  @doc "See `default_locale/1`. Raises instead of returning an error."
  @spec default_locale!(Latu.Session.t()) :: String.t()
  def default_locale!(session), do: unwrap!(default_locale(session))

  @doc """
  Spark's own stop-word list for a language.

  An action, and the other half of what `stop_words_remover/1` cannot default: the list ships
  with Spark, and `danish`, `dutch`, `english`, `finnish`, `french`, `german`, `hungarian`,
  `italian`, `norwegian`, `portuguese`, `russian`, `spanish`, `swedish` and `turkish` are the
  languages it has.

      {:ok, words} = Latu.ML.Feature.load_default_stop_words(session, "german")
      remover = Latu.ML.Feature.stop_words_remover(input_col: :words, stop_words: words)
  """
  @spec load_default_stop_words(Latu.Session.t(), String.t()) ::
          {:ok, [String.t()]} | {:error, Latu.Error.t()}
  def load_default_stop_words(session, language) when is_binary(language) do
    Latu.ML.helper(session, :stop_words_remover_load_default_stop_words, [language])
  end

  @doc "See `load_default_stop_words/2`. Raises instead of returning an error."
  @spec load_default_stop_words!(Latu.Session.t(), String.t()) :: [String.t()]
  def load_default_stop_words!(session, language) do
    unwrap!(load_default_stop_words(session, language))
  end

  defp unwrap!({:ok, value}), do: value
  defp unwrap!({:error, error}), do: raise(error)
end

require Latu.ML.Generate

Latu.ML.Generate.accessors(:feature)
