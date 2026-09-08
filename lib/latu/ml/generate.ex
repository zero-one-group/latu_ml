defmodule Latu.ML.Generate do
  @moduledoc false
  # The code generator. Compile time only: nothing here survives into a call.
  #
  # Two macros, and between them they write nearly all of this package's public surface from
  # `Latu.ML.Registry`:
  #
  #   * `constructors/1` — one function per estimator, transformer and evaluator in a group,
  #     with the operator's own description, its param table and a status line. A model gets
  #     none: it exists only by `Latu.ML.fit/2`.
  #   * `accessors/1` — one module per fitted-model class in a group, with one function per
  #     name on the server's allowlist for it. Tab completion is that allowlist.
  #
  # Why generate rather than hand-write: 68 constructors carrying 991 params, and 623
  # accessors, is a transcription job nobody would notice going stale. Why *functions* rather
  # than a `Latu.ML.new(:logistic_regression, ...)` lookup: named functions have docs, specs
  # and tab completion, and discoverability in the REPL is the goal, not a nicety.

  alias Latu.ML.Registry

  # `toString` is on all 65 classes (every Spark operator is `Identifiable`), needs
  # `import Kernel, except: [to_string: 1]` in every generated module to define, and tells the
  # caller less than the uid the model already carries. It stays reachable through
  # `Latu.ML.attribute/2`, like everything else here.
  @skip ["toString"]

  @doc "One constructor per non-model operator in a PySpark group."
  defmacro constructors(group) do
    operators =
      for operator <- Registry.operators(),
          operator.group == group,
          operator.kind != :model do
        name = operator.name
        doc = constructor_doc(operator)
        returns = struct_type(operator.kind)

        quote do
          @doc unquote(doc)
          @spec unquote(name)(keyword()) :: unquote(returns)
          def unquote(name)(opts \\ []) do
            Latu.ML.Internal.build(unquote(name), opts)
          end
        end
      end

    {:__block__, [], operators}
  end

  @doc "One module of allowlisted accessors per fitted-model class in a PySpark group."
  defmacro accessors(group) do
    modules =
      for row <- Registry.classes(),
          row.group == group,
          functions = generated(row),
          built = Registry.helpers_making(row.class),
          functions != [] or built != [] do
        body =
          {:__block__, [],
           Enum.map(functions, &accessor(row, &1)) ++ Enum.map(built, &helper_constructor/1)}

        doc = accessor_module_doc(row, functions, built)
        class = row.class

        quote do
          defmodule unquote(row.module) do
            @moduledoc unquote(doc)

            @doc "The JVM class these accessors belong to."
            @spec class() :: String.t()
            def class, do: unquote(class)

            unquote(body)
          end
        end
      end

    {:__block__, [], modules}
  end

  # =============================================
  # What gets a function
  # =============================================

  # An accessor is generated for an attribute that answers with a value or a frame and whose
  # arguments — if it has any — the registry knows the types of. Three kinds are left out, and
  # none of them by taste:
  #
  #   * `:summary` answers with another cached object, and only `evaluate` is still that shape;
  #   * an attribute whose argument types nothing could read from PySpark, which `@reasons`
  #     calls `:untyped`;
  #   * `toString`, for the reason `@skip` gives.
  #
  # An attribute that takes arguments is generated at `1 + length(args)`, with each argument's
  # type read off `priv/ml_attributes.exs` rather than asked of the caller. One whose arity
  # nothing could measure, or whose types nothing could read, is not one to guess at — there
  # are none in 4.2.0, and the filter says so rather than assuming it.
  #
  # `summary` is the exception among the `:summary`-shaped ones and gets a function: the
  # reference is composed client-side rather than fetched, so there is nothing to send and
  # `Latu.ML.summary/1` is a builder. `evaluate` still waits, because its answer has no struct.
  defp generated(row) do
    Enum.filter(row.attributes, fn attribute ->
      arguments_known?(attribute) and attribute.wire not in @skip and
        (attribute.returns in [:value, :frame, :operators] or summary?(row, attribute))
    end)
  end

  defp arguments_known?(%{arity: 0}), do: true
  defp arguments_known?(%{args: args}) when is_list(args), do: args != []
  defp arguments_known?(_attribute), do: false

  defp summary?(row, %{wire: "summary"}), do: row.summary_dataset != nil
  defp summary?(_row, _attribute), do: false

  defp accessor(row, %{returns: :value} = attribute) do
    name = attribute.name
    holder = holder(row)
    vars = argument_vars(attribute)
    specs = argument_specs(attribute)

    quote do
      @doc unquote(attribute_doc(attribute))
      @spec unquote(name)(unquote(holder), unquote_splicing(specs)) ::
              {:ok, term()} | {:error, Latu.Error.t()}
      def unquote(name)(
            %{__struct__: unquote(holder_module(row))} = holder,
            unquote_splicing(vars)
          ) do
        Latu.ML.Internal.attribute(holder, unquote(row.class), unquote(attribute.wire), [
          unquote_splicing(vars)
        ])
      end
    end
  end

  defp accessor(row, %{wire: "summary"} = attribute) do
    quote do
      @doc unquote(attribute_doc(attribute))
      @spec summary(Latu.ML.Model.t()) :: Latu.ML.Summary.t()
      def summary(%Latu.ML.Model{} = model) do
        Latu.ML.Internal.summary(model, unquote(row.class))
      end
    end
  end

  # `:operators` answers with a list of *cache entries*, so the doc says so and the spec says
  # models. There are four of these, all `trees`.
  defp accessor(row, %{returns: :operators} = attribute) do
    name = attribute.name
    holder = holder(row)

    quote do
      @doc unquote(attribute_doc(attribute))
      @spec unquote(name)(unquote(holder)) ::
              {:ok, [Latu.ML.Model.t()]} | {:error, Latu.Error.t()}
      def unquote(name)(%{__struct__: unquote(holder_module(row))} = holder) do
        Latu.ML.Internal.attribute(holder, unquote(row.class), unquote(attribute.wire))
      end
    end
  end

  defp accessor(row, %{returns: :frame} = attribute) do
    name = attribute.name
    holder = holder(row)
    vars = argument_vars(attribute)
    specs = argument_specs(attribute)

    quote do
      @doc unquote(attribute_doc(attribute))
      @spec unquote(name)(unquote(holder), unquote_splicing(specs)) :: Latu.DataFrame.t()
      def unquote(name)(
            %{__struct__: unquote(holder_module(row))} = holder,
            unquote_splicing(vars)
          ) do
        Latu.ML.Internal.attribute_frame(holder, unquote(row.class), unquote(attribute.wire), [
          unquote_splicing(vars)
        ])
      end
    end
  end

  # =============================================
  # Arguments
  # =============================================

  defp argument_vars(%{args: args}) when is_list(args) do
    Enum.map(args, &Macro.var(&1.name, nil))
  end

  defp argument_vars(_attribute), do: []

  defp argument_specs(%{args: args}) when is_list(args), do: Enum.map(args, &spec_for(&1.type))
  defp argument_specs(_attribute), do: []

  # What a caller may pass, per the type `priv/ml_attributes.exs` recorded. Deliberately as wide
  # as `Latu.ML.Plan.literal/2` actually accepts — a `:double` argument takes `1` as readily as
  # `1.0`, and a `:string` one takes an atom — so the spec does not refuse what the code allows.
  defp spec_for(type) when type in [:vector, :matrix] do
    quote(do: Nx.Tensor.t() | Latu.ML.SparseVector.t())
  end

  defp spec_for(:frame), do: quote(do: Latu.DataFrame.t())
  defp spec_for(type) when type in [:int, :long], do: quote(do: integer())
  defp spec_for(type) when type in [:float, :double], do: quote(do: number())
  defp spec_for(:string), do: quote(do: String.t() | atom())
  defp spec_for(:boolean), do: quote(do: boolean())

  defp spec_for({:either, members}) do
    members
    |> Enum.map(&spec_for/1)
    |> Enum.reduce(fn spec, acc -> quote(do: unquote(acc) | unquote(spec)) end)
  end

  defp spec_for({:list, element}), do: quote(do: [unquote(spec_for(element))])

  # =============================================
  # Models the server builds out of what you send
  # =============================================

  # Three of the eleven `ConnectHelper` methods answer with a model reference, not a value:
  # a `StringIndexerModel` from a list of labels, another from arrays of them, a
  # `CountVectorizerModel` from a vocabulary. They belong on the model's own module, beside the
  # accessors, because that is the class they make — and they are generated rather than
  # hand-written for the plain reason that the module is.
  defp helper_constructor(helper) do
    name = helper_name!(helper)
    [_uid | arguments] = helper.args
    vars = Enum.map(arguments, &Macro.var(&1.name, nil))
    specs = Enum.map(arguments, &spec_for(&1.type))

    quote do
      @doc unquote(helper_doc(helper))
      @spec unquote(name)(Latu.Session.t(), unquote_splicing(specs), keyword()) ::
              {:ok, Latu.ML.Model.t()} | {:error, Latu.Error.t()}
      def unquote(name)(session, unquote_splicing(vars), opts \\ []) do
        Latu.ML.Internal.model_from_helper(
          unquote(helper.name),
          session,
          [unquote_splicing(vars)],
          opts
        )
      end
    end
  end

  # The Elixir name is PySpark's own, snaked by the extractor. A name that is not one this can
  # define — a dunder, or anything with a character a bare `def` will not take — stops the
  # compile rather than being transliterated by a rule nobody wrote down.
  defp helper_name!(helper) do
    unless to_string(helper.function) =~ ~r/^[a-z][a-z0-9_]*$/ do
      raise CompileError,
        description:
          "the helper #{helper.method} is #{helper.python} in PySpark, and #{helper.function} " <>
            "is not a name this can define. Decide one in Latu.ML.Generate."
    end

    helper.function
  end

  defp helper_doc(helper) do
    wrap(
      "#{helper.doc || "Build this model on the server out of what you pass."} " <>
        "The server builds it and registers it, so what comes back is a cache reference like " <>
        "a fit's and is yours to `Latu.ML.delete/1`. The uid is generated here and sent with " <>
        "the call. `opts` are params set on the model afterwards, checked against its own " <>
        "table.\n\n*Built by* `#{helper.method}` on the server's helper object, from " <>
        "`#{helper.python}`."
    )
  end

  defp holder_module(%{kind: :model}), do: Latu.ML.Model
  defp holder_module(%{kind: :summary}), do: Latu.ML.Summary

  defp holder(%{kind: :model}), do: quote(do: Latu.ML.Model.t())
  defp holder(%{kind: :summary}), do: quote(do: Latu.ML.Summary.t())

  defp struct_type(:estimator), do: quote(do: Latu.ML.Estimator.t())
  defp struct_type(:transformer), do: quote(do: Latu.ML.Transformer.t())
  defp struct_type(:evaluator), do: quote(do: Latu.ML.Evaluator.t())
  defp struct_type(:helper), do: quote(do: Latu.ML.Helper.t())

  # =============================================
  # Docs
  # =============================================

  defp constructor_doc(operator) do
    Enum.join(
      [
        wrap(operator.doc),
        wrap(verb(operator)),
        wrap(status_line(operator)),
        "## Params",
        param_list(operator.params),
        wrap("""
        Defaults are documented, never sent: a param the caller did not set and a param sent
        with its default value are different requests, and only the first is right. A value's
        **kind** is refused here; its **range** is Spark's own `ParamValidators` to refuse,
        with a better message than this package could write.
        """)
      ],
      "\n\n"
    ) <> "\n"
  end

  # Generated prose is written as one long string and folded here, so a `h Latu.ML.Feature.pca`
  # in `iex` reads as a paragraph rather than as whatever line breaks the generator's own source
  # happened to have. Bullet lists are left alone: a wrapped bullet needs a continuation indent,
  # and a param doc that runs long is more readable on one line than badly folded.
  @width 92

  defp wrap(text) do
    text
    |> String.split()
    |> Enum.reduce([[]], fn word, [line | rest] ->
      if line != [] and IO.iodata_length([line, " ", word]) > @width do
        [[word], line | rest]
      else
        [if(line == [], do: [word], else: [line, " ", word]) | rest]
      end
    end)
    |> Enum.reverse()
    |> Enum.map_join("\n", &IO.iodata_to_binary/1)
  end

  defp verb(%{kind: :estimator} = operator) do
    "`Latu.ML.fit/2` fits it, and hands back a `Latu.ML.Model` — a reference into the " <>
      "session's ML cache, not a value. `Latu.ML.with_model/3` releases it for you; " <>
      "`Latu.ML.delete/1` is the explicit form. Its attributes are on " <>
      "#{model_module(operator)}."
  end

  defp verb(%{kind: :transformer}) do
    "`Latu.ML.transform/2` applies it. That is a lazy builder, not an action: it hands back " <>
      "a `Latu.DataFrame` and reaches no server until you collect one."
  end

  # The two that are neither fitted nor applied. Their one method is on the server's helper
  # object, and the facade verb is named after it — `assignClusters` is `assign_clusters/2`.
  defp verb(%{kind: :helper} = operator) do
    case Registry.helpers_of(operator.python) do
      [helper] ->
        "`Latu.ML.#{helper.function}/2` is the one thing it does, and it is a lazy builder: " <>
          "it hands back a `Latu.DataFrame` and reaches no server until you collect one. " <>
          "There is no fit and nothing cached — its params ride that call as **positional " <>
          "arguments**, so every one is sent whether you set it or not, and the ones you " <>
          "leave alone are sent as the defaults below."

      _none_or_many ->
        "It is neither fitted nor applied, and no helper method in the registry says what it " <>
          "does instead."
    end
  end

  defp verb(%{kind: :evaluator}) do
    "An evaluator scores a frame and answers with one number. The verb that runs one is not " <>
      "written yet, so for now this builds the operator and its params — and the command " <>
      "behind it is already pinned against PySpark's own bytes."
  end

  # The status line, which is the whole point of `priv/ml_probe.exs`: what a live server did
  # with this operator, said on the operator's own doc page rather than in a table somewhere.
  defp status_line(%{status: :probed} = operator) do
    "*Status `:probed`* — `dev/probe_ml.exs` #{ran(operator)} it against a live Spark " <>
      "#{operator.spark} server, and every allowlisted attribute it could ask answered."
  end

  defp status_line(%{status: :missing} = operator) do
    "*Status `:missing`* — PySpark names this operator and the Spark #{operator.spark} server " <>
      "does not load it, so calling this will fail. #{sanitise(operator.reason)}"
  end

  defp status_line(%{status: :built, reason: nil} = operator) do
    "*Status `:built`* — generated from PySpark #{operator.spark}'s own param table, and not " <>
      "yet exercised against a live server by `dev/probe_ml.exs`."
  end

  defp status_line(%{status: :built} = operator) do
    "*Status `:built`* — generated from PySpark #{operator.spark}'s own param table. " <>
      "`dev/probe_ml.exs` has not got this one working against a live server, which is as " <>
      "likely to be the probe's four-row fixture as the operator: " <> sanitise(operator.reason)
  end

  defp ran(%{kind: :estimator}), do: "fitted"
  defp ran(_operator), do: "applied"

  # A reason is a server's own error text, and it lands in a Markdown `@doc`. Backticks in it
  # would unbalance the code spans around it, and ExDoc builds with `warnings_as_errors`.
  defp sanitise(nil), do: ""
  defp sanitise(reason), do: reason |> String.replace("`", "'") |> String.trim()

  defp model_module(operator) do
    case operator.model_class && Registry.class(operator.model_class) do
      nil -> "the model's own module"
      row -> "`#{inspect(row.module)}`"
    end
  end

  defp param_list([]), do: "This operator takes none."

  defp param_list(params) do
    Enum.map_join(params, "\n", fn param ->
      "  * `#{inspect(param.name)}` — #{sentence(param.doc)}#{default(param.default)}"
    end)
  end

  # PySpark's param docs are inconsistently punctuated, and a bullet list where half the items
  # end in a full stop and half do not reads as sloppy rather than as faithful.
  defp sentence(nil), do: "No description."

  defp sentence(text) do
    if String.ends_with?(text, [".", "!", "?", ")"]), do: text, else: text <> "."
  end

  # Four defaults have no literal form, and two of those cannot be stated at all: `:varies` is
  # a seed, which PySpark derives from a hash the server does not share, and `:server_supplied`
  # is a value only a running JVM knows. Saying nothing is the honest rendering — the section's
  # closing paragraph already says an unset param takes Spark's default.
  defp default(nil), do: ""
  defp default(:varies), do: ""
  defp default(:server_supplied), do: ""
  defp default(:nan), do: " Default `NaN`."
  defp default({:uid_suffix, suffix}), do: " Defaults to this operator's uid plus `#{suffix}`."
  defp default(value), do: " Default `#{inspect(value)}`."

  defp attribute_doc(%{returns: :value} = attribute) do
    wrap(described(attribute)) <>
      "\n\n" <>
      wrap(
        "An action: it reaches the server. A `Vector` or `Matrix` comes back as an " <>
          "`Nx.Tensor`, or a `Latu.ML.SparseVector` where densifying would be this package's " <>
          "decision rather than yours."
      ) <> arguments_doc(attribute) <> "\n"
  end

  defp attribute_doc(%{returns: :operators} = attribute) do
    short = attribute.element_class |> String.split(".") |> List.last()

    wrap(described(attribute)) <>
      "\n\n" <>
      wrap(
        "An action, and it **spends cache**: the server registers every sub-model on its way " <>
          "out, so this answers with a list of `Latu.ML.Model` — one `#{short}` and one cache " <>
          "entry per tree — and they are yours to give back. `Latu.ML.delete/1` takes the " <>
          "whole list in one call."
      ) <> "\n"
  end

  defp attribute_doc(%{returns: :summary} = attribute) do
    wrap(described(attribute)) <>
      "\n\n" <>
      wrap(
        "A lazy builder: Spark reaches a summary through the model that owns it, so the " <>
          "reference is composed here and nothing is sent. `Latu.ML.summary/1` is the same " <>
          "thing without the class check."
      ) <> "\n"
  end

  defp attribute_doc(%{returns: :frame} = attribute) do
    wrap(described(attribute)) <>
      "\n\n" <>
      wrap(
        "A lazy builder: the `Fetch` rides a relation, so this hands back a `Latu.DataFrame` " <>
          "and nothing has run until you collect it."
      ) <> arguments_doc(attribute) <> "\n"
  end

  # The names and types are PySpark's own annotations, by way of `priv/ml_attributes.exs`. Said
  # here because the spec alone does not say which argument is which — two `int`s in a row is
  # exactly the case where a reader needs the names.
  defp arguments_doc(%{args: args}) when is_list(args) and args != [] do
    "\n\n## Arguments\n\n" <>
      Enum.map_join(args, "\n", &"  * `#{&1.name}` — #{argument_kind(&1.type)}") <> "\n"
  end

  defp arguments_doc(_attribute), do: ""

  defp argument_kind(type) when type in [:vector, :matrix] do
    "a Spark `Vector`: an `Nx.Tensor` or a `Latu.ML.SparseVector`"
  end

  defp argument_kind(:frame), do: "a `Latu.DataFrame`, sent as a relation rather than a literal"
  defp argument_kind(type) when type in [:int, :long], do: "an integer"
  defp argument_kind(type) when type in [:float, :double], do: "a number, sent as a double"
  defp argument_kind(:string), do: "a string, or an atom"
  defp argument_kind(:boolean), do: "a boolean"

  # Short names inside a union, so the bullet stays a line: the spec above it carries the exact
  # types, and "either a string or a Spark `Vector`" is what a reader needs from the prose.
  defp argument_kind({:either, members}) do
    "either " <> Enum.map_join(members, " or ", &short_kind/1) <> " — the value decides which"
  end

  defp short_kind(type) when type in [:vector, :matrix], do: "a Spark `Vector`"
  defp short_kind(type) when type in [:int, :long], do: "an integer"
  defp short_kind(type) when type in [:float, :double], do: "a number"
  defp short_kind(:string), do: "a string"
  defp short_kind(:boolean), do: "a boolean"
  defp short_kind(:frame), do: "a `Latu.DataFrame`"

  # Five allowlisted names have no PySpark attribute to quote a description from — PySpark
  # wraps them under another name, or does not expose them at all — so say that rather than
  # print "No description." over a name the server does answer to.
  defp described(%{doc: nil} = attribute) do
    "Spark's `#{attribute.wire}`. PySpark exposes no attribute under that name, so there is " <>
      "no description of it to quote here."
  end

  defp described(attribute), do: sentence(attribute.doc)

  defp accessor_module_doc(row, functions, built) do
    Enum.join(
      [
        opening(row),
        wrap(
          "Every accessor here is one name on the server's allowlist for `#{row.class}` — " <>
            "#{length(functions)} of them — so tab completion is that allowlist. The server " <>
            "refuses anything else with `CONNECT_ML.ATTRIBUTE_NOT_ALLOWED`; these refuse a " <>
            "model of the wrong class before it gets that far, and name the module that " <>
            "would have taken it."
        ),
        wrap(elsewhere(row, functions)),
        wrap(constructors_note(built)),
        wrap(
          "The allowlist is inherited rather than per class: this one is the union of " <>
            Enum.map_join(row.from, ", ", &"`#{&1}`") <> "."
        )
      ]
      |> Enum.reject(&(&1 == "")),
      "\n\n"
    ) <> "\n"
  end

  # Two model classes can be had without a fit at all, from a list the caller already holds.
  defp constructors_note([]), do: ""

  defp constructors_note(built) do
    "This model does not have to be fitted: " <>
      Enum.map_join(built, ", ", &"`#{&1.function}/#{length(&1.args) + 1}`") <>
      " builds one on the server out of what you pass, through its own helper method. That is " <>
      "a cache entry like a fit's, and yours to give back."
  end

  # What the class is allowed but this module does not offer, said by reason rather than as one
  # undifferentiated list — `evaluate` is missing because its answer has no struct, and
  # `toString` because it is not worth a function, and those are not the same news.
  #
  # Each name lands in exactly one bucket, shape first: `evaluate` both takes a dataset and
  # answers with a summary, and the summary is the deeper reason — the arguments are sent fine.
  # `:operators` has no bucket any more: `trees` is generated.
  @reasons [
    {:skipped, "is `Identifiable`'s, and tells you less than the model's own uid",
     "are `Identifiable`'s, and tell you less than the model's own uid"},
    {:summary, "answers with a training summary, which has no struct yet",
     "answer with a training summary, which has no struct yet"},
    {:untyped, "takes arguments whose types nothing could read from PySpark",
     "take arguments whose types nothing could read from PySpark"}
  ]

  defp opening(%{kind: :model} = row), do: "Attributes of a fitted `#{row.python}`."

  defp opening(row) do
    "Attributes of a `#{row.python}` — what an estimator recorded while it fitted. " <>
      "`Latu.ML.summary/1` is what hands you one."
  end

  defp elsewhere(row, functions) do
    generated = MapSet.new(functions, & &1.wire)

    buckets =
      row.attributes
      |> Enum.reject(&MapSet.member?(generated, &1.wire))
      |> Enum.group_by(&bucket/1, & &1.wire)

    said =
      for {key, one, many} <- @reasons, names = Map.get(buckets, key, []), names != [] do
        Enum.map_join(names, ", ", &"`#{&1}`") <> " " <> plural(names, one, many)
      end

    case said do
      [] ->
        "Every allowlisted attribute of this class has a function here."

      reasons ->
        "Allowed by the server but not here: #{Enum.join(reasons, "; ")}. " <>
          "`Latu.ML.attribute/2` and `Latu.ML.attribute/3` will send any of these names; it is " <>
          "what comes back that has nowhere to go."
    end
  end

  defp bucket(%{wire: wire}) when wire in @skip, do: :skipped
  defp bucket(%{returns: :summary}), do: :summary
  defp bucket(_attribute), do: :untyped

  defp plural([_one], singular, _plural), do: singular
  defp plural(_many, _singular, plural), do: plural
end
