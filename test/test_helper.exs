# Integration tests need Latu's compose servers. `mix check` leaves them out; `mix check.all`
# passes `--include integration`.
#
# `grpc` logs every connection teardown at debug, and this suite opens and releases a session
# per test, so the default level buries the results. `:info` keeps everything this package
# itself logs — `Latu.ML.with_model/3`'s failed-delete warning above all.
Logger.configure(level: :info)

# Golden-plan helpers. Loaded here rather than compiled into the app; `mix.exs` keeps the
# directory out of the test loader so it is not mistaken for a test file.
Code.require_file("support/wire.exs", __DIR__)

ExUnit.start(exclude: [:integration])
