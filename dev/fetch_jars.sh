#!/usr/bin/env bash
# The jars `dev/probe_ext.exs` uploads and `dev/docker-compose.ext.yml` mounts, from Maven
# Central into `tmp/jars` — gitignored, never committed, never in the Hex package.
#
#   xgboost4j-spark   3.4.0, built for Spark 3.5.3, Scala 2.13; shades xgboost4j and its native
#                     libraries (linux and macOS, x86_64 and aarch64) in
#   isolation-forest  4.1.8, built for Spark 4.1.1, Scala 2.13; pure JVM
#   spark-avro        4.2.0, Spark's own external module — isolation-forest's writer needs it,
#                     and the distribution does not ship it
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p tmp/jars

central=https://repo1.maven.org/maven2

fetch() {
  local name
  name=$(basename "$1")
  if [ -f "tmp/jars/$name" ]; then
    echo "have   $name"
  else
    echo "fetch  $name"
    curl -sSfL -o "tmp/jars/$name" "$central/$1"
  fi
}

fetch ml/dmlc/xgboost4j-spark_2.13/3.4.0/xgboost4j-spark_2.13-3.4.0.jar
fetch com/linkedin/isolation-forest/isolation-forest_4.1.1_2.13/4.1.8/isolation-forest_4.1.1_2.13-4.1.8.jar
fetch org/apache/spark/spark-avro_2.13/4.2.0/spark-avro_2.13-4.2.0.jar

ls -la tmp/jars
