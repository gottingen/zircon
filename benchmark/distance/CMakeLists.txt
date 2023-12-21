

carbin_cc_benchmark(
        NAME
        l1_benchmark
        SOURCES
        "l1_benchmark.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        ${CARBIN_DEPS_LINK}
        zircon::zircon
        benchmark::benchmark
        benchmark::benchmark_main
)

carbin_cc_benchmark(
        NAME
        l2_benchmark
        SOURCES
        "l2_benchmark.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        ${CARBIN_DEPS_LINK}
        zircon::zircon
        benchmark::benchmark
        benchmark::benchmark_main
)