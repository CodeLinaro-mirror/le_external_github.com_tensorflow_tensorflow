"""Package groups for XLA CPU internal."""

def xla_cpu_internal_packages(name = "xla_cpu_internal_packages"):
    native.package_group(
        name = "legacy_cpu_client_users",
        packages = ["//..."],
    )
