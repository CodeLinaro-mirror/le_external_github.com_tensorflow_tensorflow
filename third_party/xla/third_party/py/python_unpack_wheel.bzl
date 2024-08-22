""" Repository rule to unpack a wheel. """

def _unpacked_wheel_impl(ctx):
    output_dir = ctx.actions.declare_directory(ctx.label.name)
    ctx.actions.run(
        inputs = [ctx.file.wheel],
        outputs = [output_dir],
        executable = ctx.executable.zipper,
        arguments = ["x", ctx.file.wheel.path, "-d", output_dir.path],
    )
    return [
        DefaultInfo(files = depset([output_dir])),
    ]

unpacked_wheel = rule(
    implementation = _unpacked_wheel_impl,
    attrs = {
        "wheel": attr.label(mandatory = True, allow_single_file = True),
        "zipper": attr.label(
            default = Label("@bazel_tools//tools/zip:zipper"),
            cfg = "exec",
            executable = True,
        ),
    },
)
