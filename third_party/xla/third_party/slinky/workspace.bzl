"""slinky is a lightweight runtime for semi-automatical optimization of data flow pipelines for locality."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "slinky",
        sha256 = "a9c37be402f2f508cc4626ffcc46f60a4002641915bead4374cd62b64b91ef17",
        strip_prefix = "slinky-7203ab761bc2e7d744f8b91cfdddab7a1e4ef1a3",
        urls = tf_mirror_urls("https://github.com/dsharlet/slinky/archive/7203ab761bc2e7d744f8b91cfdddab7a1e4ef1a3.zip"),
    )
