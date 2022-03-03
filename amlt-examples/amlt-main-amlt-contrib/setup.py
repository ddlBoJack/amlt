from setuptools import setup, find_namespace_packages
from setuptools_scm import get_version

version = get_version(root="..", relative_to=__file__)
version01 = ".".join(version.split(".")[:2])

setup(
  name="amlt-contrib",
  use_scm_version=dict(root="..", relative_to=__file__),
  packages=find_namespace_packages(),
  install_requires=[f"amlt>={version01}.0rc0"],
  entry_points="""
        [amlt.cli_plugins]
        aml-runs=amlt_aml_runs.cli:aml_runs
        job-id-map=amlt_job_id_map.cli:job_id_map
        cleanup=amlt_cleanup.cli:cleanup
        cleanup-job=amlt_cleanup.cli:cleanup_job
        perf-summary=amlt_perf_summary.cli:perf_summary
      """,
)
